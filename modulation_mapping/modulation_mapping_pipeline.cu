#include "modulation_mapping_pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace modulation_mapping {

ModulationPipeline::ModulationPipeline(const PipelineConfig& config)
    : config_(config) {
    
    if (config_.modulation_params.num_subcarriers <= 0 || 
        config_.modulation_params.num_ofdm_symbols <= 0) {
        throw std::invalid_argument("Invalid modulation configuration");
    }
}

ModulationPipeline::~ModulationPipeline() {
    deallocate_buffers();
}

bool ModulationPipeline::initialize() {
    try {
        // Create modulation mapper module
        mapper_ = std::make_shared<ModulationMapper>(config_.module_id, config_.modulation_params);
        
        // Allocate internal buffers
        allocate_buffers();
        
        // Initialize performance tracking
        reset_metrics();
        
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to initialize modulation pipeline: %s\n", e.what());
        return false;
    }
}

bool ModulationPipeline::modulate(
    const std::vector<uint8_t>& input_bits,
    std::vector<std::complex<float>>& output_symbols,
    cudaStream_t stream
) {
    if (!mapper_) {
        fprintf(stderr, "Modulation mapper not initialized\n");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate input size
        size_t expected_bits = mapper_->calculate_total_bits();
        if (!validate_input_size(input_bits.size(), expected_bits)) {
            fprintf(stderr, "Invalid input bits size: %zu, expected: %zu\n", 
                   input_bits.size(), expected_bits);
            return false;
        }
        
        // Resize output buffer
        size_t expected_symbols = mapper_->calculate_total_symbols();
        output_symbols.resize(expected_symbols);
        
        // Copy data to GPU
        size_t bits_bytes = input_bits.size() * sizeof(uint8_t);
        cudaError_t err = cudaMemcpyAsync(
            d_input_buffer_, 
            input_bits.data(), 
            bits_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy input bits to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Process on GPU
        bool success = process_device(d_input_buffer_, d_output_buffer_, true, stream);
        if (!success) {
            return false;
        }
        
        // Copy result back to host
        size_t symbols_bytes = output_symbols.size() * sizeof(std::complex<float>);
        err = cudaMemcpyAsync(
            output_symbols.data(), 
            d_output_buffer_, 
            symbols_bytes, 
            cudaMemcpyDeviceToHost, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy output symbols from GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Wait for completion
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA stream synchronization failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Update performance metrics
        if (config_.enable_profiling) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            update_metrics(duration.count() / 1000.0, output_symbols.size());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Modulation processing failed: %s\n", e.what());
        return false;
    }
}

bool ModulationPipeline::demodulate(
    const std::vector<std::complex<float>>& input_symbols,
    std::vector<uint8_t>& output_bits,
    std::vector<float>* soft_bits,
    cudaStream_t stream
) {
    if (!mapper_) {
        fprintf(stderr, "Modulation mapper not initialized\n");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate input size
        size_t expected_symbols = mapper_->calculate_total_symbols();
        if (!validate_input_size(input_symbols.size(), expected_symbols)) {
            fprintf(stderr, "Invalid input symbols size: %zu, expected: %zu\n", 
                   input_symbols.size(), expected_symbols);
            return false;
        }
        
        // Resize output buffers
        size_t expected_bits = mapper_->calculate_total_bits();
        output_bits.resize(expected_bits);
        
        // Initialize output buffer to detect memory corruption
        std::fill(output_bits.begin(), output_bits.end(), 255);
        
        if (soft_bits && config_.modulation_params.soft_output) {
            soft_bits->resize(expected_bits);
        }
        
        // Debug: print first 4 input symbols before copying to GPU
        std::cout << "First 4 input symbols to demodulation: ";
        for (size_t i = 0; i < std::min<size_t>(4, input_symbols.size()); ++i) {
            std::cout << "(" << input_symbols[i].real() << ", " << input_symbols[i].imag() << ") ";
        }
        std::cout << std::endl;

        // Copy data to GPU
        size_t symbols_bytes = input_symbols.size() * sizeof(std::complex<float>);
        cudaError_t err = cudaMemcpyAsync(
            d_input_buffer_, 
            input_symbols.data(), 
            symbols_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy input symbols to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Ensure d_output_buffer_ points to the correct device output bits buffer for demodulation
        // This is a direct fix for the output buffer mismatch bug
        if (mapper_) {
            // Get the device pointer for output bits from the mapper
            auto outputs = mapper_->get_outputs();
            for (const auto& port : outputs) {
                if (port.name == "output_bits" && !port.tensors.empty()) {
                    d_output_buffer_ = port.tensors[0].device_ptr;
                }
            }
        }
        // Process on GPU
        bool success = process_device(d_input_buffer_, d_output_buffer_, false, stream);
        if (!success) {
            return false;
        }
        
        // Copy result back to host
        size_t bits_bytes = output_bits.size() * sizeof(uint8_t);
        err = cudaMemcpyAsync(
            output_bits.data(), 
            d_output_buffer_, 
            bits_bytes, 
            cudaMemcpyDeviceToHost, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy output bits from GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Copy soft bits if requested
        if (soft_bits && config_.modulation_params.soft_output) {
            // Note: This assumes soft bits are available from a separate output port
            // Implementation would need to be extended to handle soft bits properly
        }
        
        // Wait for completion
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA stream synchronization failed: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Update performance metrics
        if (config_.enable_profiling) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            update_metrics(duration.count() / 1000.0, input_symbols.size());
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Demodulation processing failed: %s\n", e.what());
        return false;
    }
}

bool ModulationPipeline::process_device(
    const void* d_input_data,
    void* d_output_data,
    bool is_modulation,
    cudaStream_t stream
) {
    if (!mapper_) {
        fprintf(stderr, "Modulation mapper not initialized\n");
        return false;
    }
    
    try {
        // Set up input port information based on operation
        std::vector<framework::pipeline::PortInfo> inputs(1);
        
        if (is_modulation) {
            // Modulation: input_bits
            inputs[0].name = "input_bits";
            inputs[0].tensors.resize(1);
            inputs[0].tensors[0].device_ptr = const_cast<void*>(d_input_data);
            inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::NvDataType::TensorR8U,
                std::vector<std::size_t>{static_cast<std::size_t>(mapper_->calculate_total_bits())}
            );
        } else {
            // Demodulation: input_symbols
            inputs[0].name = "input_symbols";
            inputs[0].tensors.resize(1);
            inputs[0].tensors[0].device_ptr = const_cast<void*>(d_input_data);
            inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::NvDataType::TensorC32F,
                std::vector<std::size_t>{
                    static_cast<std::size_t>(config_.modulation_params.num_subcarriers),
                    static_cast<std::size_t>(config_.modulation_params.num_ofdm_symbols)
                }
            );
        }
        
        // Set the processing mode based on operation
        if (is_modulation) {
            mapper_->set_processing_mode(modulation_mapping::ProcessingMode::MODULATION);
        } else {
            mapper_->set_processing_mode(modulation_mapping::ProcessingMode::DEMODULATION);
        }
        
        // Set inputs to mapper
        mapper_->set_inputs(inputs);
        
        // Execute processing
        mapper_->execute(stream);
        
        // Get outputs and copy to provided buffer
        auto outputs = mapper_->get_outputs();
        const void* output_ptr = nullptr;
        if (is_modulation) {
            // Modulation: use first output port (output_symbols)
            if (!outputs.empty() && !outputs[0].tensors.empty()) {
                output_ptr = outputs[0].tensors[0].device_ptr;
            }
        } else {
            // Demodulation: find port named 'output_bits'
            for (const auto& port : outputs) {
                if (port.name == "output_bits" && !port.tensors.empty()) {
                    output_ptr = port.tensors[0].device_ptr;
                    break;
                }
            }
        }
        if (output_ptr) {
            size_t output_bytes = is_modulation ? get_symbols_size() : get_bits_size();
            cudaError_t err = cudaMemcpyAsync(
                d_output_data,
                output_ptr,
                output_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            );
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy processing results: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU modulation processing failed: %s\n", e.what());
        return false;
    }
}

bool ModulationPipeline::update_parameters(const ModulationParams& new_params) {
    if (new_params.num_subcarriers != config_.modulation_params.num_subcarriers ||
        new_params.num_ofdm_symbols != config_.modulation_params.num_ofdm_symbols ||
        new_params.scheme != config_.modulation_params.scheme) {
        
        // Parameters that affect memory layout require re-initialization
        config_.modulation_params = new_params;
        deallocate_buffers();
        
        try {
            mapper_ = std::make_shared<ModulationMapper>(config_.module_id, config_.modulation_params);
            allocate_buffers();
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Failed to update modulation parameters: %s\n", e.what());
            return false;
        }
    } else {
        // Parameters that don't affect memory layout can be updated directly
        config_.modulation_params = new_params;
        return true;
    }
}

double ModulationPipeline::calculate_theoretical_ser(double snr_db) const {
    double snr_linear = std::pow(10.0, snr_db / 10.0);
    
    switch (config_.modulation_params.scheme) {
        case ModulationScheme::BPSK:
            return 0.5 * std::erfc(std::sqrt(snr_linear));
        
        case ModulationScheme::QPSK:
            return std::erfc(std::sqrt(snr_linear));
        
        case ModulationScheme::QAM16: {
            double sqrt_snr = std::sqrt(snr_linear / 10.0);
            return 3.0/2.0 * std::erfc(sqrt_snr);
        }
        
        case ModulationScheme::QAM64: {
            double sqrt_snr = std::sqrt(snr_linear / 42.0);
            return 7.0/3.0 * std::erfc(sqrt_snr);
        }
        
        default:
            return 0.0; // Unknown modulation
    }
}

ModulationPipeline::PerformanceMetrics ModulationPipeline::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void ModulationPipeline::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = {};
    start_time_ = std::chrono::high_resolution_clock::now();
}

void ModulationPipeline::get_constellation_diagram(
    std::vector<std::complex<float>>& points,
    std::vector<std::string>& labels
) const {
    points.clear();
    labels.clear();
    
    switch (config_.modulation_params.scheme) {
        case ModulationScheme::QPSK: {
            float norm = 1.0f / std::sqrt(2.0f);
            points = {
                {norm, norm}, {norm, -norm}, {-norm, norm}, {-norm, -norm}
            };
            labels = {"00", "01", "10", "11"};
            break;
        }
        
        case ModulationScheme::QAM16: {
            float norm = 1.0f / std::sqrt(10.0f);
            const float coords[] = {-3.0f, -1.0f, 1.0f, 3.0f};
            for (int i = 0; i < 4; ++i) {
                for (int q = 0; q < 4; ++q) {
                    points.emplace_back(coords[i] * norm, coords[q] * norm);
                    
                    // Generate Gray-coded labels
                    int bits = (i << 2) | q;
                    std::string label;
                    for (int b = 3; b >= 0; --b) {
                        label += ((bits >> b) & 1) ? '1' : '0';
                    }
                    labels.push_back(label);
                }
            }
            break;
        }
        
        default:
            // Default to QPSK
            float norm = 1.0f / std::sqrt(2.0f);
            points = {
                {norm, norm}, {norm, -norm}, {-norm, norm}, {-norm, -norm}
            };
            labels = {"00", "01", "10", "11"};
            break;
    }
}

size_t ModulationPipeline::calculate_total_bits() const {
    if (mapper_) {
        return mapper_->calculate_total_bits();
    }
    
    // Fallback calculation
    int bits_per_symbol = 2; // Default to QPSK
    switch (config_.modulation_params.scheme) {
        case ModulationScheme::BPSK: bits_per_symbol = 1; break;
        case ModulationScheme::QPSK: bits_per_symbol = 2; break;
        case ModulationScheme::QAM16: bits_per_symbol = 4; break;
        case ModulationScheme::QAM64: bits_per_symbol = 6; break;
        case ModulationScheme::QAM256: bits_per_symbol = 8; break;
    }
    
    return static_cast<size_t>(config_.modulation_params.num_subcarriers * 
                              config_.modulation_params.num_ofdm_symbols * 
                              bits_per_symbol);
}

size_t ModulationPipeline::calculate_total_symbols() const {
    if (mapper_) {
        return mapper_->calculate_total_symbols();
    }
    
    // Fallback calculation  
    return static_cast<size_t>(config_.modulation_params.num_subcarriers * 
                              config_.modulation_params.num_ofdm_symbols);
}

void ModulationPipeline::allocate_buffers() {
    size_t bits_bytes = get_bits_size();
    size_t symbols_bytes = get_symbols_size();
    
    // Determine maximum buffer size needed
    size_t max_bytes = std::max(bits_bytes, symbols_bytes);
    
    // Allocate host pinned memory for faster transfers
    cudaError_t err = cudaMallocHost(&h_input_buffer_, max_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host input buffer");
    }
    
    err = cudaMallocHost(&h_output_buffer_, max_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host output buffer");
    }
    
    // Allocate device memory
    err = cudaMalloc(&d_input_buffer_, max_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device input buffer");
    }
    
    err = cudaMalloc(&d_output_buffer_, max_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device output buffer");
    }
}

void ModulationPipeline::deallocate_buffers() {
    if (h_input_buffer_) {
        cudaFreeHost(h_input_buffer_);
        h_input_buffer_ = nullptr;
    }
    if (h_output_buffer_) {
        cudaFreeHost(h_output_buffer_);
        h_output_buffer_ = nullptr;
    }
    if (d_input_buffer_) {
        cudaFree(d_input_buffer_);
        d_input_buffer_ = nullptr;
    }
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
}

void ModulationPipeline::update_metrics(double processing_time_ms, size_t num_symbols, size_t num_errors) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_processed_frames++;
    metrics_.peak_processing_time_ms = std::max(metrics_.peak_processing_time_ms, processing_time_ms);
    
    // Update running average
    double alpha = 0.1; // Smoothing factor
    if (metrics_.total_processed_frames == 1) {
        metrics_.avg_processing_time_ms = processing_time_ms;
    } else {
        metrics_.avg_processing_time_ms = alpha * processing_time_ms + 
                                         (1.0 - alpha) * metrics_.avg_processing_time_ms;
    }
    
    // Calculate throughput (rough estimate based on symbols processed)
    double symbols_per_frame = static_cast<double>(num_symbols);
    double symbols_per_second = symbols_per_frame * 1000.0 / processing_time_ms;
    double bits_per_symbol = mapper_->get_bits_per_symbol();
    metrics_.throughput_mbps = (symbols_per_second * bits_per_symbol) / (1024.0 * 1024.0);
    
    // Update error statistics
    if (num_errors > 0) {
        metrics_.total_symbol_errors += num_errors;
        metrics_.symbol_error_rate = static_cast<double>(metrics_.total_symbol_errors) / 
                                    (metrics_.total_processed_frames * num_symbols);
    }
}

size_t ModulationPipeline::get_bits_size() const {
    return mapper_->calculate_total_bits() * sizeof(uint8_t);
}

size_t ModulationPipeline::get_symbols_size() const {
    return mapper_->calculate_total_symbols() * sizeof(std::complex<float>);
}

bool ModulationPipeline::validate_input_size(size_t input_size, size_t expected_size) const {
    return input_size == expected_size;
}

} // namespace modulation_mapping