#include "neural_beamforming_pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace neural_beamforming {

NeuralBeamformingPipeline::NeuralBeamformingPipeline(const PipelineConfig& config)
    : config_(config) {
    
    if (config_.beamforming_params.num_antennas <= 0 || 
        config_.beamforming_params.num_users <= 0 ||
        config_.beamforming_params.num_subcarriers <= 0 ||
        config_.beamforming_params.num_ofdm_symbols <= 0) {
        throw std::invalid_argument("Invalid beamforming configuration");
    }
}

NeuralBeamformingPipeline::~NeuralBeamformingPipeline() {
    deallocate_buffers();
}

bool NeuralBeamformingPipeline::initialize() {
    try {
        // Create neural beamformer module
        beamformer_ = std::make_shared<NeuralBeamformer>(config_.module_id, config_.beamforming_params);

        auto requirements = beamformer_->get_requirements();
        module_tensor_bytes_ = requirements.device_tensor_bytes;
        if (module_tensor_bytes_ > 0) {
            cudaMalloc(&d_module_tensor_, module_tensor_bytes_);
            framework::pipeline::ModuleMemorySlice slice{};
            slice.device_tensor_ptr = reinterpret_cast<std::byte*>(d_module_tensor_);
            slice.device_tensor_bytes = module_tensor_bytes_;
            beamformer_->setup_memory(slice);
        }
        
        // Allocate internal buffers
        allocate_buffers();
        
        // Initialize performance tracking
        reset_metrics();
        
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to initialize neural beamforming pipeline: %s\n", e.what());
        return false;
    }
}

bool NeuralBeamformingPipeline::process_beamforming(
    const std::vector<std::complex<float>>& input_symbols,
    const std::vector<std::complex<float>>& channel_estimates,
    std::vector<std::complex<float>>& output_symbols,
    std::vector<std::complex<float>>& beamforming_weights,
    std::vector<float>& performance_metrics,
    cudaStream_t stream
) {
    if (!beamformer_) {
        fprintf(stderr, "Neural beamformer not initialized\n");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate input sizes
        size_t expected_input_symbols = calculate_input_symbols();
        size_t expected_channel_estimates = config_.beamforming_params.num_users * 
                                           config_.beamforming_params.num_antennas * 
                                           config_.beamforming_params.num_subcarriers;
        
        if (!validate_input_sizes(input_symbols.size(), channel_estimates.size())) {
            fprintf(stderr, "Invalid input sizes: input_symbols=%zu (expected %zu), channel_estimates=%zu (expected %zu)\n",
                   input_symbols.size(), expected_input_symbols,
                   channel_estimates.size(), expected_channel_estimates);
            return false;
        }
        
        // Resize output buffers
        size_t expected_output_symbols = calculate_output_symbols();
        size_t expected_weights = config_.beamforming_params.num_users * 
                                 config_.beamforming_params.num_antennas * 
                                 config_.beamforming_params.num_subcarriers;
        size_t expected_metrics = config_.beamforming_params.num_users * 
                                 config_.beamforming_params.num_subcarriers;
        
        output_symbols.resize(expected_output_symbols);
        beamforming_weights.resize(expected_weights);
        performance_metrics.resize(expected_metrics);
        
        // Copy data to GPU
        size_t input_symbols_bytes = input_symbols.size() * sizeof(std::complex<float>);
        size_t channel_estimates_bytes = channel_estimates.size() * sizeof(std::complex<float>);
        
        cudaError_t err = cudaMemcpyAsync(
            d_input_buffer_, 
            input_symbols.data(), 
            input_symbols_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy input symbols to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        err = cudaMemcpyAsync(
            static_cast<char*>(d_input_buffer_) + input_symbols_bytes,
            channel_estimates.data(), 
            channel_estimates_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy channel estimates to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Process on GPU
        bool success = process_device(
            d_input_buffer_,
            static_cast<char*>(d_input_buffer_) + input_symbols_bytes,
            d_output_buffer_,
            static_cast<char*>(d_output_buffer_) + expected_output_symbols * sizeof(std::complex<float>),
            static_cast<char*>(d_output_buffer_) + expected_output_symbols * sizeof(std::complex<float>) + 
                expected_weights * sizeof(std::complex<float>),
            stream
        );
        if (!success) {
            return false;
        }
        
        // Copy results back to host
        size_t output_symbols_bytes = output_symbols.size() * sizeof(std::complex<float>);
        size_t weights_bytes = beamforming_weights.size() * sizeof(std::complex<float>);
        size_t metrics_bytes = performance_metrics.size() * sizeof(float);
        
        err = cudaMemcpyAsync(
            output_symbols.data(), 
            d_output_buffer_, 
            output_symbols_bytes, 
            cudaMemcpyDeviceToHost, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy output symbols from GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        err = cudaMemcpyAsync(
            beamforming_weights.data(),
            static_cast<char*>(d_output_buffer_) + output_symbols_bytes,
            weights_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy beamforming weights from GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        err = cudaMemcpyAsync(
            performance_metrics.data(),
            static_cast<char*>(d_output_buffer_) + output_symbols_bytes + weights_bytes,
            metrics_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy performance metrics from GPU: %s\n", cudaGetErrorString(err));
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
            
            // Calculate average SINR
            double avg_sinr = 0.0;
            for (float sinr : performance_metrics) {
                avg_sinr += sinr;
            }
            avg_sinr /= performance_metrics.size();
            
            update_metrics(duration.count() / 1000.0, output_symbols.size(), avg_sinr);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "Beamforming processing failed: %s\n", e.what());
        return false;
    }
}

bool NeuralBeamformingPipeline::process_device(
    const void* d_input_symbols,
    const void* d_channel_estimates,
    void* d_output_symbols,
    void* d_beamforming_weights,
    void* d_performance_metrics,
    cudaStream_t stream
) {
    if (!beamformer_) {
        fprintf(stderr, "Neural beamformer not initialized\n");
        return false;
    }
    
    try {
        // Set up input port information
        std::vector<framework::pipeline::PortInfo> inputs(2);
        
        // Input symbols
        inputs[0].name = "input_symbols";
        inputs[0].tensors.resize(1);
        inputs[0].tensors[0].device_ptr = const_cast<void*>(d_input_symbols);
        inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(config_.beamforming_params.num_antennas),
                static_cast<std::size_t>(config_.beamforming_params.num_subcarriers),
                static_cast<std::size_t>(config_.beamforming_params.num_ofdm_symbols)
            }
        );
        
        // Channel estimates
        inputs[1].name = "channel_estimates";
        inputs[1].tensors.resize(1);
        inputs[1].tensors[0].device_ptr = const_cast<void*>(d_channel_estimates);
        inputs[1].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(config_.beamforming_params.num_users),
                static_cast<std::size_t>(config_.beamforming_params.num_antennas),
                static_cast<std::size_t>(config_.beamforming_params.num_subcarriers)
            }
        );
        
        // Set inputs to beamformer
        beamformer_->set_inputs(inputs);
        
        // Execute processing
        beamformer_->execute(stream);
        
        // Get outputs and copy to provided buffers
        auto outputs = beamformer_->get_outputs();
        if (outputs.size() >= 3) {
            // Copy output symbols
            if (!outputs[0].tensors.empty()) {
                size_t output_symbols_bytes = get_output_symbols_size();
                cudaError_t err = cudaMemcpyAsync(
                    d_output_symbols,
                    outputs[0].tensors[0].device_ptr,
                    output_symbols_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to copy beamformed symbols: %s\n", cudaGetErrorString(err));
                    return false;
                }
            }
            
            // Copy beamforming weights
            if (!outputs[1].tensors.empty()) {
                size_t weights_bytes = get_beamforming_weights_size();
                cudaError_t err = cudaMemcpyAsync(
                    d_beamforming_weights,
                    outputs[1].tensors[0].device_ptr,
                    weights_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to copy beamforming weights: %s\n", cudaGetErrorString(err));
                    return false;
                }
            }
            
            // Copy performance metrics
            if (!outputs[2].tensors.empty()) {
                size_t metrics_bytes = get_performance_metrics_size();
                cudaError_t err = cudaMemcpyAsync(
                    d_performance_metrics,
                    outputs[2].tensors[0].device_ptr,
                    metrics_bytes,
                    cudaMemcpyDeviceToDevice,
                    stream
                );
                if (err != cudaSuccess) {
                    fprintf(stderr, "Failed to copy performance metrics: %s\n", cudaGetErrorString(err));
                    return false;
                }
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU beamforming processing failed: %s\n", e.what());
        return false;
    }
}

bool NeuralBeamformingPipeline::update_parameters(const BeamformingParams& new_params) {
    if (new_params.num_antennas != config_.beamforming_params.num_antennas ||
        new_params.num_users != config_.beamforming_params.num_users ||
        new_params.num_subcarriers != config_.beamforming_params.num_subcarriers ||
        new_params.num_ofdm_symbols != config_.beamforming_params.num_ofdm_symbols) {
        
        // Parameters that affect memory layout require re-initialization
        config_.beamforming_params = new_params;
        deallocate_buffers();
        
        try {
            beamformer_ = std::make_shared<NeuralBeamformer>(config_.module_id, config_.beamforming_params);
            allocate_buffers();
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Failed to update beamforming parameters: %s\n", e.what());
            return false;
        }
    } else {
        // Parameters that don't affect memory layout can be updated directly
        config_.beamforming_params = new_params;
        return true;
    }
}

double NeuralBeamformingPipeline::calculate_theoretical_gain(int num_antennas, BeamformingAlgorithm algorithm) const {
    switch (algorithm) {
        case BeamformingAlgorithm::CONVENTIONAL:
            // Array gain: 10*log10(N)
            return 10.0 * std::log10(static_cast<double>(num_antennas));
        
        case BeamformingAlgorithm::MVDR:
            // MVDR provides optimal SNR, typically 1-2 dB better than conventional
            return 10.0 * std::log10(static_cast<double>(num_antennas)) + 1.5;
        
        case BeamformingAlgorithm::ZERO_FORCING:
            // ZF can provide interference nulling but may suffer noise enhancement
            return 10.0 * std::log10(static_cast<double>(num_antennas)) + 0.5;
        
        case BeamformingAlgorithm::NEURAL_NETWORK:
            // Neural networks can potentially outperform classical algorithms
            return 10.0 * std::log10(static_cast<double>(num_antennas)) + 2.0;
        
        default:
            return 0.0;
    }
}

NeuralBeamformingPipeline::PerformanceMetrics NeuralBeamformingPipeline::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void NeuralBeamformingPipeline::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = {};
    start_time_ = std::chrono::high_resolution_clock::now();
}

void NeuralBeamformingPipeline::get_steering_vector(
    float angle_degrees,
    std::vector<std::complex<float>>& steering_vector
) const {
    steering_vector.resize(config_.beamforming_params.num_antennas);
    
    // Uniform linear array steering vector
    float angle_rad = angle_degrees * M_PI / 180.0f;
    float k = 2.0f * M_PI; // Assume normalized frequency
    float d = 0.5f; // Half-wavelength spacing
    
    for (int i = 0; i < config_.beamforming_params.num_antennas; ++i) {
        float phase = k * d * i * std::sin(angle_rad);
        steering_vector[i] = std::complex<float>(std::cos(phase), std::sin(phase));
    }
}

size_t NeuralBeamformingPipeline::calculate_input_symbols() const {
    if (beamformer_) {
        return beamformer_->calculate_input_symbols();
    }
    
    return static_cast<size_t>(config_.beamforming_params.num_antennas * 
                              config_.beamforming_params.num_subcarriers * 
                              config_.beamforming_params.num_ofdm_symbols);
}

size_t NeuralBeamformingPipeline::calculate_output_symbols() const {
    if (beamformer_) {
        return beamformer_->calculate_output_symbols();
    }
    
    return static_cast<size_t>(config_.beamforming_params.num_users * 
                              config_.beamforming_params.num_subcarriers * 
                              config_.beamforming_params.num_ofdm_symbols);
}

void NeuralBeamformingPipeline::allocate_buffers() {
    size_t input_symbols_bytes = get_input_symbols_size();
    size_t channel_estimates_bytes = get_channel_estimates_size();
    size_t output_symbols_bytes = get_output_symbols_size();
    size_t weights_bytes = get_beamforming_weights_size();
    size_t metrics_bytes = get_performance_metrics_size();
    
    // Calculate total buffer sizes
    size_t total_input_bytes = input_symbols_bytes + channel_estimates_bytes;
    size_t total_output_bytes = output_symbols_bytes + weights_bytes + metrics_bytes;
    
    // Allocate host pinned memory for faster transfers
    cudaError_t err = cudaMallocHost(&h_input_buffer_, total_input_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host input buffer");
    }
    
    err = cudaMallocHost(&h_output_buffer_, total_output_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host output buffer");
    }
    
    // Allocate device memory
    err = cudaMalloc(&d_input_buffer_, total_input_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device input buffer");
    }
    
    err = cudaMalloc(&d_output_buffer_, total_output_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device output buffer");
    }
}

void NeuralBeamformingPipeline::deallocate_buffers() {
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
    if (d_module_tensor_) {
        cudaFree(d_module_tensor_);
        d_module_tensor_ = nullptr;
    }
    module_tensor_bytes_ = 0;
}

void NeuralBeamformingPipeline::update_metrics(double processing_time_ms, size_t num_symbols, double avg_sinr_db) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.total_processed_frames++;
    metrics_.peak_processing_time_ms = std::max(metrics_.peak_processing_time_ms, processing_time_ms);
    
    // Update running average
    double alpha = 0.1; // Smoothing factor
    if (metrics_.total_processed_frames == 1) {
        metrics_.avg_processing_time_ms = processing_time_ms;
        metrics_.avg_sinr_db = avg_sinr_db;
    } else {
        metrics_.avg_processing_time_ms = alpha * processing_time_ms + 
                                         (1.0 - alpha) * metrics_.avg_processing_time_ms;
        metrics_.avg_sinr_db = alpha * avg_sinr_db + (1.0 - alpha) * metrics_.avg_sinr_db;
    }
    
    // Calculate throughput (rough estimate)
    double symbols_per_frame = static_cast<double>(num_symbols);
    double symbols_per_second = symbols_per_frame * 1000.0 / processing_time_ms;
    metrics_.throughput_mbps = (symbols_per_second * 32.0) / (1024.0 * 1024.0); // 32 bits per complex symbol
    
    // Calculate beamforming gain
    metrics_.beamforming_gain_db = calculate_theoretical_gain(
        config_.beamforming_params.num_antennas, 
        config_.beamforming_params.algorithm
    );
    
    // Update total processed symbols
    metrics_.total_beamformed_symbols += num_symbols;
}

size_t NeuralBeamformingPipeline::get_input_symbols_size() const {
    return calculate_input_symbols() * sizeof(std::complex<float>);
}

size_t NeuralBeamformingPipeline::get_channel_estimates_size() const {
    return static_cast<size_t>(config_.beamforming_params.num_users * 
                              config_.beamforming_params.num_antennas * 
                              config_.beamforming_params.num_subcarriers) * sizeof(std::complex<float>);
}

size_t NeuralBeamformingPipeline::get_output_symbols_size() const {
    return calculate_output_symbols() * sizeof(std::complex<float>);
}

size_t NeuralBeamformingPipeline::get_beamforming_weights_size() const {
    return static_cast<size_t>(config_.beamforming_params.num_users * 
                              config_.beamforming_params.num_antennas * 
                              config_.beamforming_params.num_subcarriers) * sizeof(std::complex<float>);
}

size_t NeuralBeamformingPipeline::get_performance_metrics_size() const {
    return static_cast<size_t>(config_.beamforming_params.num_users * 
                              config_.beamforming_params.num_subcarriers) * sizeof(float);
}

bool NeuralBeamformingPipeline::validate_input_sizes(size_t input_symbols_size, size_t channel_estimates_size) const {
    size_t expected_input_symbols = calculate_input_symbols();
    size_t expected_channel_estimates = config_.beamforming_params.num_users * 
                                       config_.beamforming_params.num_antennas * 
                                       config_.beamforming_params.num_subcarriers;
    
    return (input_symbols_size == expected_input_symbols) && 
           (channel_estimates_size == expected_channel_estimates);
}

} // namespace neural_beamforming