#include "mimo_detection_pipeline.hpp"

#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

namespace mimo_detection {

MIMODetectionPipeline::MIMODetectionPipeline(const PipelineConfig& config)
    : config_(config) {
    
    if (config_.mimo_params.num_tx_antennas <= 0 || config_.mimo_params.num_rx_antennas <= 0) {
        throw std::invalid_argument("Invalid antenna configuration");
    }
}

MIMODetectionPipeline::~MIMODetectionPipeline() {
    deallocate_buffers();
}

bool MIMODetectionPipeline::initialize() {
    try {
        // Create MIMO detector module
        detector_ = std::make_shared<MIMODetector>(config_.module_id, config_.mimo_params);

        auto requirements = detector_->get_requirements();
        module_tensor_bytes_ = requirements.device_tensor_bytes;
        if (module_tensor_bytes_ > 0) {
            cudaMalloc(&d_module_tensor_, module_tensor_bytes_);
            framework::pipeline::ModuleMemorySlice slice{};
            slice.device_tensor_ptr = reinterpret_cast<std::byte*>(d_module_tensor_);
            slice.device_tensor_bytes = module_tensor_bytes_;
            detector_->setup_memory(slice);
        }
        
        // Allocate internal buffers
        allocate_buffers();
        
        // Initialize performance tracking
        reset_metrics();
        
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to initialize MIMO detection pipeline: %s\n", e.what());
        return false;
    }
}

bool MIMODetectionPipeline::process(
    const std::vector<std::complex<float>>& received_symbols,
    const std::vector<std::complex<float>>& channel_matrix,
    std::vector<std::complex<float>>& detected_symbols,
    cudaStream_t stream
) {
    if (!detector_) {
        fprintf(stderr, "MIMO detector not initialized\n");
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate input sizes
        size_t expected_received_size = config_.mimo_params.num_rx_antennas * 
                                       config_.mimo_params.num_subcarriers * 
                                       config_.mimo_params.num_ofdm_symbols;
        size_t expected_channel_size = config_.mimo_params.num_rx_antennas * 
                                      config_.mimo_params.num_tx_antennas * 
                                      config_.mimo_params.num_subcarriers;
        
        if (received_symbols.size() != expected_received_size) {
            fprintf(stderr, "Invalid received symbols size: %zu, expected: %zu\n", 
                   received_symbols.size(), expected_received_size);
            return false;
        }
        
        if (channel_matrix.size() != expected_channel_size) {
            fprintf(stderr, "Invalid channel matrix size: %zu, expected: %zu\n", 
                   channel_matrix.size(), expected_channel_size);
            return false;
        }
        
        // Resize output buffer
        size_t output_size = config_.mimo_params.num_tx_antennas * 
                            config_.mimo_params.num_subcarriers * 
                            config_.mimo_params.num_ofdm_symbols;
        detected_symbols.resize(output_size);
        
        // Copy data to GPU
        size_t received_bytes = received_symbols.size() * sizeof(std::complex<float>);
        size_t channel_bytes = channel_matrix.size() * sizeof(std::complex<float>);
        
        cudaError_t err = cudaMemcpyAsync(
            d_received_buffer_, 
            received_symbols.data(), 
            received_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy received symbols to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        err = cudaMemcpyAsync(
            d_channel_buffer_, 
            channel_matrix.data(), 
            channel_bytes, 
            cudaMemcpyHostToDevice, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy channel matrix to GPU: %s\n", cudaGetErrorString(err));
            return false;
        }
        
        // Process on GPU
        bool success = process_device(d_received_buffer_, d_channel_buffer_, d_detected_buffer_, stream);
        if (!success) {
            return false;
        }
        
        // Copy result back to host
        size_t output_bytes = detected_symbols.size() * sizeof(std::complex<float>);
        err = cudaMemcpyAsync(
            detected_symbols.data(), 
            d_detected_buffer_, 
            output_bytes, 
            cudaMemcpyDeviceToHost, 
            stream
        );
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to copy detected symbols from GPU: %s\n", cudaGetErrorString(err));
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
            update_metrics(duration.count() / 1000.0);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "MIMO detection processing failed: %s\n", e.what());
        return false;
    }
}

bool MIMODetectionPipeline::process_device(
    const void* d_received_symbols,
    const void* d_channel_matrix,
    void* d_detected_symbols,
    cudaStream_t stream
) {
    if (!detector_) {
        fprintf(stderr, "MIMO detector not initialized\n");
        return false;
    }
    
    try {
        // Set up input port information
        std::vector<framework::pipeline::PortInfo> inputs(2);
        
        // Input 0: received_symbols
        inputs[0].name = "received_symbols";
        inputs[0].tensors.resize(1);
        inputs[0].tensors[0].device_ptr = const_cast<void*>(d_received_symbols);
        inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(config_.mimo_params.num_rx_antennas),
                static_cast<std::size_t>(config_.mimo_params.num_subcarriers),
                static_cast<std::size_t>(config_.mimo_params.num_ofdm_symbols)
            }
        );
        
        // Input 1: channel_matrix
        inputs[1].name = "channel_matrix";
        inputs[1].tensors.resize(1);
        inputs[1].tensors[0].device_ptr = const_cast<void*>(d_channel_matrix);
        inputs[1].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(config_.mimo_params.num_rx_antennas),
                static_cast<std::size_t>(config_.mimo_params.num_tx_antennas),
                static_cast<std::size_t>(config_.mimo_params.num_subcarriers)
            }
        );
        
        // Set inputs to detector
        detector_->set_inputs(inputs);
        
        // Execute detection
        detector_->execute(stream);
        
        // Get outputs and copy to provided buffer
        auto outputs = detector_->get_outputs();
        if (!outputs.empty() && !outputs[0].tensors.empty()) {
            size_t output_bytes = get_detected_symbols_size();
            cudaError_t err = cudaMemcpyAsync(
                d_detected_symbols,
                outputs[0].tensors[0].device_ptr,
                output_bytes,
                cudaMemcpyDeviceToDevice,
                stream
            );
            if (err != cudaSuccess) {
                fprintf(stderr, "Failed to copy detection results: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        fprintf(stderr, "GPU MIMO detection failed: %s\n", e.what());
        return false;
    }
}

bool MIMODetectionPipeline::update_parameters(const MIMOParams& new_params) {
    if (new_params.num_tx_antennas != config_.mimo_params.num_tx_antennas ||
        new_params.num_rx_antennas != config_.mimo_params.num_rx_antennas ||
        new_params.num_subcarriers != config_.mimo_params.num_subcarriers ||
        new_params.num_ofdm_symbols != config_.mimo_params.num_ofdm_symbols) {
        
        // Parameters that affect memory layout require re-initialization
        config_.mimo_params = new_params;
        deallocate_buffers();
        
        try {
            detector_ = std::make_shared<MIMODetector>(config_.module_id, config_.mimo_params);
            allocate_buffers();
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Failed to update MIMO parameters: %s\n", e.what());
            return false;
        }
    } else {
        // Parameters that don't affect memory layout can be updated directly
        config_.mimo_params = new_params;
        return true;
    }
}

MIMODetectionPipeline::PerformanceMetrics MIMODetectionPipeline::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void MIMODetectionPipeline::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = {};
    start_time_ = std::chrono::high_resolution_clock::now();
}

void MIMODetectionPipeline::allocate_buffers() {
    size_t received_bytes = get_received_symbols_size();
    size_t channel_bytes = get_channel_matrix_size();
    size_t detected_bytes = get_detected_symbols_size();
    
    // Allocate host pinned memory for faster transfers
    cudaError_t err = cudaMallocHost(&h_received_buffer_, received_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host memory for received symbols");
    }
    
    err = cudaMallocHost(&h_channel_buffer_, channel_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host memory for channel matrix");
    }
    
    err = cudaMallocHost(&h_detected_buffer_, detected_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate host memory for detected symbols");
    }
    
    // Allocate device memory
    err = cudaMalloc(&d_received_buffer_, received_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for received symbols");
    }
    
    err = cudaMalloc(&d_channel_buffer_, channel_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for channel matrix");
    }
    
    err = cudaMalloc(&d_detected_buffer_, detected_bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for detected symbols");
    }
}

void MIMODetectionPipeline::deallocate_buffers() {
    if (h_received_buffer_) {
        cudaFreeHost(h_received_buffer_);
        h_received_buffer_ = nullptr;
    }
    if (h_channel_buffer_) {
        cudaFreeHost(h_channel_buffer_);
        h_channel_buffer_ = nullptr;
    }
    if (h_detected_buffer_) {
        cudaFreeHost(h_detected_buffer_);
        h_detected_buffer_ = nullptr;
    }
    if (d_received_buffer_) {
        cudaFree(d_received_buffer_);
        d_received_buffer_ = nullptr;
    }
    if (d_channel_buffer_) {
        cudaFree(d_channel_buffer_);
        d_channel_buffer_ = nullptr;
    }
    if (d_detected_buffer_) {
        cudaFree(d_detected_buffer_);
        d_detected_buffer_ = nullptr;
    }
    if (d_module_tensor_) {
        cudaFree(d_module_tensor_);
        d_module_tensor_ = nullptr;
    }
    module_tensor_bytes_ = 0;
}

void MIMODetectionPipeline::update_metrics(double processing_time_ms) {
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
    
    // Calculate throughput (rough estimate based on data processed)
    size_t total_bytes = get_received_symbols_size() + get_channel_matrix_size() + get_detected_symbols_size();
    double mbytes_per_frame = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
    
    if (metrics_.avg_processing_time_ms > 0.0) {
        metrics_.throughput_mbps = (mbytes_per_frame * 1000.0) / metrics_.avg_processing_time_ms;
    }
}

size_t MIMODetectionPipeline::get_received_symbols_size() const {
    return config_.mimo_params.num_rx_antennas * 
           config_.mimo_params.num_subcarriers * 
           config_.mimo_params.num_ofdm_symbols * 
           sizeof(std::complex<float>);
}

size_t MIMODetectionPipeline::get_channel_matrix_size() const {
    return config_.mimo_params.num_rx_antennas * 
           config_.mimo_params.num_tx_antennas * 
           config_.mimo_params.num_subcarriers * 
           sizeof(std::complex<float>);
}

size_t MIMODetectionPipeline::get_detected_symbols_size() const {
    return config_.mimo_params.num_tx_antennas * 
           config_.mimo_params.num_subcarriers * 
           config_.mimo_params.num_ofdm_symbols * 
           sizeof(std::complex<float>);
}

} // namespace mimo_detection