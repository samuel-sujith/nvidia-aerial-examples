#include "neural_beamforming_pipeline_simple.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>

namespace neural_beamforming {

NeuralBeamformingPipeline::NeuralBeamformingPipeline(const Config& config)
    : config_(config) {
    std::cout << "Creating neural beamforming pipeline with " << config.num_antennas 
              << " antennas, " << config.num_users << " users\n";
}

NeuralBeamformingPipeline::~NeuralBeamformingPipeline() {
    finalize();
    std::cout << "Neural beamforming pipeline destroyed\n";
}

bool NeuralBeamformingPipeline::initialize() {
    if (is_initialized_) {
        return true;
    }
    
    try {
        initialize_cuda_resources();
        
        // Allocate GPU memory
        size_t channel_size = config_.num_antennas * config_.num_users * sizeof(std::complex<float>);
        size_t weights_size = config_.num_antennas * sizeof(std::complex<float>);
        size_t temp_size = config_.batch_size * config_.num_users * sizeof(std::complex<float>);
        
        cudaMalloc(&d_channel_matrix_, channel_size);
        cudaMalloc(&d_beamforming_weights_, weights_size);
        cudaMalloc(&d_temp_buffer_, temp_size);
        
        is_initialized_ = true;
        std::cout << "Neural beamforming pipeline initialized successfully\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize neural beamforming pipeline: " << e.what() << "\n";
        return false;
    }
}

void NeuralBeamformingPipeline::finalize() {
    if (!is_initialized_) {
        return;
    }
    
    cleanup_cuda_resources();
    is_initialized_ = false;
    std::cout << "Neural beamforming pipeline finalized\n";
}

bool NeuralBeamformingPipeline::process_beamforming(
    const std::vector<std::complex<float>>& channel_input,
    const std::vector<std::complex<float>>& user_signals,
    std::vector<std::complex<float>>& beamformed_output) {
    
    if (!is_initialized_) {
        std::cerr << "Pipeline not initialized\n";
        return false;
    }
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Resize output vector
    beamformed_output.resize(config_.num_users);
    
    // Simple stub implementation - just copy and scale input
    size_t min_size = std::min(channel_input.size(), user_signals.size());
    for (size_t i = 0; i < config_.num_users && i < min_size; ++i) {
        beamformed_output[i] = channel_input[i] * user_signals[i];
    }
    
    // Update metrics
    auto end_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    metrics_.total_latency_ms = elapsed.count() / 1000.0f;
    metrics_.processed_samples++;
    metrics_.signal_to_interference_ratio = 15.0f; // Stub value
    metrics_.spectral_efficiency = 4.5f; // Stub value
    
    return true;
}

void NeuralBeamformingPipeline::initialize_cuda_resources() {
    cudaStreamCreate(&stream_);
    cublasCreate(&cublas_handle_);
    cublasSetStream(cublas_handle_, stream_);
    std::cout << "CUDA resources initialized for neural beamforming\n";
}

void NeuralBeamformingPipeline::cleanup_cuda_resources() {
    if (d_channel_matrix_) { cudaFree(d_channel_matrix_); d_channel_matrix_ = nullptr; }
    if (d_beamforming_weights_) { cudaFree(d_beamforming_weights_); d_beamforming_weights_ = nullptr; }
    if (d_temp_buffer_) { cudaFree(d_temp_buffer_); d_temp_buffer_ = nullptr; }
    
    if (cublas_handle_) { cublasDestroy(cublas_handle_); cublas_handle_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    
    std::cout << "CUDA resources cleaned up\n";
}

void NeuralBeamformingPipeline::compute_traditional_mmse() {
    // Stub implementation for traditional MMSE beamforming
    std::cout << "Computing traditional MMSE beamforming\n";
}

void NeuralBeamformingPipeline::run_neural_inference() {
    // Stub implementation for neural network inference
    std::cout << "Running neural network inference\n";
    metrics_.inference_latency_ms = 2.5f; // Stub value
}

// Factory implementation
std::unique_ptr<NeuralBeamformingPipeline> NeuralBeamformingPipelineFactory::create(
    const NeuralBeamformingPipeline::Config& config) {
    return std::make_unique<NeuralBeamformingPipeline>(config);
}

NeuralBeamformingPipeline::Config NeuralBeamformingPipeline::create_default_config(
    std::size_t num_antennas, std::size_t num_users) {
    Config config;
    config.num_antennas = num_antennas;
    config.num_users = num_users;
    config.batch_size = 32;
    config.mode = BeamformingMode::NEURAL_DNN;
    config.precision = ModelPrecision::FP32;
    config.snr_threshold_db = 10.0f;
    config.history_length = 10;
    return config;
}

} // namespace neural_beamforming