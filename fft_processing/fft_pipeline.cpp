#include "fft_pipeline.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>

namespace fft_processing {

FFTPipeline::FFTPipeline(const FFTPipelineConfig& config)
    : config_(config), stats_{}, is_initialized_(false) {
    // Initialize CUDA context to prevent issues later
    cudaError_t cuda_err = cudaSetDevice(config_.gpu_device_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Warning: Failed to set CUDA device " << config_.gpu_device_id 
                  << ": " << cudaGetErrorString(cuda_err) << std::endl;
    }
}

FFTPipeline::~FFTPipeline() {
    teardown();
}

bool FFTPipeline::setup(const ::framework::pipeline::PipelineSpec& spec) {
    if (is_initialized_) {
        return true;
    }

    if (!initialize_cuda_resources()) {
        return false;
    }

    is_initialized_ = true;
    return true;
}

void FFTPipeline::teardown() {
    if (!is_initialized_) {
        return;
    }

    cleanup_cuda_resources();
    is_initialized_ = false;
}

bool FFTPipeline::initialize_cuda_resources() {
    // Check CUDA device availability
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA devices available: " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    // Set device
    cuda_err = cudaSetDevice(config_.gpu_device_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error: Failed to set CUDA device " << config_.gpu_device_id 
                  << ": " << cudaGetErrorString(cuda_err) << std::endl;
        return false;
    }
    
    // Initialize cuFFT plans with error checking
    for (auto fft_size : config_.fft_sizes) {
        if (fft_size == 0 || (fft_size & (fft_size - 1)) != 0) {
            std::cerr << "Error: FFT size " << fft_size << " must be a power of 2 and > 0" << std::endl;
            return false;
        }
        
        cufftHandle plan;
        cufftResult_t cufft_result = cufftPlan1d(&plan, static_cast<int>(fft_size), CUFFT_C2C, 1);
        if (cufft_result != CUFFT_SUCCESS) {
            std::cerr << "Error: Failed to create cuFFT plan for size " << fft_size 
                      << ", error code: " << cufft_result << std::endl;
            cleanup_cuda_resources(); // Clean up any previously created plans
            return false;
        }
        
        fft_plans_[fft_size] = plan;
        ifft_plans_[fft_size] = plan; // Reuse same plan for simplicity
        
        std::cout << "Created cuFFT plan for FFT size: " << fft_size << std::endl;
    }
    
    // Synchronize to ensure initialization is complete
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        std::cerr << "Error: CUDA synchronization failed: " << cudaGetErrorString(cuda_err) << std::endl;
        cleanup_cuda_resources();
        return false;
    }
    
    return true;
}

void FFTPipeline::cleanup_cuda_resources() {
    // Clean up cuFFT plans
    for (auto& [size, plan] : fft_plans_) {
        cufftResult_t result = cufftDestroy(plan);
        if (result != CUFFT_SUCCESS) {
            std::cerr << "Warning: Failed to destroy cuFFT plan for size " << size 
                      << ", error code: " << result << std::endl;
        }
    }
    fft_plans_.clear();
    ifft_plans_.clear();
    
    // Synchronize and reset device
    cudaError_t cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        std::cerr << "Warning: CUDA synchronization failed during cleanup: " 
                  << cudaGetErrorString(cuda_err) << std::endl;
    }
}

bool FFTPipeline::is_fft_size_supported(size_t fft_size) const {
    return std::find(config_.fft_sizes.begin(), config_.fft_sizes.end(), fft_size) != config_.fft_sizes.end();
}

::framework::task::TaskResult FFTPipeline::execute_pipeline(
    std::span<const ::framework::tensor::TensorInfo> inputs,
    std::span<::framework::tensor::TensorInfo> outputs,
    const ::framework::task::CancellationToken& token) {
    
    if (!is_initialized_) {
        return ::framework::task::TaskResult(::framework::task::TaskStatus::Failed, 
                                           "Pipeline not initialized");
    }

    // Simple stub implementation
    stats_.total_batches_processed++;
    stats_.total_samples_processed += 1024; // Assume 1024 samples per batch
    
    return ::framework::task::TaskResult(::framework::task::TaskStatus::Completed);
}

// Factory implementation
std::unique_ptr<FFTPipeline> FFTPipelineFactory::create_pipeline(const FFTPipelineConfig& config) {
    return std::make_unique<FFTPipeline>(config);
}

FFTPipelineConfig FFTPipelineFactory::get_default_config(const std::vector<size_t>& fft_sizes) {
    FFTPipelineConfig config;
    config.fft_sizes = fft_sizes;
    config.max_batch_size = 1024;
    config.enable_cuda_graphs = true;
    return config;
}

} // namespace fft_processing