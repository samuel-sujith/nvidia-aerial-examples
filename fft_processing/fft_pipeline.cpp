#include "fft_pipeline.hpp"
#include <iostream>

namespace fft_processing {

FFTPipeline::FFTPipeline(const FFTPipelineConfig& config)
    : config_(config), stats_{}, is_initialized_(false) {
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
    // Simplified CUDA initialization
    for (auto fft_size : config_.fft_sizes) {
        cufftHandle plan;
        if (cufftPlan1d(&plan, static_cast<int>(fft_size), CUFFT_C2C, 1) != CUFFT_SUCCESS) {
            return false;
        }
        fft_plans_[fft_size] = plan;
        ifft_plans_[fft_size] = plan; // Reuse same plan for simplicity
    }
    return true;
}

void FFTPipeline::cleanup_cuda_resources() {
    for (auto& [size, plan] : fft_plans_) {
        cufftDestroy(plan);
    }
    fft_plans_.clear();
    ifft_plans_.clear();
}

bool FFTPipeline::is_fft_size_supported(size_t fft_size) const {
    return std::find(config_.fft_sizes.begin(), config_.fft_sizes.end(), fft_size) != config_.fft_sizes.end();
}

::framework::task::TaskResult FFTPipeline::execute_pipeline(
    std::span<const ::framework::tensor::TensorInfo> inputs,
    std::span<::framework::tensor::TensorInfo> outputs,
    const ::framework::task::CancellationToken& token) {
    
    if (!is_initialized_) {
        return {false, "Pipeline not initialized"};
    }

    // Simple stub implementation
    stats_.total_batches_processed++;
    stats_.total_samples_processed += 1024; // Assume 1024 samples per batch
    
    return {true, ""};
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