/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "path_loss_prediction_pipeline.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace path_loss_prediction {

PathLossPredictionPipeline::PathLossPredictionPipeline(const PipelineConfig& config)
    : config_(config) {}

PathLossPredictionPipeline::~PathLossPredictionPipeline() {
    deallocate_buffers();
}

bool PathLossPredictionPipeline::initialize() {
    try {
        module_ = std::make_shared<PathLossPredictionModule>(
            config_.module_id,
            config_.model_params);

        auto requirements = module_->get_requirements();
        module_tensor_bytes_ = requirements.device_tensor_bytes;
        static_desc_bytes_ = requirements.static_kernel_descriptor_bytes;
        dynamic_desc_bytes_ = requirements.dynamic_kernel_descriptor_bytes;

        if (module_tensor_bytes_ > 0) {
            cudaMalloc(&d_module_tensor_, module_tensor_bytes_);
            framework::pipeline::ModuleMemorySlice slice{};
            slice.device_tensor_ptr = reinterpret_cast<std::byte*>(d_module_tensor_);
            slice.device_tensor_bytes = module_tensor_bytes_;
            if (static_desc_bytes_ > 0) {
                cudaMallocHost(&static_desc_cpu_, static_desc_bytes_);
                cudaMalloc(&static_desc_gpu_, static_desc_bytes_);
                slice.static_kernel_descriptor_cpu_ptr = static_desc_cpu_;
                slice.static_kernel_descriptor_gpu_ptr = static_desc_gpu_;
                slice.static_kernel_descriptor_bytes = static_desc_bytes_;
            }
            if (dynamic_desc_bytes_ > 0) {
                cudaMallocHost(&dynamic_desc_cpu_, dynamic_desc_bytes_);
                cudaMalloc(&dynamic_desc_gpu_, dynamic_desc_bytes_);
                slice.dynamic_kernel_descriptor_cpu_ptr = dynamic_desc_cpu_;
                slice.dynamic_kernel_descriptor_gpu_ptr = dynamic_desc_gpu_;
                slice.dynamic_kernel_descriptor_bytes = dynamic_desc_bytes_;
            }
            module_->setup_memory(slice);
        }

        allocate_buffers();
        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to initialize path loss prediction pipeline: %s\n", e.what());
        return false;
    }
}

bool PathLossPredictionPipeline::predict(
    const std::vector<float>& features,
    std::vector<float>& path_loss_db,
    cudaStream_t stream) {
    if (!module_) {
        fprintf(stderr, "Path loss prediction module not initialized\n");
        return false;
    }

    const size_t expected_size =
        static_cast<size_t>(config_.model_params.batch_size) *
        static_cast<size_t>(config_.model_params.num_features);

    if (features.size() != expected_size) {
        fprintf(stderr, "Invalid feature size: %zu, expected %zu\n",
                features.size(), expected_size);
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    std::copy(features.begin(), features.end(), h_features_);
    cudaMemcpyAsync(d_features_, h_features_,
                    expected_size * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    std::vector<framework::pipeline::PortInfo> inputs(1);
    inputs[0].name = "features";
    inputs[0].tensors.resize(1);
    inputs[0].tensors[0].device_ptr = d_features_;
    inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(config_.model_params.batch_size),
         static_cast<size_t>(config_.model_params.num_features)});

    module_->set_inputs(inputs);
    module_->configure_io({}, stream);
    module_->execute(stream);

    auto outputs = module_->get_outputs();
    if (outputs.empty() || outputs[0].tensors.empty()) {
        fprintf(stderr, "Path loss prediction produced no output\n");
        return false;
    }

    const size_t output_count = static_cast<size_t>(config_.model_params.batch_size);
    path_loss_db.resize(output_count);

    cudaMemcpyAsync(h_outputs_, outputs[0].tensors[0].device_ptr,
                    output_count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA stream synchronization failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    std::copy(h_outputs_, h_outputs_ + output_count, path_loss_db.begin());

    if (config_.enable_profiling) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        update_metrics(duration.count() / 1000.0, output_count);
    }

    return true;
}

PathLossPredictionPipeline::PerformanceMetrics
PathLossPredictionPipeline::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void PathLossPredictionPipeline::allocate_buffers() {
    const size_t feature_bytes =
        static_cast<size_t>(config_.model_params.batch_size) *
        static_cast<size_t>(config_.model_params.num_features) * sizeof(float);
    const size_t output_bytes =
        static_cast<size_t>(config_.model_params.batch_size) * sizeof(float);

    cudaMallocHost(&h_features_, feature_bytes);
    cudaMallocHost(&h_outputs_, output_bytes);
    cudaMalloc(&d_features_, feature_bytes);
}

void PathLossPredictionPipeline::deallocate_buffers() {
    if (h_features_) {
        cudaFreeHost(h_features_);
        h_features_ = nullptr;
    }
    if (h_outputs_) {
        cudaFreeHost(h_outputs_);
        h_outputs_ = nullptr;
    }
    if (d_features_) {
        cudaFree(d_features_);
        d_features_ = nullptr;
    }

    if (d_module_tensor_) {
        cudaFree(d_module_tensor_);
        d_module_tensor_ = nullptr;
    }
    if (static_desc_cpu_) {
        cudaFreeHost(static_desc_cpu_);
        static_desc_cpu_ = nullptr;
    }
    if (static_desc_gpu_) {
        cudaFree(static_desc_gpu_);
        static_desc_gpu_ = nullptr;
    }
    if (dynamic_desc_cpu_) {
        cudaFreeHost(dynamic_desc_cpu_);
        dynamic_desc_cpu_ = nullptr;
    }
    if (dynamic_desc_gpu_) {
        cudaFree(dynamic_desc_gpu_);
        dynamic_desc_gpu_ = nullptr;
    }

    module_tensor_bytes_ = 0;
    static_desc_bytes_ = 0;
    dynamic_desc_bytes_ = 0;
}

void PathLossPredictionPipeline::update_metrics(double processing_time_ms, size_t num_samples) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.total_processed_batches++;
    metrics_.avg_processing_time_ms =
        (metrics_.avg_processing_time_ms * (metrics_.total_processed_batches - 1) + processing_time_ms) /
        metrics_.total_processed_batches;
    metrics_.peak_processing_time_ms = std::max(metrics_.peak_processing_time_ms, processing_time_ms);
    metrics_.throughput_samples_per_ms = num_samples / std::max(processing_time_ms, 1e-6);
}

} // namespace path_loss_prediction
