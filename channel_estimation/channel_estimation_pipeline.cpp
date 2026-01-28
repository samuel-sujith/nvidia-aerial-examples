/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_estimation_pipeline.hpp"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

namespace channel_estimation {

ChannelEstimationPipeline::ChannelEstimationPipeline(
    std::string pipeline_id,
    const ChannelEstParams& params
) : pipeline_id_(std::move(pipeline_id)), channel_params_(params), device_memory_(nullptr) {
    
}

ChannelEstimationPipeline::~ChannelEstimationPipeline() {
    if (device_memory_) {
        cudaFree(device_memory_);
        device_memory_ = nullptr;
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
}

void ChannelEstimationPipeline::setup() {
    // Create the channel estimator module
    std::string module_id = pipeline_id_ + "_channel_estimator";
    if (channel_params_.algorithm == ChannelEstAlgorithm::ML_TENSORRT) {
        std::cout << "[INFO] Using ML-based channel estimator (TensorRT): " << channel_params_.model_path << std::endl;
        channel_estimator_ = std::make_unique<MLChannelEstimatorTRT>(module_id, channel_params_);
    } else {
        channel_estimator_ = std::make_unique<ChannelEstimator>(module_id, channel_params_);
    }

    // Allocate pipeline memory
    allocate_pipeline_memory();

    // Setup module memory slice
    channel_estimator_->setup_memory(module_slice_);

    // Setup tensor connections
    setup_tensor_connections();
}

void ChannelEstimationPipeline::warmup(cudaStream_t stream) {
    if (channel_estimator_) {
        channel_estimator_->warmup(stream);
    }
}

void ChannelEstimationPipeline::configure_io(
    const framework::pipeline::DynamicParams& params,
    std::span<const framework::pipeline::PortInfo> external_inputs,
    std::span<framework::pipeline::PortInfo> external_outputs,
    cudaStream_t stream
) {
    // Validate external I/O
    if (external_inputs.size() != 2) {
        throw std::runtime_error("Channel estimation pipeline requires exactly 2 external inputs");
    }
    if (external_outputs.size() != 1) {
        throw std::runtime_error("Channel estimation pipeline requires exactly 1 external output");
    }
    
    // Set inputs and configure
    channel_estimator_->set_inputs(external_inputs);
    std::cout << "[DEBUG] Pipeline: set_inputs called with external inputs. Now calling configure_io..." << std::endl;
    channel_estimator_->configure_io(params, stream);
    // Get module outputs and update external outputs
    auto module_output_ports = channel_estimator_->get_outputs();
    for (size_t i = 0; i < external_outputs.size() && i < module_output_ports.size(); ++i) {
        external_outputs[i] = module_output_ports[i];
    }
}

void ChannelEstimationPipeline::execute_stream(cudaStream_t stream) {
    if (channel_estimator_) {
        channel_estimator_->execute(stream);
    }
}

void ChannelEstimationPipeline::execute_graph(cudaStream_t stream) {
    // For this example, graph mode is the same as stream mode
    // In a real implementation, this would use CUDA graphs
    if (channel_estimator_) {
        channel_estimator_->execute(stream);
    }
}

void ChannelEstimationPipeline::allocate_pipeline_memory() {
    auto requirements = channel_estimator_->get_requirements();
    memory_size_ = requirements.device_tensor_bytes;
    static_desc_bytes_ = requirements.static_kernel_descriptor_bytes;
    dynamic_desc_bytes_ = requirements.dynamic_kernel_descriptor_bytes;
    cudaError_t err = cudaMalloc(&device_memory_, memory_size_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pipeline device memory");
    }

    module_slice_ = {};
    module_slice_.device_tensor_ptr = reinterpret_cast<std::byte*>(device_memory_);
    module_slice_.device_tensor_bytes = memory_size_;
    if (static_desc_bytes_ > 0) {
        cudaMallocHost(&static_desc_cpu_, static_desc_bytes_);
        cudaMalloc(&static_desc_gpu_, static_desc_bytes_);
        module_slice_.static_kernel_descriptor_cpu_ptr = static_desc_cpu_;
        module_slice_.static_kernel_descriptor_gpu_ptr = static_desc_gpu_;
        module_slice_.static_kernel_descriptor_bytes = static_desc_bytes_;
    }
    if (dynamic_desc_bytes_ > 0) {
        cudaMallocHost(&dynamic_desc_cpu_, dynamic_desc_bytes_);
        cudaMalloc(&dynamic_desc_gpu_, dynamic_desc_bytes_);
        module_slice_.dynamic_kernel_descriptor_cpu_ptr = dynamic_desc_cpu_;
        module_slice_.dynamic_kernel_descriptor_gpu_ptr = dynamic_desc_gpu_;
        module_slice_.dynamic_kernel_descriptor_bytes = dynamic_desc_bytes_;
    }
}

void ChannelEstimationPipeline::setup_tensor_connections() {
    // Setup internal tensor connections between modules
    // For this simple pipeline with one module, no internal connections needed
    
    // In a multi-module pipeline, this would connect outputs of one module
    // to inputs of another module
}

} // namespace channel_estimation