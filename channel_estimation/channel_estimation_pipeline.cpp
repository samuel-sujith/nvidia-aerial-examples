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

void ChannelEstimationPipeline::setup() {
    // Create the channel estimator module
    std::string module_id = pipeline_id_ + "_channel_estimator";
    channel_estimator_ = std::make_unique<ChannelEstimator>(module_id, channel_params_);
    
    // Allocate pipeline memory
    allocate_pipeline_memory();
    
    // Setup module memory slice
    framework::pipeline::ModuleMemorySlice memory_slice;
    // In a real implementation, this would provide the proper memory slice
    
    channel_estimator_->setup(memory_slice);
    
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
    
    // Get tensor info from external ports
    std::vector<framework::tensor::TensorInfo> module_inputs;
    std::vector<framework::tensor::TensorInfo> module_outputs;
    
    // Map external inputs to module inputs
    for (const auto& port : external_inputs) {
        module_inputs.push_back(port.tensor_info);
    }
    
    // Map external outputs to module outputs  
    for (auto& port : external_outputs) {
        module_outputs.push_back(port.tensor_info);
    }
    
    // Configure the channel estimator module
    channel_estimator_->configure_io(params, module_inputs, module_outputs, stream);
    
    // Update external output port information
    auto output_tensor_info = channel_estimator_->get_output_tensor_info();
    for (std::size_t i = 0; i < external_outputs.size() && i < output_tensor_info.size(); ++i) {
        external_outputs[i].tensor_info = output_tensor_info[i];
        // In a real implementation, set external_outputs[i].device_ptr
    }
}

void ChannelEstimationPipeline::execute(cudaStream_t stream) {
    if (channel_estimator_) {
        channel_estimator_->execute(stream);
    }
}

void ChannelEstimationPipeline::allocate_pipeline_memory() {
    // Calculate memory requirements
    auto memory_req = channel_estimator_->get_memory_requirements();
    
    // For simplicity, allocate a fixed amount of memory
    memory_size_ = 64 * 1024 * 1024; // 64MB
    
    cudaError_t err = cudaMalloc(&device_memory_, memory_size_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pipeline device memory");
    }
}

void ChannelEstimationPipeline::setup_tensor_connections() {
    // Setup internal tensor connections between modules
    // For this simple pipeline with one module, no internal connections needed
    
    // In a multi-module pipeline, this would connect outputs of one module
    // to inputs of another module
}

} // namespace channel_estimation