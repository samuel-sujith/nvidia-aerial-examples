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
    if (channel_params_.algorithm == ChannelEstAlgorithm::ML_TENSORRT) {
        // Placeholder: In a full implementation, instantiate MLChannelEstimatorTRT and integrate with pipeline
        std::cout << "[INFO] Using ML-based channel estimator (TensorRT): " << channel_params_.model_path << std::endl;
        // channel_estimator_ = std::make_unique<MLChannelEstimatorTRT>(...);
        // For now, fallback to base estimator to avoid breaking pipeline
        channel_estimator_ = std::make_unique<ChannelEstimator>(module_id, channel_params_);
    } else {
        channel_estimator_ = std::make_unique<ChannelEstimator>(module_id, channel_params_);
    }

    // Allocate pipeline memory
    allocate_pipeline_memory();

    // Setup module memory slice
    framework::pipeline::ModuleMemorySlice memory_slice;
    // In a real implementation, this would provide the proper memory slice

    channel_estimator_->setup_memory(memory_slice);

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
    
    // Combine all input and output ports, then set and configure once
    std::vector<framework::pipeline::PortInfo> all_ports;
    all_ports.reserve(external_inputs.size() + external_outputs.size());
    for (const auto& p : external_inputs) all_ports.push_back(p);
    for (const auto& p : external_outputs) all_ports.push_back(p);
    channel_estimator_->set_inputs(all_ports);
    std::cout << "[DEBUG] Pipeline: set_inputs called with all_ports. Now calling configure_io..." << std::endl;
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
    // For this example, we'll use a simple fixed memory allocation
    // In a real implementation, you would query module requirements
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