/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_est_pipeline.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>

// Extensions for TensorInfo to provide missing API
namespace tensor_extensions {
    // Helper to get tensor size in bytes
    std::size_t size_bytes(const framework::tensor::TensorInfo& tensor) {
        auto dims = tensor.get_dimensions();
        std::size_t total = 1;
        for (auto dim : dims) {
            total *= dim;
        }
        return total * sizeof(cuComplex); // Assume complex float for now
    }
    
    // Helper to calculate tensor size - TensorInfo provides total_size_in_bytes method
}

namespace framework::examples {

ChannelEstimationPipeline::ChannelEstimationPipeline(
    std::string pipeline_id,
    gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
    const pipeline::PipelineSpec& spec
) : pipeline_id_(std::move(pipeline_id)), module_factory_(module_factory) {
    
    if (!setup(spec)) {
        throw std::runtime_error("Failed to setup channel estimation pipeline");
    }
}

bool ChannelEstimationPipeline::setup(const pipeline::PipelineSpec& spec) {
    try {
        // Setup memory pool for large tensor allocations
        setup_memory_pool(spec);
        
        // Create channel estimator module via factory
        nlohmann::json module_config;
        module_config["num_rx_antennas"] = 4;
        module_config["num_resource_blocks"] = 25;  
        module_config["pilot_spacing"] = 4;
        module_config["algorithm"] = "least_squares";
        
        auto module = module_factory_->create_module(
            pipeline_id_ + "_channel_estimator",
            "channel_estimator", 
            module_config
        );
        channel_estimator_ = std::unique_ptr<ChannelEstimator>(
            static_cast<ChannelEstimator*>(module.release())
        );
        
        // Setup CUDA graph for optimized execution
        setup_cuda_graph();
        
        // Initialize performance stats
        total_executions_ = 0;
        failed_executions_ = 0;
        
        return true;
    } catch (const std::exception& e) {
        // Log error in real implementation
        return false;
    }
}

void ChannelEstimationPipeline::setup_memory_pool(const pipeline::PipelineSpec& spec) {
    // Calculate memory requirements
    size_t max_resource_blocks = 273;  // 5G NR max
    size_t max_symbols = 14;           // Symbols per slot
    size_t subcarriers_per_rb = 12;
    size_t complex_size = sizeof(cuComplex);
    
    // Memory for channel estimates: RBs * subcarriers * symbols * antennas * layers
    size_t channel_memory = max_resource_blocks * subcarriers_per_rb * max_symbols * 
                           4 * 4 * complex_size; // Max 4 antennas, 4 layers
    
    // Memory for pilot symbols
    size_t pilot_memory = (max_resource_blocks * subcarriers_per_rb / 4) * complex_size;
    
    // Total memory with overhead
    size_t total_memory = (channel_memory + pilot_memory * 2) * 2; // Double buffer
    
    // Create framework memory pool
    memory::MemoryPoolConfig pool_config{};
    pool_config.device_id = 0;
    pool_config.initial_size = total_memory;
    pool_config.max_size = total_memory * 2;
    
    memory_pool_ = std::make_unique<memory::MemoryPool>(pool_config);
}

void ChannelEstimationPipeline::setup_cuda_graph() {
    // Create CUDA graph for optimized execution
    cudaError_t err = cudaGraphCreate(&cuda_graph_, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA graph");
    }
    graph_instantiated_ = false;
}

task::TaskResult ChannelEstimationPipeline::execute_pipeline(
    std::span<const tensor::TensorInfo> external_inputs,
    std::span<tensor::TensorInfo> external_outputs,
    const task::CancellationToken& token
) {
    return execute_internal(external_inputs, external_outputs, false);
}

task::TaskResult ChannelEstimationPipeline::execute_pipeline_graph(
    std::span<const tensor::TensorInfo> external_inputs,
    std::span<tensor::TensorInfo> external_outputs,
    const task::CancellationToken& token
) {
    return execute_internal(external_inputs, external_outputs, true);
}

task::TaskResult ChannelEstimationPipeline::execute_internal(
    std::span<const tensor::TensorInfo> inputs,
    std::span<tensor::TensorInfo> outputs,
    bool use_graph
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate inputs
        if (inputs.size() < 2) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                "Insufficient inputs: need rx_pilots and tx_pilots"
            );
        }
        
        if (outputs.size() < 1) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                "Insufficient outputs: need channel_estimates"
            );
        }
        
        // Validate tensor dimensions
        const auto& rx_tensor = inputs[0];
        const auto& tx_tensor = inputs[1]; 
        auto& output_tensor = outputs[0];
        
        // In real implementation, get params from module
        ChannelEstParams dummy_params{}; // This should come from the module
        
        if (!tensor_utils::validate_channel_est_tensors(
                rx_tensor, tx_tensor, output_tensor, dummy_params)) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                "Invalid tensor dimensions for channel estimation"
            );
        }
        
        // Convert to vector format expected by module
        std::vector<tensor::TensorInfo> input_vec{inputs.begin(), inputs.end()};
        std::vector<tensor::TensorInfo> output_vec{outputs.begin(), outputs.end()};
        
        // Execute channel estimation module
        task::CancellationToken dummy_token; // In real implementation, pass actual token
        auto result = channel_estimator_->execute(input_vec, output_vec, dummy_token);
        
        // Update performance stats
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time
        );
        
        stats_.total_executions++;
        
        if (result.is_success()) {
            // Successful execution
            stats_.total_execution_time_us += duration.count();
        } else {
            stats_.failed_executions++;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        stats_.failed_executions++;
        return task::TaskResult(
            task::TaskStatus::Failed,
            std::string("Pipeline execution failed: ") + e.what()
        );
    }
}

void ChannelEstimationPipeline::teardown() {
    // Cleanup CUDA graph
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
        cuda_graph_ = nullptr;
    }
    
    // Cleanup modules
    channel_estimator_.reset();
    
    // Cleanup memory pool
    memory_pool_.reset();
}

bool ChannelEstimationPipeline::is_ready() const {
bool ChannelEstimationPipeline::is_ready() const {
    return channel_estimator_ && memory_pool_;
}

pipeline::PipelineStats ChannelEstimationPipeline::get_stats() const {
    return stats_;
}

// Factory implementations
std::unique_ptr<pipeline::IPipeline> ChannelEstimationPipelineFactory::create_pipeline(
    const std::string& pipeline_id,
    gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
    const pipeline::PipelineSpec& spec
) {
    return std::make_unique<ChannelEstimationPipeline>(
        pipeline_id, module_factory, spec
    );
}

ChannelEstimatorModuleFactory::ChannelEstimatorModuleFactory(
    const ChannelEstParams& default_params
) : default_params_(default_params) {}

std::unique_ptr<pipeline::IModule> ChannelEstimatorModuleFactory::create_module(
    const std::string& module_id,
    const std::string& module_type,
    const nlohmann::json& config
) {
    if (module_type != "channel_estimator") {
        return nullptr;
    }
    
    ChannelEstParams params = default_params_;
    
    // Parse configuration
    if (config.contains("num_rx_antennas")) {
        params.num_rx_antennas = config["num_rx_antennas"];
    }
    if (config.contains("num_resource_blocks")) {
        params.num_resource_blocks = config["num_resource_blocks"];
    }
    if (config.contains("pilot_spacing")) {
        params.pilot_spacing = config["pilot_spacing"];
    }
    if (config.contains("algorithm")) {
        std::string algo = config["algorithm"];
        if (algo == "least_squares") {
            params.algorithm = ChannelEstAlgorithm::LEAST_SQUARES;
        } else if (algo == "mmse") {
            params.algorithm = ChannelEstAlgorithm::MMSE;
        }
    }
    
    return std::make_unique<ChannelEstimator>(module_id, params);
}

// Tensor utility implementations
namespace tensor_utils {

::framework::tensor::TensorInfo allocate_complex_tensor(
    const std::vector<std::size_t>& dimensions,
    size_t memory_size
) {
    // Stub implementation - would normally allocate from pool
    ::framework::tensor::TensorInfo tensor;
    // In real implementation:
    // - Calculate size from dimensions
    // - Allocate from pool
    // - Initialize tensor with data pointer and metadata
    return tensor;
}

cudaError_t copy_tensor_async(
    const ::framework::tensor::TensorInfo& src,
    ::framework::tensor::TensorInfo& dst,
    cudaStream_t stream
) {
    // Stub implementation - would copy tensor data
    // In real implementation would access tensor data pointers
    return cudaSuccess;
}

bool validate_channel_est_tensors(
    const ::framework::tensor::TensorInfo& rx_tensor,
    const ::framework::tensor::TensorInfo& tx_tensor,
    const ::framework::tensor::TensorInfo& output_tensor,
    const ChannelEstParams& params
) {
    // Validate rx tensor: [num_pilots]
    auto rx_dims = rx_tensor.get_dimensions();
    int expected_pilots = (params.num_resource_blocks * 12) / params.pilot_spacing;
    
    if (rx_dims.size() != 1 || rx_dims[0] != expected_pilots) {
        return false;
    }
    
    // Validate tx tensor: [num_pilots]  
    auto tx_dims = tx_tensor.get_dimensions();
    if (tx_dims.size() != 1 || tx_dims[0] != expected_pilots) {
        return false;
    }
    
    // Validate output tensor: [num_subcarriers, num_symbols]
    auto out_dims = output_tensor.get_dimensions();
    int expected_subcarriers = params.num_resource_blocks * 12;
    
    if (out_dims.size() != 2 || 
        out_dims[0] != expected_subcarriers ||
        out_dims[1] != params.num_ofdm_symbols) {
        return false;
    }
    
    // Skip data type validation for stub implementation
    return true;
}

} // namespace tensor_utils

} // namespace framework::examples