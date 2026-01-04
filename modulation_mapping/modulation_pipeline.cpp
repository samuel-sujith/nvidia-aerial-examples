/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "modulation_pipeline.hpp"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace framework::examples {

ModulationPipeline::ModulationPipeline(
    std::string pipeline_id,
    gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
    const pipeline::PipelineSpec& spec
) : pipeline_id_(std::move(pipeline_id)), module_factory_(module_factory) {
    
    if (!setup(spec)) {
        throw std::runtime_error("Failed to setup modulation pipeline");
    }
}

bool ModulationPipeline::setup(const pipeline::PipelineSpec& spec) {
    try {
        // Setup memory pool for large tensor allocations
        setup_memory_pool(spec);
        
        // Create QAM modulator module via factory
        nlohmann::json module_config;
        module_config["modulation_order"] = "QAM_16";
        module_config["max_batch_size"] = 1024;
        module_config["max_symbols_per_batch"] = 10000;
        module_config["scaling_factor"] = 1.0f;
        module_config["normalize_power"] = true;
        
        auto module = module_factory_->create_module(
            pipeline_id_ + "_qam_modulator",
            "qam_modulator", 
            module_config
        );
        qam_modulator_ = std::unique_ptr<QAMModulator>(
            static_cast<QAMModulator*>(module.release())
        );
        
        // Setup CUDA graph if supported
        setup_cuda_graph();
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void ModulationPipeline::setup_memory_pool(const pipeline::PipelineSpec& spec) {
    // Calculate memory requirements for modulation
    size_t max_bits_per_batch = 1024 * 8;        // Max bits to modulate
    size_t max_symbols_per_batch = 10000;        // Max output symbols
    size_t complex_size = sizeof(cuComplex);
    
    // Memory for input bits and output symbols
    size_t input_memory = max_bits_per_batch * sizeof(uint32_t);
    size_t output_memory = max_symbols_per_batch * complex_size;
    
    // Total memory with overhead
    size_t total_memory = (input_memory + output_memory) * 2; // Double buffer
    
    // Create framework memory pool
    memory::MemoryPoolConfig pool_config{};
    pool_config.device_id = 0;
    pool_config.initial_size = total_memory;
    pool_config.max_size = total_memory * 2;
    
    memory_pool_ = std::make_unique<memory::MemoryPool>(pool_config);
}

void ModulationPipeline::setup_cuda_graph() {
    // Create CUDA graph for optimized execution
    cudaError_t err = cudaGraphCreate(&cuda_graph_, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA graph");
    }
    graph_instantiated_ = false;
}

task::TaskResult ModulationPipeline::execute_pipeline(
    std::span<const tensor::TensorInfo> external_inputs,
    std::span<tensor::TensorInfo> external_outputs,
    const task::CancellationToken& token
) {
    return execute_internal(external_inputs, external_outputs, false);
}

task::TaskResult ModulationPipeline::execute_pipeline_graph(
    std::span<const tensor::TensorInfo> external_inputs,
    std::span<tensor::TensorInfo> external_outputs,
    const task::CancellationToken& token
) {
    return execute_internal(external_inputs, external_outputs, true);
}

task::TaskResult ModulationPipeline::execute_internal(
    std::span<const tensor::TensorInfo> inputs,
    std::span<tensor::TensorInfo> outputs,
    bool use_graph
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Validate inputs
        if (inputs.size() < 1) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                "Insufficient inputs: need input_bits"
            );
        }
        
        if (outputs.size() < 1) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                "Insufficient outputs: need output_symbols"
            );
        }
        
        // Validate tensor dimensions
        const auto& input_tensor = inputs[0];
        auto& output_tensor = outputs[0];
        
        // Convert to vector format expected by module
        std::vector<tensor::TensorInfo> input_vec{inputs.begin(), inputs.end()};
        std::vector<tensor::TensorInfo> output_vec{outputs.begin(), outputs.end()};
        
        // Execute QAM modulation module
        task::CancellationToken dummy_token; // In real implementation, pass actual token
        auto result = qam_modulator_->execute(input_vec, output_vec, dummy_token);
        
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

void ModulationPipeline::teardown() {
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
    qam_modulator_.reset();
    
    // Cleanup memory pool
    memory_pool_.reset();
}

bool ModulationPipeline::is_ready() const {
    return qam_modulator_ && memory_pool_;
}

pipeline::PipelineStats ModulationPipeline::get_stats() const {
    return stats_;
}

// Factory implementations
std::unique_ptr<pipeline::IPipeline> ModulationPipelineFactory::create_pipeline(
    const std::string& pipeline_id,
    gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
    const pipeline::PipelineSpec& spec
) {
    return std::make_unique<ModulationPipeline>(
        pipeline_id, module_factory, spec
    );
}

QAMModulatorModuleFactory::QAMModulatorModuleFactory(
    const ModulationPipelineConfig& default_config
) : default_config_(default_config) {}

std::unique_ptr<pipeline::IModule> QAMModulatorModuleFactory::create_module(
    const std::string& module_id,
    const std::string& module_type,
    const nlohmann::json& config
) {
    if (module_type != "qam_modulator") {
        return nullptr;
    }
    
    ModulationParams params;
    
    // Parse configuration
    if (config.contains("modulation_order")) {
        std::string order = config["modulation_order"];
        if (order == "QPSK") {
            params.scheme = ModulationScheme::QPSK;
        } else if (order == "QAM_16") {
            params.scheme = ModulationScheme::QAM_16;
        } else if (order == "QAM_64") {
            params.scheme = ModulationScheme::QAM_64;
        } else if (order == "QAM_256") {
            params.scheme = ModulationScheme::QAM_256;
        }
    }
    
    if (config.contains("scaling_factor")) {
        params.scaling_factor = config["scaling_factor"];
    }
    if (config.contains("normalize_power")) {
        params.normalize_power = config["normalize_power"];
    }
    if (config.contains("max_symbols_per_batch")) {
        params.num_symbols = config["max_symbols_per_batch"];
    }
    
    return std::make_unique<QAMModulator>(module_id, params);
}

} // namespace framework::examples