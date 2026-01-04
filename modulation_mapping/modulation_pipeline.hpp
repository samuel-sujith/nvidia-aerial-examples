/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <span>

#include "pipeline/ipipeline.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/imodule_factory.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "task/task.hpp"
#include "gsl-lite/gsl-lite.hpp"
#include "memory/memory_pool.hpp"
#include <nlohmann/json.hpp>

#include "modulator.hpp"

namespace framework::examples {

/// Configuration for modulation pipeline
struct ModulationPipelineConfig {
    ModulationScheme modulation_order{ModulationScheme::QAM_16};
    size_t max_batch_size{1024};
    size_t max_symbols_per_batch{10000};
    int gpu_device_id{0};
};

/**
 * QAM Modulation Pipeline
 * 
 * Pipeline Flow:
 * External Input 0 (Input Bits) ──→ QAMModulator ──→ External Output (Complex Symbols)
 * 
 * This pipeline demonstrates:
 * - High-performance GPU-based QAM modulation
 * - Batched symbol processing
 * - Multiple modulation schemes (QPSK, 16-QAM, 64-QAM, 256-QAM)
 * - Integration with Aerial framework patterns
 */
class ModulationPipeline final : public pipeline::IPipeline {
public:
    /**
     * Constructor
     * @param pipeline_id Unique identifier for this pipeline
     * @param module_factory Factory for creating modules
     * @param spec Pipeline specification
     */
    ModulationPipeline(
        std::string pipeline_id,
        gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
        const pipeline::PipelineSpec& spec
    );
    
    ~ModulationPipeline() override = default;

    // Non-copyable, non-movable
    ModulationPipeline(const ModulationPipeline&) = delete;
    ModulationPipeline& operator=(const ModulationPipeline&) = delete;
    ModulationPipeline(ModulationPipeline&&) = delete;
    ModulationPipeline& operator=(ModulationPipeline&&) = delete;

    // IPipeline interface
    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }
    [[nodiscard]] std::size_t get_num_external_inputs() const override { return 1; }
    [[nodiscard]] std::size_t get_num_external_outputs() const override { return 1; }

    /// Execute pipeline with external inputs
    task::TaskResult execute_pipeline(
        std::span<const tensor::TensorInfo> external_inputs,
        std::span<tensor::TensorInfo> external_outputs,
        const task::CancellationToken& token
    ) override;

    /// Execute using CUDA graphs for better performance
    task::TaskResult execute_pipeline_graph(
        std::span<const tensor::TensorInfo> external_inputs,
        std::span<tensor::TensorInfo> external_outputs,
        const task::CancellationToken& token
    ) override;

    /// Setup pipeline resources
    bool setup(const pipeline::PipelineSpec& spec) override;
    
    /// Cleanup pipeline resources
    void teardown() override;

    /// Check if pipeline is ready for execution
    [[nodiscard]] bool is_ready() const override;

    /// Get performance statistics
    [[nodiscard]] pipeline::PipelineStats get_stats() const override;

private:
    std::string pipeline_id_;
    gsl_lite::not_null<pipeline::IModuleFactory*> module_factory_;
    std::unique_ptr<QAMModulator> qam_modulator_;
    
    // Memory management
    std::unique_ptr<memory::MemoryPool> memory_pool_;
    
    // CUDA graph support
    cudaGraph_t cuda_graph_{};
    cudaGraphExec_t graph_exec_{};
    bool graph_instantiated_{false};
    
    // Performance tracking
    mutable pipeline::PipelineStats stats_;
    
    // Internal methods
    void setup_memory_pool(const pipeline::PipelineSpec& spec);
    void setup_cuda_graph();
    task::TaskResult execute_internal(
        std::span<const tensor::TensorInfo> inputs,
        std::span<tensor::TensorInfo> outputs,
        bool use_graph
    );
};

/// Factory for creating modulation pipeline
class ModulationPipelineFactory final : public pipeline::IPipelineFactory {
public:
    ModulationPipelineFactory() = default;
    ~ModulationPipelineFactory() override = default;

    /// Create pipeline instance
    std::unique_ptr<pipeline::IPipeline> create_pipeline(
        const std::string& pipeline_id,
        gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
        const pipeline::PipelineSpec& spec
    ) override;

    /// Get supported pipeline types
    [[nodiscard]] std::vector<std::string> get_supported_types() const override {
        return {"modulation_pipeline"};
    }
};

/// Module factory for QAM modulator
class QAMModulatorModuleFactory final : public pipeline::IModuleFactory {
public:
    explicit QAMModulatorModuleFactory(const ModulationPipelineConfig& default_config);
    ~QAMModulatorModuleFactory() override = default;

    /// Create module instance
    std::unique_ptr<pipeline::IModule> create_module(
        const std::string& module_id,
        const std::string& module_type,
        const nlohmann::json& config
    ) override;

    /// Get supported module types
    [[nodiscard]] std::vector<std::string> get_supported_types() const override {
        return {"qam_modulator"};
    }

private:
    ModulationPipelineConfig default_config_;
};

} // namespace framework::examples