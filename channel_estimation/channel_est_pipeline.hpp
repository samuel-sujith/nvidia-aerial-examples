/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FRAMEWORK_EXAMPLES_CHANNEL_EST_PIPELINE_HPP
#define FRAMEWORK_EXAMPLES_CHANNEL_EST_PIPELINE_HPP

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
#include "memory/memory_pool.hpp"

#include "channel_estimator.hpp"

namespace framework::examples {

/**
 * Channel Estimation Pipeline
 * 
 * Pipeline Flow:
 * External Input 0 (Received Signal) ──┐
 *                                      ├─→ ChannelEstimator ─→ External Output (Channel Estimates)
 * External Input 1 (Pilot Symbols) ───┘
 * 
 * This pipeline demonstrates:
 * - GPU-based signal processing
 * - Memory management for large tensors
 * - Stream-based execution
 * - Integration with Aerial framework patterns
 */
class ChannelEstimationPipeline final : public pipeline::IPipeline {
public:
    /**
     * Constructor
     * @param pipeline_id Unique identifier for this pipeline
     * @param module_factory Factory for creating modules
     * @param spec Pipeline specification
     */
    ChannelEstimationPipeline(
        std::string pipeline_id,
        gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
        const pipeline::PipelineSpec& spec
    );
    
    ~ChannelEstimationPipeline() override = default;

    // Non-copyable, non-movable
    ChannelEstimationPipeline(const ChannelEstimationPipeline&) = delete;
    ChannelEstimationPipeline& operator=(const ChannelEstimationPipeline&) = delete;
    ChannelEstimationPipeline(ChannelEstimationPipeline&&) = delete;
    ChannelEstimationPipeline& operator=(ChannelEstimationPipeline&&) = delete;

    // IPipeline interface
    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }
    [[nodiscard]] std::size_t get_num_external_inputs() const override { return 2; }
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
    std::unique_ptr<ChannelEstimator> channel_estimator_;
    
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

/// Factory for creating channel estimation pipeline
class ChannelEstimationPipelineFactory final : public pipeline::IPipelineFactory {
public:
    ChannelEstimationPipelineFactory() = default;
    ~ChannelEstimationPipelineFactory() override = default;

    /// Create pipeline instance
    std::unique_ptr<pipeline::IPipeline> create_pipeline(
        const std::string& pipeline_id,
        gsl_lite::not_null<pipeline::IModuleFactory*> module_factory,
        const pipeline::PipelineSpec& spec
    ) override;

    /// Get supported pipeline types
    [[nodiscard]] std::vector<std::string> get_supported_types() const override {
        return {"channel_estimation_pipeline"};
    }
};

/// Module factory for channel estimator
class ChannelEstimatorModuleFactory final : public pipeline::IModuleFactory {
public:
    explicit ChannelEstimatorModuleFactory(const ChannelEstParams& default_params);
    ~ChannelEstimatorModuleFactory() override = default;

    /// Create module instance
    std::unique_ptr<pipeline::IModule> create_module(
        const std::string& module_id,
        const pipeline::ModuleSpec& spec
    ) override;

    /// Get supported module types
    [[nodiscard]] std::vector<std::string> get_supported_types() const override {
        return {"channel_estimator"};
    }

private:
    ChannelEstParams default_params_;
    
    /// Parse module-specific parameters from spec
    ChannelEstParams parse_params(const pipeline::ModuleSpec& spec) const;
};

/// Utility functions for tensor operations
namespace tensor_utils {

/// Allocate GPU tensor for complex data
tensor::TensorInfo allocate_complex_tensor(
    const std::vector<std::size_t>& dimensions,
    memory::MemoryPool& pool
);

/// Copy tensor data asynchronously
cudaError_t copy_tensor_async(
    const tensor::TensorInfo& src,
    tensor::TensorInfo& dst,
    cudaStream_t stream
);

/// Validate tensor dimensions for channel estimation
bool validate_channel_est_tensors(
    const tensor::TensorInfo& rx_tensor,
    const tensor::TensorInfo& tx_tensor,
    const tensor::TensorInfo& output_tensor,
    const ChannelEstParams& params
);

} // namespace tensor_utils

} // namespace framework::examples

#endif // FRAMEWORK_EXAMPLES_CHANNEL_EST_PIPELINE_HPP