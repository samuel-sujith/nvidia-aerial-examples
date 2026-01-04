/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHANNEL_ESTIMATION_PIPELINE_HPP
#define CHANNEL_ESTIMATION_PIPELINE_HPP

#include <memory>
#include <string>
#include <vector>
#include <span>

#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

#include "channel_estimation_module.hpp"

namespace channel_estimation {

/**
 * Channel Estimation Pipeline
 * 
 * Pipeline Flow:
 * External Input 0 (Received Pilots) ──┐
 *                                      ├─→ ChannelEstimator ─→ External Output (Channel Estimates)
 * External Input 1 (Reference Pilots) ─┘
 * 
 * This pipeline demonstrates:
 * - GPU-based channel estimation
 * - Least squares and interpolation algorithms
 * - Integration with Aerial framework patterns
 */
class ChannelEstimationPipeline final : public framework::pipeline::IPipeline {
public:
    /**
     * Constructor
     * @param pipeline_id Unique identifier for this pipeline
     * @param params Channel estimation parameters
     */
    explicit ChannelEstimationPipeline(
        std::string pipeline_id,
        const ChannelEstParams& params = {}
    );
    
    ~ChannelEstimationPipeline() override = default;

    // IPipeline interface
    [[nodiscard]] std::string_view get_pipeline_id() const override { return pipeline_id_; }

    void setup() override;
    
    void warmup(cudaStream_t stream) override;
    
    void configure_io(
        const framework::pipeline::DynamicParams& params,
        std::span<const framework::pipeline::PortInfo> external_inputs,
        std::span<framework::pipeline::PortInfo> external_outputs,
        cudaStream_t stream
    ) override;
    
    void execute(cudaStream_t stream) override;

private:
    std::string pipeline_id_;
    ChannelEstParams channel_params_;
    std::unique_ptr<ChannelEstimator> channel_estimator_;
    
    // Internal tensor connections
    std::vector<framework::tensor::TensorInfo> internal_tensors_;
    
    // Memory management
    void* device_memory_{nullptr};
    std::size_t memory_size_{0};
    
    void allocate_pipeline_memory();
    void setup_tensor_connections();
};

} // namespace channel_estimation

#endif // CHANNEL_ESTIMATION_PIPELINE_HPP