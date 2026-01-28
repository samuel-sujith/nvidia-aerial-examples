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
#include "ml_channel_estimator_tensorrt.hpp"

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
    
    [[nodiscard]] std::size_t get_num_external_inputs() const override {
        return 2; // rx_pilots and tx_pilots
    }
    
    [[nodiscard]] std::size_t get_num_external_outputs() const override {
        return 1; // channel_estimates
    }

    void setup() override;
    
    void warmup(cudaStream_t stream) override;
    
    void configure_io(
        const framework::pipeline::DynamicParams& params,
        std::span<const framework::pipeline::PortInfo> external_inputs,
        std::span<framework::pipeline::PortInfo> external_outputs,
        cudaStream_t stream
    ) override;
    
    void execute_stream(cudaStream_t stream) override;
    void execute_graph(cudaStream_t stream) override;

private:
    std::string pipeline_id_;
    ChannelEstParams channel_params_;
    std::unique_ptr<IChannelEstimator> channel_estimator_;
    std::vector<framework::tensor::TensorInfo> internal_tensors_;
    void* device_memory_{nullptr};
    std::size_t memory_size_{0};
    framework::pipeline::ModuleMemorySlice module_slice_{};
    void allocate_pipeline_memory();
    void setup_tensor_connections();
};

} // namespace channel_estimation

#endif // CHANNEL_ESTIMATION_PIPELINE_HPP