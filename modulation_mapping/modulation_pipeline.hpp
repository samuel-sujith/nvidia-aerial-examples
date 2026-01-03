/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "modulator.hpp"
#include "task/task.hpp"
#include <vector>
#include <memory>
#include <string>

namespace framework {
namespace examples {

/// Configuration for modulation pipeline
struct ModulationPipelineConfig {
    ModulationScheme modulation_order{ModulationScheme::QAM_16};
    size_t max_batch_size{1024};
    size_t max_symbols_per_batch{10000};
    int gpu_device_id{0};
};

/// Simple QAM modulation pipeline (follows channel_estimation pattern)
class ModulationPipeline {
public:
    explicit ModulationPipeline(const ModulationPipelineConfig& config = {});
    ~ModulationPipeline();

    // Simple task-based interface like channel_estimation
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    );

    /// Get pipeline identifier
    std::string_view get_pipeline_id() const { return pipeline_id_; }
    
    /// Get current configuration  
    const ModulationPipelineConfig& get_config() const { return config_; }

private:
    ModulationPipelineConfig config_;
    std::string pipeline_id_;
    std::unique_ptr<QAMModulator> modulator_;
    
    // State tracking
    std::atomic<bool> is_initialized_{false};
    
    // Helper methods
    bool initialize();
    void cleanup();
};

/// Factory for creating modulation pipelines
class ModulationPipelineFactory {
public:
    static std::unique_ptr<ModulationPipeline> create(
        const ModulationPipelineConfig& config = {});
        
    static ModulationPipelineConfig get_default_config(ModulationScheme order);
};

} // namespace examples
} // namespace framework