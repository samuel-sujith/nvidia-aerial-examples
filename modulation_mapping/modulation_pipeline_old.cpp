/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "modulation_pipeline.hpp"
#include <sstream>

namespace framework {
namespace examples {

ModulationPipeline::ModulationPipeline(const ModulationPipelineConfig& config)
    : config_(config), pipeline_id_("modulation_pipeline_v1") {
    
    // Generate unique pipeline ID
    std::stringstream ss;
    ss << "modulation_pipeline_" << static_cast<int>(config_.modulation_order) 
       << "_" << config_.max_batch_size;
    pipeline_id_ = ss.str();
    
    initialize();
}

ModulationPipeline::~ModulationPipeline() {
    cleanup();
}

bool ModulationPipeline::initialize() {
    try {
        // Create modulator with appropriate parameters
        ModulationParams params;
        params.scheme = config_.modulation_order;
        params.num_symbols = config_.max_symbols_per_batch;
        params.scaling_factor = 1.0f;
        params.normalize_power = true;
        
        modulator_ = std::make_unique<QAMModulator>("pipeline_modulator", params);
        is_initialized_ = true;
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void ModulationPipeline::cleanup() {
    modulator_.reset();
    is_initialized_ = false;
}

task::TaskResult ModulationPipeline::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token
) {
    // Note: Simplified cancellation check - actual interface may differ
    // if (token.is_cancelled()) {
    //     return task::TaskResult(task::TaskStatus::Cancelled, "Task cancelled");
    // }
    
    if (!is_initialized_) {
        return task::TaskResult(task::TaskStatus::Failed, "Pipeline not initialized");
    }
    
    if (!modulator_) {
        return task::TaskResult(task::TaskStatus::Failed, "Modulator not available");
    }
    
    // Delegate to the modulator
    return modulator_->execute(inputs, outputs, token);
}

// Factory methods
std::unique_ptr<ModulationPipeline> ModulationPipelineFactory::create(
    const ModulationPipelineConfig& config) {
    return std::make_unique<ModulationPipeline>(config);
}

ModulationPipelineConfig ModulationPipelineFactory::get_default_config(ModulationScheme order) {
    ModulationPipelineConfig config;
    config.modulation_order = order;
    config.max_batch_size = 1024;
    config.max_symbols_per_batch = 10000;
    config.gpu_device_id = 0;
    return config;
}

} // namespace examples
} // namespace framework