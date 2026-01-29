/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "path_loss_prediction_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace path_loss_prediction {

class PathLossPredictionPipeline {
public:
    struct PipelineConfig {
        PathLossModelParams model_params;
        std::string module_id;
        bool enable_profiling = false;

        PipelineConfig() { module_id = "path_loss_prediction_pipeline"; }
    };

    struct PerformanceMetrics {
        size_t total_processed_batches = 0;
        double avg_processing_time_ms = 0.0;
        double peak_processing_time_ms = 0.0;
        double throughput_samples_per_ms = 0.0;
    };

    explicit PathLossPredictionPipeline(const PipelineConfig& config);
    ~PathLossPredictionPipeline();

    bool initialize();
    bool predict(
        const std::vector<float>& features,
        std::vector<float>& path_loss_db,
        cudaStream_t stream = 0);

    PerformanceMetrics get_performance_metrics() const;

private:
    void allocate_buffers();
    void deallocate_buffers();
    void update_metrics(double processing_time_ms, size_t num_samples);

    PipelineConfig config_;
    std::shared_ptr<PathLossPredictionModule> module_;

    float* h_features_{nullptr};
    float* h_outputs_{nullptr};
    float* d_features_{nullptr};

    void* d_module_tensor_{nullptr};
    size_t module_tensor_bytes_{0};

    std::byte* static_desc_cpu_{nullptr};
    std::byte* static_desc_gpu_{nullptr};
    std::byte* dynamic_desc_cpu_{nullptr};
    std::byte* dynamic_desc_gpu_{nullptr};
    size_t static_desc_bytes_{0};
    size_t dynamic_desc_bytes_{0};

    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
};

} // namespace path_loss_prediction
