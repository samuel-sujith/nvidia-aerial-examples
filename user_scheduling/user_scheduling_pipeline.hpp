#pragma once

#include "user_scheduling_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace user_scheduling {

struct UeFeatures {
    int ue_id = 0;
    float sinr_db = 0.0f;
    float buffer_bytes = 0.0f;
    float avg_rate_mbps = 0.0f;
    float qos_priority = 0.0f;
    float harq_pending = 0.0f;
};

class UserSchedulingPipeline {
public:
    struct PipelineConfig {
        SchedulingParams scheduling_params;
        int num_scheduled = 4;
        std::string model_path;
        std::string module_id;
        bool enable_profiling = false;

        PipelineConfig() { module_id = "user_scheduling_pipeline"; }
    };

    struct PerformanceMetrics {
        size_t total_processed_frames = 0;
        double avg_processing_time_ms = 0.0;
        double peak_processing_time_ms = 0.0;
        double throughput_ues_per_ms = 0.0;
    };

    explicit UserSchedulingPipeline(const PipelineConfig& config);
    ~UserSchedulingPipeline();

    bool initialize();

    bool process_scheduling(
        const std::vector<UeFeatures>& ues,
        std::vector<int>& scheduled_ue_ids,
        std::vector<float>* scores_out = nullptr,
        cudaStream_t stream = 0
    );

    PerformanceMetrics get_performance_metrics() const;

private:
    PipelineConfig config_;
    std::shared_ptr<UserSchedulingModule> scheduling_module_;

    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;

    // Memory buffers
    float* h_features_ = nullptr;
    float* h_scores_ = nullptr;
    float* d_features_ = nullptr;
    void* d_module_tensor_ = nullptr;
    size_t module_tensor_bytes_ = 0;

    void allocate_buffers();
    void deallocate_buffers();
    void update_metrics(double processing_time_ms, size_t num_ues);

    bool load_model_file(
        const std::string& path,
        std::vector<float>& mean,
        std::vector<float>& std,
        std::vector<float>& weights,
        float& bias
    ) const;

    void load_default_model(
        std::vector<float>& mean,
        std::vector<float>& std,
        std::vector<float>& weights,
        float& bias
    ) const;
};

} // namespace user_scheduling
