#pragma once

#include "mimo_detector.hpp"
#include "pipeline/ipipeline.hpp"
#include "task/task.hpp"
#include <cublas_v2.h>
#include <vector>
#include <memory>
#include <map>
#include <string>

namespace mimo_detection {

/// MIMO detection algorithms
enum class MIMOAlgorithm {
    ZF,              ///< Zero Forcing
    ZeroForcing = ZF, ///< Alias for Zero Forcing
    MMSE,            ///< Minimum Mean Square Error  
    ML,              ///< Maximum Likelihood
    MaximumLikelihood = ML, ///< Alias for Maximum Likelihood
    VBlast           ///< Vertical-Bell Labs Layered Space-Time
};

/// Configuration for MIMO pipeline
struct MIMOPipelineConfig {
    size_t max_tx_antennas{8};      
    size_t max_rx_antennas{8};      
    size_t max_symbols_per_batch{1000}; 
    size_t max_batch_size{64};      
    MIMOAlgorithm detection_algorithm{MIMOAlgorithm::MMSE};
    int modulation_order{16}; // Simple int instead of complex enum
    bool enable_cuda_graphs{true};
    bool enable_profiling{false};
    int gpu_device_id{0};
    
    size_t memory_pool_size{1024 * 1024 * 1024}; // 1GB
    size_t memory_alignment{256};
    int num_cuda_streams{2};
};

/// Pipeline statistics
struct MIMOPipelineStats {
    uint64_t total_symbols_processed{0};
    uint64_t total_symbols_detected{0}; // Alias for compatibility
    uint64_t total_batches_processed{0};
    uint64_t total_execution_time_us{0};
    uint64_t last_execution_time_us{0};
    uint64_t min_execution_time_us{UINT64_MAX};
    uint64_t max_execution_time_us{0};
    
    double average_throughput_msps() const {
        if (total_execution_time_us == 0) return 0.0;
        return (static_cast<double>(total_symbols_processed) / total_execution_time_us);
    }
    
    double average_throughput_msymbols_per_sec() const {
        return average_throughput_msps(); // Alias for compatibility
    }
    
    double average_latency_us() const {
        if (total_batches_processed == 0) return 0.0;
        return static_cast<double>(total_execution_time_us) / total_batches_processed;
    }
    
    double average_detection_time_us() const {
        return average_latency_us(); // Alias for compatibility
    }
};

/// Simplified MIMO detection pipeline
class MIMOPipeline {
private:
    MIMOPipelineConfig config_;
    std::string pipeline_id_;
    MIMOPipelineStats stats_;
    bool is_initialized_{false};
    
    // Simplified CUDA resources
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    cublasHandle_t cublas_handle_;

public:
    explicit MIMOPipeline(const MIMOPipelineConfig& config);
    ~MIMOPipeline();
    
    // Basic interface
    std::string_view get_pipeline_id() const;
    std::size_t get_num_external_inputs() const { return 2; }
    std::size_t get_num_external_outputs() const { return 1; }
    
    ::framework::task::TaskResult execute_pipeline(
        std::span<const ::framework::tensor::TensorInfo> inputs,
        std::span<::framework::tensor::TensorInfo> outputs,
        const ::framework::task::CancellationToken& token = {});
    
    bool setup(const ::framework::pipeline::PipelineSpec& spec);
    void teardown();
    bool is_ready() const { return is_initialized_; }
    
    MIMOPipelineStats get_mimo_stats() const { return stats_; }
    const MIMOPipelineConfig& get_config() const { return config_; }
    
    // MIMO-specific detection methods
    ::framework::task::TaskResult detect_symbols(
        const std::vector<std::complex<float>>& rx_signals,
        const std::vector<std::complex<float>>& channel,
        std::vector<std::complex<float>>& detected_symbols,
        MIMOAlgorithm algorithm = MIMOAlgorithm::MMSE,
        const ::framework::task::CancellationToken& token = {});
    
    ::framework::task::TaskResult detect_batch_symbols(
        const std::vector<std::vector<std::complex<float>>>& rx_batches,
        const std::vector<std::vector<std::complex<float>>>& channel_batches,
        std::vector<std::vector<std::complex<float>>>& detected_batches,
        MIMOAlgorithm algorithm = MIMOAlgorithm::MMSE,
        const ::framework::task::CancellationToken& token = {});
};

/// Factory for creating MIMO pipelines
class MIMOPipelineFactory {
public:
    static std::unique_ptr<MIMOPipeline> create_pipeline(
        const MIMOPipelineConfig& config = {});
    
    static MIMOPipelineConfig get_default_config(size_t num_tx = 4, size_t num_rx = 4);
    static MIMOPipelineConfig get_high_performance_config(size_t num_tx = 8, size_t num_rx = 8);
    static MIMOPipelineConfig get_low_latency_config(size_t num_tx = 2, size_t num_rx = 2);
};

} // namespace mimo_detection