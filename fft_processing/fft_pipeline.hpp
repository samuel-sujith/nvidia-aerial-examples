#pragma once

#include "fft_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "task/task.hpp"
#include <cufft.h>
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace fft_processing {

/// FFT operation type
enum class FFTType {
    Forward,    ///< Forward FFT (time to frequency domain)
    Inverse,    ///< Inverse FFT (frequency to time domain)
    Both        ///< Both forward and inverse operations
};

/// FFT precision type
enum class FFTPrecision {
    Float32,    ///< Single precision (32-bit)
    Float64     ///< Double precision (64-bit)
};

/// Configuration for FFT pipeline
struct FFTPipelineConfig {
    std::vector<size_t> fft_sizes{1024, 2048, 4096};
    size_t max_batch_size{1024};
    bool enable_cuda_graphs{true};
    bool enable_profiling{false};
    int gpu_device_id{0};
    FFTPrecision precision{FFTPrecision::Float32};
    
    // Memory management
    size_t memory_pool_size{256 * 1024 * 1024}; // 256MB
    size_t memory_alignment{256};
    
    // Performance tuning
    int num_cuda_streams{2};
    size_t shared_memory_size{48 * 1024}; // 48KB
    
    // GPU configuration
    std::vector<int> gpu_device_ids{0};
    
    // cuFFT specific options  
    bool enable_autotuning{true};
};

/// Performance statistics for FFT pipeline
struct FFTPipelineStats {
    uint64_t total_ffts_processed{0};
    uint64_t total_samples_processed{0};
    uint64_t total_batches_processed{0};
    uint64_t total_execution_time_us{0};
    uint64_t last_execution_time_us{0};
    uint64_t min_execution_time_us{UINT64_MAX};
    uint64_t max_execution_time_us{0};
    
    double average_throughput_msps() const {
        if (total_execution_time_us == 0) return 0.0;
        return (static_cast<double>(total_samples_processed) / total_execution_time_us);
    }
    
    double average_latency_us() const {
        if (total_batches_processed == 0) return 0.0;
        return static_cast<double>(total_execution_time_us) / total_batches_processed;
    }
};

/// High-performance FFT pipeline with GPU optimization
class FFTPipeline {
private:
    FFTPipelineConfig config_;
    FFTPipelineStats stats_;
    bool is_initialized_{false};
    std::map<size_t, cufftHandle> fft_plans_;
    std::map<size_t, cufftHandle> ifft_plans_;

public:
    explicit FFTPipeline(const FFTPipelineConfig& config);
    ~FFTPipeline();

    // Core pipeline interface (simplified, no override keywords)
    std::size_t get_num_external_inputs() const { return 1; }
    std::size_t get_num_external_outputs() const { return 1; }
    
    ::framework::task::TaskResult execute_pipeline(
        std::span<const ::framework::tensor::TensorInfo> inputs,
        std::span<::framework::tensor::TensorInfo> outputs,
        const ::framework::task::CancellationToken& token = {});
    
    bool setup(const ::framework::pipeline::PipelineSpec& spec);
    void teardown();
    bool is_ready() const { return is_initialized_; }
    
    // FFT-specific interface
    FFTPipelineStats get_fft_stats() const { return stats_; }
    
    // Configuration access
    const FFTPipelineConfig& get_config() const { return config_; }
    
    // FFT size support
    const std::vector<size_t>& get_supported_sizes() const { return config_.fft_sizes; }
    bool is_fft_size_supported(size_t fft_size) const;

private:
    // Simple initialization
    bool initialize_cuda_resources();
    void cleanup_cuda_resources();
};

/// Factory for creating FFT pipelines
class FFTPipelineFactory {
public:
    static std::unique_ptr<FFTPipeline> create_pipeline(
        const FFTPipelineConfig& config = {});
        
    static FFTPipelineConfig get_default_config(const std::vector<size_t>& fft_sizes = {1024});
};

} // namespace fft_processing