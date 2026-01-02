#pragma once

#include "fft_module.hpp"
#include <framework/pipeline/pipeline_spec.hpp>
#include <framework/pipeline/pipeline_factory.hpp>
#include <framework/task/task_result.hpp>
#include <cufft.h>
#include <vector>
#include <memory>
#include <string>

namespace fft_processing {

/// FFT operation type
enum class FFTType {
    Forward,    ///< Forward FFT (time to frequency domain)
    Inverse,    ///< Inverse FFT (frequency to time domain)
    Both        ///< Both forward and inverse operations
};

/// FFT precision mode
enum class FFTPrecision {
    Single,     ///< Single precision (float/cufftComplex)
    Double      ///< Double precision (double/cufftDoubleComplex)
};

/// Configuration for FFT pipeline
struct FFTPipelineConfig {
    std::vector<size_t> fft_sizes{1024};  ///< Supported FFT sizes
    size_t max_batch_size{64};             ///< Maximum batch size
    FFTType operation_type{FFTType::Both}; ///< Forward, inverse, or both
    FFTPrecision precision{FFTPrecision::Single};
    bool enable_cuda_graphs{true};
    bool enable_profiling{false};
    int gpu_device_id{0};
    
    // Memory management
    size_t memory_pool_size{512 * 1024 * 1024}; // 512MB
    size_t memory_alignment{256};
    
    // Performance tuning
    int num_cuda_streams{2};
    bool use_multiple_gpus{false};
    std::vector<int> gpu_device_ids{0};
    
    // cuFFT specific options
    bool enable_autotuning{true};
    cufftXtWorkAreaPolicy workspace_policy{CUFFT_WORKSPACE_MINIMAL};
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
    
    // Operation-specific stats
    uint64_t forward_ffts{0};
    uint64_t inverse_ffts{0};
    uint64_t forward_time_us{0};
    uint64_t inverse_time_us{0};
    
    double average_throughput_msamples_per_sec() const {
        if (total_execution_time_us == 0) return 0.0;
        return (static_cast<double>(total_samples_processed) / total_execution_time_us);
    }
    
    double average_latency_us() const {
        if (total_batches_processed == 0) return 0.0;
        return static_cast<double>(total_execution_time_us) / total_batches_processed;
    }
    
    double average_forward_time_us() const {
        if (forward_ffts == 0) return 0.0;
        return static_cast<double>(forward_time_us) / forward_ffts;
    }
    
    double average_inverse_time_us() const {
        if (inverse_ffts == 0) return 0.0;
        return static_cast<double>(inverse_time_us) / inverse_ffts;
    }
};

/// High-performance FFT pipeline with GPU optimization
class FFTPipeline final : public aerial::pipeline::IPipeline {
private:
    FFTPipelineConfig config_;
    std::unique_ptr<FFTProcessor> fft_processor_;
    std::unique_ptr<aerial::memory::MemoryPool> memory_pool_;
    
    // cuFFT resources
    std::map<size_t, cufftHandle> fft_plans_;
    std::map<size_t, cufftHandle> ifft_plans_;
    cudaStream_t streams_[4];
    cudaEvent_t events_[8];
    
    // CUDA graph support
    std::map<std::pair<size_t, size_t>, cudaGraph_t> cuda_graphs_;
    std::map<std::pair<size_t, size_t>, cudaGraphExec_t> graph_execs_;
    
    // Buffer management
    void* d_input_data_{nullptr};
    void* d_output_data_{nullptr};
    void* d_temp_buffer_{nullptr};
    size_t current_buffer_size_{0};
    
    // Performance tracking
    FFTPipelineStats stats_;
    std::chrono::high_resolution_clock::time_point last_start_time_;
    
    // State management
    bool is_initialized_{false};
    std::string pipeline_id_;
    
public:
    explicit FFTPipeline(const FFTPipelineConfig& config);
    ~FFTPipeline();
    
    // IPipeline interface
    std::string_view get_pipeline_id() const override;
    std::size_t get_num_external_inputs() const override { return 1; }
    std::size_t get_num_external_outputs() const override { return 1; }
    
    aerial::task::TaskResult execute_pipeline(
        std::span<const aerial::tensor::TensorInfo> inputs,
        std::span<aerial::tensor::TensorInfo> outputs,
        const aerial::task::CancellationToken& token) override;
    
    aerial::task::TaskResult execute_pipeline_graph(
        std::span<const aerial::tensor::TensorInfo> inputs,
        std::span<aerial::tensor::TensorInfo> outputs,
        const aerial::task::CancellationToken& token) override;
    
    bool setup(const aerial::pipeline::PipelineSpec& spec) override;
    void teardown() override;
    bool is_ready() const override { return is_initialized_; }
    
    aerial::pipeline::PipelineStats get_stats() const override;
    
    // FFT-specific interface
    FFTPipelineStats get_fft_stats() const { return stats_; }
    
    /// Execute forward FFT
    aerial::task::TaskResult execute_forward_fft(
        const std::vector<std::complex<float>>& input_data,
        std::vector<std::complex<float>>& output_data,
        size_t fft_size,
        size_t batch_size = 1,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute inverse FFT
    aerial::task::TaskResult execute_inverse_fft(
        const std::vector<std::complex<float>>& input_data,
        std::vector<std::complex<float>>& output_data,
        size_t fft_size,
        size_t batch_size = 1,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute batch of FFTs with different sizes
    aerial::task::TaskResult execute_mixed_batch(
        const std::vector<std::vector<std::complex<float>>>& input_batches,
        std::vector<std::vector<std::complex<float>>>& output_batches,
        const std::vector<size_t>& fft_sizes,
        FFTType operation_type,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute OFDM-specific operations (FFT + windowing + overlap-add)
    aerial::task::TaskResult execute_ofdm_processing(
        const std::vector<std::complex<float>>& input_symbols,
        std::vector<std::complex<float>>& output_samples,
        size_t fft_size,
        size_t cp_length,
        const std::vector<float>& window_function = {},
        const aerial::task::CancellationToken& token = {});
    
    /// Update configuration dynamically
    bool update_config(const FFTPipelineConfig& new_config);
    
    /// Get current configuration
    const FFTPipelineConfig& get_config() const { return config_; }
    
    /// Reset performance statistics
    void reset_stats();
    
    /// Get supported FFT sizes
    const std::vector<size_t>& get_supported_sizes() const { return config_.fft_sizes; }
    
    /// Check if specific FFT size is supported
    bool is_fft_size_supported(size_t fft_size) const;

private:
    bool initialize_cuda_resources();
    void cleanup_cuda_resources();
    bool create_fft_plans();
    void destroy_fft_plans();
    bool ensure_buffer_capacity(size_t required_samples);
    bool create_cuda_graph(size_t fft_size, size_t batch_size);
    void update_performance_stats(uint64_t execution_time_us, size_t samples_processed, FFTType operation);
    aerial::task::TaskResult validate_inputs(std::span<const aerial::tensor::TensorInfo> inputs) const;
    aerial::task::TaskResult validate_outputs(std::span<aerial::tensor::TensorInfo> outputs) const;
    
    // cuFFT helpers
    cufftResult execute_cufft_plan(cufftHandle plan, void* input, void* output, int direction, cudaStream_t stream);
    size_t calculate_workspace_size(size_t fft_size, size_t batch_size);
};

/// Factory for creating FFT pipelines
class FFTPipelineFactory {
public:
    static std::unique_ptr<FFTPipeline> create_pipeline(
        const FFTPipelineConfig& config = {});
    
    static std::unique_ptr<FFTPipeline> create_from_spec(
        const aerial::pipeline::PipelineSpec& spec);
    
    static FFTPipelineConfig get_default_config(const std::vector<size_t>& fft_sizes = {1024});
    static FFTPipelineConfig get_high_performance_config(const std::vector<size_t>& fft_sizes = {1024, 2048, 4096});
    static FFTPipelineConfig get_low_latency_config(const std::vector<size_t>& fft_sizes = {256, 512});
    static FFTPipelineConfig get_ofdm_config(size_t subcarrier_count = 1024);
    
    /// Get optimal FFT configuration for specific use case
    static FFTPipelineConfig get_optimized_config(
        const std::vector<size_t>& target_sizes,
        size_t expected_batch_size,
        FFTPrecision precision = FFTPrecision::Single,
        bool prioritize_latency = false);
};

} // namespace fft_processing