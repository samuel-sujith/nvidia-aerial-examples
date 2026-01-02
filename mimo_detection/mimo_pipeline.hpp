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

/// MIMO detection algorithm type
enum class MIMOAlgorithm {
    ZeroForcing,     ///< Zero-Forcing (ZF) detection
    MMSE,            ///< Minimum Mean Square Error detection
    MLApprox,        ///< Maximum Likelihood approximation
    SIC,             ///< Successive Interference Cancellation
    VBlast           ///< Vertical-Bell Labs Layered Space-Time
};

/// Configuration for MIMO pipeline
struct MIMOPipelineConfig {
    size_t max_tx_antennas{8};      ///< Maximum transmit antennas
    size_t max_rx_antennas{8};      ///< Maximum receive antennas
    size_t max_symbols_per_batch{1000}; ///< Max symbols per processing batch
    size_t max_batch_size{64};      ///< Maximum batch size
    MIMOAlgorithm detection_algorithm{MIMOAlgorithm::MMSE};
    int modulation_order{16}; // Simple int instead of complex enum
    bool enable_cuda_graphs{true};
    bool enable_profiling{false};
    int gpu_device_id{0};
    
    // Memory management
    size_t memory_pool_size{1024 * 1024 * 1024}; // 1GB
    size_t memory_alignment{256};
    
    // Performance tuning
    int num_cuda_streams{2};
    bool enable_parallel_detection{true};
    
    // Algorithm-specific parameters
    float noise_variance{0.1f};      ///< Noise variance for MMSE
    float regularization_factor{1e-6f}; ///< Matrix regularization
    int max_iterations{10};          ///< For iterative algorithms
};

/// Performance statistics for MIMO pipeline
struct MIMOPipelineStats {
    uint64_t total_symbols_processed{0};
    uint64_t total_vectors_processed{0};
    uint64_t total_batches_processed{0};
    uint64_t total_execution_time_us{0};
    uint64_t last_execution_time_us{0};
    uint64_t min_execution_time_us{UINT64_MAX};
    uint64_t max_execution_time_us{0};
    
    // Algorithm-specific stats
    uint64_t channel_inversions{0};
    uint64_t qr_decompositions{0};
    uint64_t matrix_multiplications{0};
    
    double average_throughput_msymbols_per_sec() const {
        if (total_execution_time_us == 0) return 0.0;
        return (static_cast<double>(total_symbols_processed) / total_execution_time_us);
    }
    
    double average_latency_us() const {
        if (total_batches_processed == 0) return 0.0;
        return static_cast<double>(total_execution_time_us) / total_batches_processed;
    }
    
    double average_detection_complexity() const {
        if (total_vectors_processed == 0) return 0.0;
        return static_cast<double>(matrix_multiplications) / total_vectors_processed;
    }
};

/// High-performance MIMO detection pipeline with GPU optimization
class MIMOPipeline {
private:
    MIMOPipelineConfig config_;
    std::unique_ptr<MIMODetector> mimo_detector_;
    std::unique_ptr<aerial::memory::MemoryPool> memory_pool_;
    
    // CUDA resources
    cublasHandle_t cublas_handle_;
    cusolverDnHandle_t cusolver_handle_;
    cudaStream_t streams_[4];
    cudaEvent_t events_[8];
    
    // CUDA graph support
    std::map<std::tuple<size_t, size_t, size_t>, cudaGraph_t> cuda_graphs_;
    std::map<std::tuple<size_t, size_t, size_t>, cudaGraphExec_t> graph_execs_;
    
    // Buffer management
    void* d_channel_matrix_{nullptr};    ///< Device channel matrix
    void* d_received_symbols_{nullptr};  ///< Device received symbols
    void* d_detected_symbols_{nullptr};  ///< Device detected symbols
    void* d_work_buffer_{nullptr};       ///< Device workspace buffer
    void* d_temp_matrices_{nullptr};     ///< Temporary matrix storage
    size_t current_buffer_size_{0};
    
    // Algorithm-specific buffers
    void* d_channel_inverse_{nullptr};   ///< For ZF/MMSE
    void* d_qr_matrices_{nullptr};       ///< For QR decomposition
    void* d_permutation_{nullptr};       ///< For ordering
    
    // Performance tracking
    MIMOPipelineStats stats_;
    std::chrono::high_resolution_clock::time_point last_start_time_;
    
    // State management
    bool is_initialized_{false};
    std::string pipeline_id_;
    
public:
    explicit MIMOPipeline(const MIMOPipelineConfig& config);
    ~MIMOPipeline();
    
    // IPipeline interface
    std::string_view get_pipeline_id() const override;
    std::size_t get_num_external_inputs() const override { return 2; } // Channel + received symbols
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
    
    // MIMO-specific interface
    MIMOPipelineStats get_mimo_stats() const { return stats_; }
    
    /// Detect transmitted symbols using specified algorithm
    aerial::task::TaskResult detect_symbols(
        const std::vector<std::vector<std::complex<float>>>& channel_matrix,
        const std::vector<std::complex<float>>& received_symbols,
        std::vector<std::complex<float>>& detected_symbols,
        size_t num_tx_antennas,
        size_t num_rx_antennas,
        const aerial::task::CancellationToken& token = {});
    
    /// Batch MIMO detection for multiple symbol vectors
    aerial::task::TaskResult detect_batch(
        const std::vector<std::vector<std::complex<float>>>& channel_matrix,
        const std::vector<std::vector<std::complex<float>>>& received_batches,
        std::vector<std::vector<std::complex<float>>>& detected_batches,
        size_t num_tx_antennas,
        size_t num_rx_antennas,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute Zero-Forcing detection
    aerial::task::TaskResult execute_zero_forcing(
        const std::vector<std::vector<std::complex<float>>>& H,
        const std::vector<std::complex<float>>& y,
        std::vector<std::complex<float>>& x_hat,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute MMSE detection
    aerial::task::TaskResult execute_mmse(
        const std::vector<std::vector<std::complex<float>>>& H,
        const std::vector<std::complex<float>>& y,
        std::vector<std::complex<float>>& x_hat,
        float noise_variance,
        const aerial::task::CancellationToken& token = {});
    
    /// Execute ML approximation detection
    aerial::task::TaskResult execute_ml_approximation(
        const std::vector<std::vector<std::complex<float>>>& H,
        const std::vector<std::complex<float>>& y,
        std::vector<std::complex<float>>& x_hat,
        const std::vector<std::complex<float>>& constellation,
        const aerial::task::CancellationToken& token = {});
    
    /// Update configuration dynamically
    bool update_config(const MIMOPipelineConfig& new_config);
    
    /// Get current configuration
    const MIMOPipelineConfig& get_config() const { return config_; }
    
    /// Reset performance statistics
    void reset_stats();
    
    /// Get supported antenna configurations
    std::vector<std::pair<size_t, size_t>> get_supported_antenna_configs() const;

private:
    bool initialize_cuda_resources();
    void cleanup_cuda_resources();
    bool ensure_buffer_capacity(size_t tx_antennas, size_t rx_antennas, size_t num_symbols);
    bool create_cuda_graph(size_t tx_antennas, size_t rx_antennas, size_t batch_size);
    void update_performance_stats(uint64_t execution_time_us, size_t symbols_processed, size_t vectors_processed);
    
    aerial::task::TaskResult validate_inputs(std::span<const aerial::tensor::TensorInfo> inputs) const;
    aerial::task::TaskResult validate_outputs(std::span<aerial::tensor::TensorInfo> outputs) const;
    
    // Algorithm implementations
    aerial::task::TaskResult execute_zf_detection_cuda(
        const std::complex<float>* d_H, const std::complex<float>* d_y,
        std::complex<float>* d_x_hat, size_t tx_antennas, size_t rx_antennas,
        size_t num_symbols, cudaStream_t stream);
    
    aerial::task::TaskResult execute_mmse_detection_cuda(
        const std::complex<float>* d_H, const std::complex<float>* d_y,
        std::complex<float>* d_x_hat, size_t tx_antennas, size_t rx_antennas,
        size_t num_symbols, float noise_var, cudaStream_t stream);
    
    // Linear algebra helpers
    aerial::task::TaskResult compute_matrix_inverse(
        const std::complex<float>* d_matrix, std::complex<float>* d_inverse,
        size_t matrix_size, cudaStream_t stream);
    
    aerial::task::TaskResult compute_qr_decomposition(
        const std::complex<float>* d_matrix, std::complex<float>* d_Q,
        std::complex<float>* d_R, size_t rows, size_t cols, cudaStream_t stream);
};

/// Factory for creating MIMO pipelines
class MIMOPipelineFactory {
public:
    static std::unique_ptr<MIMOPipeline> create_pipeline(
        const MIMOPipelineConfig& config = {});
    
    static std::unique_ptr<MIMOPipeline> create_from_spec(
        const aerial::pipeline::PipelineSpec& spec);
    
    static MIMOPipelineConfig get_default_config(
        size_t tx_antennas = 4, size_t rx_antennas = 4);
    
    static MIMOPipelineConfig get_high_performance_config(
        size_t tx_antennas = 8, size_t rx_antennas = 8);
    
    static MIMOPipelineConfig get_low_latency_config(
        size_t tx_antennas = 2, size_t rx_antennas = 2);
    
    static MIMOPipelineConfig get_massive_mimo_config(
        size_t tx_antennas = 16, size_t rx_antennas = 64);
    
    /// Get optimal configuration for specific antenna setup and algorithm
    static MIMOPipelineConfig get_optimized_config(
        size_t tx_antennas,
        size_t rx_antennas,
        MIMOAlgorithm algorithm,
        ModulationOrder modulation,
        bool prioritize_latency = false);
};

} // namespace mimo_detection