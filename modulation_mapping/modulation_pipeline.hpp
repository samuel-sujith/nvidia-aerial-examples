#pragma once

#include "modulator.hpp"
#include "pipeline/ipipeline.hpp"
#include "tensor/tensor_arena.hpp"
#include "task/task.hpp"
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace framework {
namespace examples {

/// Configuration for modulation pipeline
struct ModulationPipelineConfig {
    ModulationScheme modulation_order{ModulationScheme::QAM_16};
    size_t max_batch_size{1024};
    size_t max_symbols_per_batch{10000};
    bool enable_cuda_graphs{true};
    bool enable_profiling{false};
    int gpu_device_id{0};
    
    // Memory management
    size_t memory_pool_size{256 * 1024 * 1024}; // 256MB
    size_t memory_alignment{256};
    
    // Performance tuning
    int num_cuda_streams{2};
    size_t shared_memory_size{48 * 1024}; // 48KB
};

/// Performance statistics for modulation pipeline
struct ModulationPipelineStats {
    uint64_t total_symbols_processed{0};
    uint64_t total_batches_processed{0};
    uint64_t total_execution_time_us{0};
    uint64_t last_execution_time_us{0};
    uint64_t min_execution_time_us{UINT64_MAX};
    uint64_t max_execution_time_us{0};
    
    double average_throughput_msps() const {
        if (total_execution_time_us == 0) return 0.0;
        return (static_cast<double>(total_symbols_processed) / total_execution_time_us);
    }
    
    double average_latency_us() const {
        if (total_batches_processed == 0) return 0.0;
        return static_cast<double>(total_execution_time_us) / total_batches_processed;
    }
};

/// High-performance modulation pipeline with GPU optimization
class ModulationPipeline final : public framework::pipeline::IPipeline {
private:
    ModulationPipelineConfig config_;
    std::unique_ptr<GPUModulator> modulator_;
    std::unique_ptr<framework::tensor::TensorArena> memory_pool_;
    
    // CUDA resources
    cudaStream_t streams_[2];
    cudaEvent_t events_[4];
    cudaGraph_t cuda_graph_;
    cudaGraphExec_t graph_exec_;
    bool graph_created_{false};
    
    // Buffer management
    void* d_input_bits_{nullptr};
    void* d_output_symbols_{nullptr};
    void* d_temp_buffer_{nullptr};
    size_t current_buffer_size_{0};
    
    // Performance tracking
    ModulationPipelineStats stats_;
    std::chrono::high_resolution_clock::time_point last_start_time_;
    
    // State management
    bool is_initialized_{false};
    std::string pipeline_id_;
    
public:
    explicit ModulationPipeline(const ModulationPipelineConfig& config);
    ~ModulationPipeline();
    
    // IPipeline interface
    std::string_view get_pipeline_id() const override;
    std::size_t get_num_external_inputs() const override { return 1; }
    std::size_t get_num_external_outputs() const override { return 1; }
    
    framework::task::TaskResult execute_pipeline(
        std::span<const framework::tensor::TensorInfo> inputs,
        std::span<framework::tensor::TensorInfo> outputs,
        const framework::task::CancellationToken& token) override;
    
    framework::task::TaskResult execute_pipeline_graph(
        std::span<const framework::tensor::TensorInfo> inputs,
        std::span<framework::tensor::TensorInfo> outputs,
        const framework::task::CancellationToken& token) override;
    
    bool setup(const framework::pipeline::PipelineSpec& spec) override;
    void teardown() override;
    bool is_ready() const override { return is_initialized_; }
    
    framework::pipeline::PipelineStats get_stats() const override;
    
    // Modulation-specific interface
    ModulationPipelineStats get_modulation_stats() const { return stats_; }
    
    /// Process bits and generate modulated symbols
    framework::task::TaskResult modulate_bits(
        const std::vector<uint8_t>& input_bits,
        std::vector<std::complex<float>>& output_symbols,
        const framework::task::CancellationToken& token = {});
    
    /// Process symbols and generate demodulated bits  
    framework::task::TaskResult demodulate_symbols(
        const std::vector<std::complex<float>>& input_symbols,
        std::vector<uint8_t>& output_bits,
        const framework::task::CancellationToken& token = {});
    
    /// Batch processing for high throughput
    framework::task::TaskResult modulate_batch(
        const std::vector<std::vector<uint8_t>>& input_batches,
        std::vector<std::vector<cuComplex>>& output_batches,
        const framework::task::CancellationToken& token = {});
    
    /// Update configuration dynamically
    bool update_config(const ModulationPipelineConfig& new_config);
    
    /// Get current configuration
    const ModulationPipelineConfig& get_config() const { return config_; }
    
    /// Reset performance statistics
    void reset_stats();

private:
    bool initialize_cuda_resources();
    void cleanup_cuda_resources();
    bool ensure_buffer_capacity(size_t required_bits, size_t required_symbols);
    bool create_cuda_graph(size_t num_bits);
    void update_performance_stats(uint64_t execution_time_us, size_t symbols_processed);
    framework::task::TaskResult validate_inputs(std::span<const framework::tensor::TensorInfo> inputs) const;
    framework::task::TaskResult validate_outputs(std::span<framework::tensor::TensorInfo> outputs) const;
};

/// Factory for creating modulation pipelines
class ModulationPipelineFactory {
public:
    static std::unique_ptr<ModulationPipeline> create_pipeline(
        const ModulationPipelineConfig& config = {});
    
    static std::unique_ptr<ModulationPipeline> create_from_spec(
        const framework::pipeline::PipelineSpec& spec);
    
    static ModulationPipelineConfig get_default_config(ModulationScheme order);
    static ModulationPipelineConfig get_high_performance_config(ModulationScheme order);
    static ModulationPipelineConfig get_low_latency_config(ModulationScheme order);\n};\n\n} // namespace examples\n} // namespace framework
};

} // namespace modulation