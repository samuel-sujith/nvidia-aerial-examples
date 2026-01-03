#include "mimo_pipeline.hpp"
#include <aerial/cuda_utils/CudaGraphHelper.hpp>
#include <aerial/profiling/NvtxRange.hpp>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace mimo_detection {

// Helper macros for error checking
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error("cuSOLVER error: " + std::to_string(status)); \
    } \
} while(0)

MIMOPipeline::MIMOPipeline(const MIMOPipelineConfig& config)
    : config_(config), pipeline_id_("mimo_pipeline_v1") {
    
    // Generate unique pipeline ID
    std::stringstream ss;
    ss << "mimo_pipeline_" << config_.max_tx_antennas << "x" << config_.max_rx_antennas
       << "_" << static_cast<int>(config_.detection_algorithm) 
       << "_" << config_.max_batch_size;
    pipeline_id_ = ss.str();
}

MIMOPipeline::~MIMOPipeline() {
    teardown();
}

std::string_view MIMOPipeline::get_pipeline_id() const {
    return pipeline_id_;
}

bool MIMOPipeline::setup(const aerial::pipeline::PipelineSpec& spec) {
    aerial::profiling::NvtxRange range("MIMOPipeline::setup");
    
    try {
        // Set GPU device
        CUDA_CHECK(cudaSetDevice(config_.gpu_device_id));
        
        // Create MIMO detector
        MIMOConfig mimo_config;
        mimo_config.max_tx_antennas = config_.max_tx_antennas;
        mimo_config.max_rx_antennas = config_.max_rx_antennas;
        mimo_config.detection_algorithm = config_.detection_algorithm;
        mimo_config.modulation_order = config_.modulation_order;
        
        mimo_detector_ = std::make_unique<MIMODetector>(mimo_config);
        if (!mimo_detector_->initialize()) {
            return false;
        }
        
        // Initialize CUDA resources
        if (!initialize_cuda_resources()) {
            return false;
        }
        
        // Create memory pool
        memory_pool_ = std::make_unique<aerial::memory::MemoryPool>(
            config_.memory_pool_size);
        
        is_initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        teardown();
        return false;
    }
}

void MIMOPipeline::teardown() {
    aerial::profiling::NvtxRange range("MIMOPipeline::teardown");
    
    cleanup_cuda_resources();
    mimo_detector_.reset();
    memory_pool_.reset();
    is_initialized_ = false;
}

bool MIMOPipeline::initialize_cuda_resources() {
    // Create cuBLAS handle
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    
    // Create cuSOLVER handle
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle_));
    
    // Create CUDA streams
    for (int i = 0; i < config_.num_cuda_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
    
    // Create CUDA events
    for (int i = 0; i < 8; ++i) {
        CUDA_CHECK(cudaEventCreate(&events_[i]));
    }
    
    return true;
}

void MIMOPipeline::cleanup_cuda_resources() {
    // Destroy CUDA graphs
    for (auto& [key, graph_exec] : graph_execs_) {
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
        }
    }
    graph_execs_.clear();
    
    for (auto& [key, graph] : cuda_graphs_) {
        if (graph) {
            cudaGraphDestroy(graph);
        }
    }
    cuda_graphs_.clear();
    
    // Destroy handles
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
        cublas_handle_ = nullptr;
    }
    
    if (cusolver_handle_) {
        cusolverDnDestroy(cusolver_handle_);
        cusolver_handle_ = nullptr;
    }
    
    // Free device memory and other resources
    if (d_channel_matrix_) {
        cudaFree(d_channel_matrix_);
        d_channel_matrix_ = nullptr;
    }
    
    current_buffer_size_ = 0;
}

aerial::task::TaskResult MIMOPipeline::execute_pipeline(
    std::span<const aerial::tensor::TensorInfo> inputs,
    std::span<aerial::tensor::TensorInfo> outputs,
    const aerial::task::CancellationToken& token) {
    
    aerial::profiling::NvtxRange range("MIMOPipeline::execute_pipeline");
    
    if (!is_initialized_) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed, 
                                      "Pipeline not initialized");
    }
    
    return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
}

aerial::task::TaskResult MIMOPipeline::execute_pipeline_graph(
    std::span<const aerial::tensor::TensorInfo> inputs,
    std::span<aerial::tensor::TensorInfo> outputs,
    const aerial::task::CancellationToken& token) {
    
    return execute_pipeline(inputs, outputs, token);
}

aerial::task::TaskResult MIMOPipeline::validate_inputs(
    std::span<const aerial::tensor::TensorInfo> inputs) const {
    
    if (inputs.size() != 2) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Expected exactly 2 input tensors");
    }
    
    return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
}

aerial::task::TaskResult MIMOPipeline::validate_outputs(
    std::span<aerial::tensor::TensorInfo> outputs) const {
    
    if (outputs.size() != 1) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Expected exactly 1 output tensor");
    }
    
    return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
}

void MIMOPipeline::update_performance_stats(uint64_t execution_time_us, size_t symbols_processed, size_t vectors_processed) {
    stats_.total_symbols_processed += symbols_processed;
    stats_.total_vectors_processed += vectors_processed;
    stats_.total_batches_processed++;
    stats_.total_execution_time_us += execution_time_us;
    stats_.last_execution_time_us = execution_time_us;
    stats_.min_execution_time_us = std::min(stats_.min_execution_time_us, execution_time_us);
    stats_.max_execution_time_us = std::max(stats_.max_execution_time_us, execution_time_us);
}

aerial::pipeline::PipelineStats MIMOPipeline::get_stats() const {
    aerial::pipeline::PipelineStats base_stats;
    base_stats.pipeline_id = pipeline_id_;
    base_stats.total_executions = stats_.total_batches_processed;
    base_stats.successful_executions = stats_.total_batches_processed;
    base_stats.total_execution_time_us = stats_.total_execution_time_us;
    base_stats.last_execution_time_us = stats_.last_execution_time_us;
    base_stats.min_execution_time_us = stats_.min_execution_time_us;
    base_stats.max_execution_time_us = stats_.max_execution_time_us;
    
    return base_stats;
}

void MIMOPipeline::reset_stats() {
    stats_ = MIMOPipelineStats{};
}

// Factory implementations
std::unique_ptr<MIMOPipeline> MIMOPipelineFactory::create_pipeline(
    const MIMOPipelineConfig& config) {
    return std::make_unique<MIMOPipeline>(config);
}

MIMOPipelineConfig MIMOPipelineFactory::get_default_config(size_t tx_antennas, size_t rx_antennas) {
    MIMOPipelineConfig config;
    config.max_tx_antennas = tx_antennas;
    config.max_rx_antennas = rx_antennas;
    config.detection_algorithm = MIMOAlgorithm::MMSE;
    config.modulation_order = ModulationOrder::QAM16;
    config.max_batch_size = 64;
    config.enable_cuda_graphs = true;
    return config;
}

} // namespace mimo_detection