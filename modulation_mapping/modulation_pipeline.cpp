#include "modulation_pipeline.hpp"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>

// Simple CUDA error check macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)

namespace modulation {

ModulationPipeline::ModulationPipeline(const ModulationPipelineConfig& config)
    : config_(config), pipeline_id_("modulation_pipeline_v1") {
    
    // Generate unique pipeline ID
    std::stringstream ss;
    ss << "modulation_pipeline_" << static_cast<int>(config_.modulation_order) 
       << "_" << config_.max_batch_size;
    pipeline_id_ = ss.str();
}

ModulationPipeline::~ModulationPipeline() {
    teardown();
}

std::string_view ModulationPipeline::get_pipeline_id() const {
    return pipeline_id_;
}

bool ModulationPipeline::setup(const ::framework::pipeline::PipelineSpec& spec) {
    // Profiling removed for simplification
    // aerial::profiling::NvtxRange range("ModulationPipeline::setup");
    
    try {
        // Set GPU device
        CUDA_CHECK(cudaSetDevice(config_.gpu_device_id));
        
        // Create modulator
        ModulationParams mod_config;
        mod_config.scheme = config_.modulation_scheme;
        mod_config.enable_fast_math = true;
        mod_config.use_lookup_tables = true;
        
        modulator_ = std::make_unique<::aerial::examples::QAMModulator>("modulator", mod_config);
        // QAMModulator is ready after construction
        
        // Initialize CUDA resources
        if (!initialize_cuda_resources()) {
            return false;
        }
        
        // Create memory pool
        // Memory pool creation removed for simplification
        // memory_pool_ = std::make_unique<::framework::memory::MemoryPool>(
            config_.memory_pool_size
        
        is_initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        teardown();
        return false;
    }
}

void ModulationPipeline::teardown() {
    // Profiling removed for simplification
    // aerial::profiling::NvtxRange range("ModulationPipeline::teardown");
    
    cleanup_cuda_resources();
    modulator_.reset();
    // Memory cleanup handled by destructor
    is_initialized_ = false;
}

bool ModulationPipeline::initialize_cuda_resources() {
    // Create CUDA streams
    for (int i = 0; i < config_.num_cuda_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
    }
    
    // Create CUDA events
    for (int i = 0; i < 4; ++i) {
        CUDA_CHECK(cudaEventCreate(&events_[i]));
    }
    
    graph_created_ = false;
    cuda_graph_ = nullptr;
    graph_exec_ = nullptr;
    
    return true;
}

void ModulationPipeline::cleanup_cuda_resources() {
    // Destroy CUDA graph
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
        cuda_graph_ = nullptr;
    }
    graph_created_ = false;
    
    // Destroy streams
    for (int i = 0; i < config_.num_cuda_streams; ++i) {
        if (streams_[i]) {
            cudaStreamDestroy(streams_[i]);
            streams_[i] = nullptr;
        }
    }
    
    // Destroy events
    for (int i = 0; i < 4; ++i) {
        if (events_[i]) {
            cudaEventDestroy(events_[i]);
            events_[i] = nullptr;
        }
    }
    
    // Free device memory
    if (d_input_bits_) {
        cudaFree(d_input_bits_);
        d_input_bits_ = nullptr;
    }
    if (d_output_symbols_) {
        cudaFree(d_output_symbols_);
        d_output_symbols_ = nullptr;
    }
    if (d_temp_buffer_) {
        cudaFree(d_temp_buffer_);
        d_temp_buffer_ = nullptr;
    }
    
    current_buffer_size_ = 0;
}

bool ModulationPipeline::ensure_buffer_capacity(size_t required_bits, size_t required_symbols) {
    size_t bits_bytes = required_bits * sizeof(uint8_t);
    size_t symbols_bytes = required_symbols * sizeof(std::complex<float>);
    size_t temp_bytes = std::max(bits_bytes, symbols_bytes);
    
    size_t total_required = bits_bytes + symbols_bytes + temp_bytes;
    
    if (total_required <= current_buffer_size_) {
        return true; // Current buffers are sufficient
    }
    
    // Free existing buffers
    if (d_input_bits_) cudaFree(d_input_bits_);
    if (d_output_symbols_) cudaFree(d_output_symbols_);
    if (d_temp_buffer_) cudaFree(d_temp_buffer_);
    
    // Allocate new buffers with some extra capacity
    size_t buffer_multiplier = 2; // 100% extra capacity
    size_t new_bits_size = bits_bytes * buffer_multiplier;
    size_t new_symbols_size = symbols_bytes * buffer_multiplier;
    size_t new_temp_size = temp_bytes * buffer_multiplier;
    
    CUDA_CHECK(cudaMalloc(&d_input_bits_, new_bits_size));
    CUDA_CHECK(cudaMalloc(&d_output_symbols_, new_symbols_size));
    CUDA_CHECK(cudaMalloc(&d_temp_buffer_, new_temp_size));
    
    current_buffer_size_ = new_bits_size + new_symbols_size + new_temp_size;
    
    return true;
}

::framework::task::TaskResult ModulationPipeline::execute_pipeline(
    std::span<const ::framework::tensor::TensorInfo> inputs,
    std::span<::framework::tensor::TensorInfo> outputs,
    const ::framework::task::CancellationToken& token) {
    
    // Profiling removed for simplification
    // aerial::profiling::NvtxRange range("ModulationPipeline::execute_pipeline");
    
    if (!is_initialized_) {
        return ::framework::task::TaskResult(::framework::task::TaskStatus::Failed, 
                                      "Pipeline not initialized");
    }
    
    // Validate inputs/outputs
    auto validation_result = validate_inputs(inputs);
    if (!validation_result.is_success()) {
        return validation_result;
    }
    
    validation_result = validate_outputs(outputs);
    if (!validation_result.is_success()) {
        return validation_result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    last_start_time_ = start_time;
    
    try {
        // Extract input data
        const auto& input_tensor = inputs[0];
        auto& output_tensor = outputs[0];
        
        size_t num_bits = input_tensor.tensor_elements();
        size_t num_symbols = modulator_->calculate_output_symbols(num_bits);
        
        // Ensure buffer capacity
        if (!ensure_buffer_capacity(num_bits, num_symbols)) {
            return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                          "Failed to allocate GPU memory");
        }
        
        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_input_bits_, input_tensor.data(), 
                                   num_bits * sizeof(uint8_t), 
                                   cudaMemcpyHostToDevice, streams_[0]));
        
        // Record start event
        CUDA_CHECK(cudaEventRecord(events_[0], streams_[0]));
        
        // Execute modulation
        // Process without cancellation check
        
        auto modulation_result = modulator_->modulate(
            static_cast<uint8_t*>(d_input_bits_), num_bits,
            static_cast<std::complex<float>*>(d_output_symbols_), streams_[0]);
        
        if (!modulation_result.is_success()) {
            return modulation_result;
        }
        
        // Record end event
        CUDA_CHECK(cudaEventRecord(events_[1], streams_[0]));
        
        // Copy output data back to host
        CUDA_CHECK(cudaMemcpyAsync(output_tensor.data(), d_output_symbols_,
                                   num_symbols * sizeof(std::complex<float>),
                                   cudaMemcpyDeviceToHost, streams_[0]));
        
        // Synchronize
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
        
        // Calculate execution time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        update_performance_stats(duration.count(), num_symbols);
        
        return framework::task::TaskResult(framework::task::TaskStatus::Completed);
        
    } catch (const std::exception& e) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed, e.what());
    }
}

framework::task::TaskResult ModulationPipeline::execute_pipeline_graph(
    std::span<const aerial::tensor::TensorInfo> inputs,
    std::span<aerial::tensor::TensorInfo> outputs,
    const framework::task::CancellationToken& token) {
    
    if (!config_.enable_cuda_graphs) {
        return execute_pipeline(inputs, outputs, token);
    }
    
    // Profiling removed for simplification
    // aerial::profiling::NvtxRange range("ModulationPipeline::execute_pipeline_graph");
    
    const auto& input_tensor = inputs[0];
    size_t num_bits = input_tensor.num_elements();
    
    // Create CUDA graph if not exists or size changed
    if (!graph_created_ || !graph_exec_) {
        if (!create_cuda_graph(num_bits)) {
            return execute_pipeline(inputs, outputs, token); // Fallback
        }
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Copy input data
        CUDA_CHECK(cudaMemcpyAsync(d_input_bits_, input_tensor.data(),
                                   num_bits * sizeof(uint8_t),
                                   cudaMemcpyHostToDevice, streams_[0]));
        
        // Launch CUDA graph
        CUDA_CHECK(cudaGraphLaunch(graph_exec_, streams_[0]));
        
        // Copy output data
        size_t num_symbols = modulator_->calculate_output_symbols(num_bits);
        auto& output_tensor = outputs[0];
        CUDA_CHECK(cudaMemcpyAsync(output_tensor.data(), d_output_symbols_,
                                   num_symbols * sizeof(std::complex<float>),
                                   cudaMemcpyDeviceToHost, streams_[0]));
        
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        update_performance_stats(duration.count(), num_symbols);
        
        return framework::task::TaskResult(framework::task::TaskStatus::Completed);
        
    } catch (const std::exception& e) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed, e.what());
    }
}

bool ModulationPipeline::create_cuda_graph(size_t num_bits) {
    try {
        // Cleanup existing graph
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (cuda_graph_) {
            cudaGraphDestroy(cuda_graph_);
            cuda_graph_ = nullptr;
        }
        
        size_t num_symbols = modulator_->calculate_output_symbols(num_bits);
        
        // Ensure buffer capacity
        if (!ensure_buffer_capacity(num_bits, num_symbols)) {
            return false;
        }
        
        // Begin graph capture
        CUDA_CHECK(cudaStreamBeginCapture(streams_[0], cudaStreamCaptureModeGlobal));
        
        // Record modulation operation
        auto result = modulator_->modulate(
            static_cast<uint8_t*>(d_input_bits_), num_bits,
            static_cast<std::complex<float>*>(d_output_symbols_), streams_[0]);
        
        // End graph capture
        CUDA_CHECK(cudaStreamEndCapture(streams_[0], &cuda_graph_));
        
        // Create executable graph
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, cuda_graph_, nullptr, nullptr, 0));
        
        graph_created_ = true;
        return true;
        
    } catch (const std::exception& e) {
        graph_created_ = false;
        return false;
    }
}

framework::task::TaskResult ModulationPipeline::modulate_bits(
    const std::vector<uint8_t>& input_bits,
    std::vector<std::complex<float>>& output_symbols,
    const framework::task::CancellationToken& token) {
    
    if (!is_initialized_) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Pipeline not initialized");
    }
    
    size_t num_symbols = modulator_->calculate_output_symbols(input_bits.size());
    output_symbols.resize(num_symbols);
    
    // Create tensor info for pipeline execution
    aerial::tensor::TensorInfo input_tensor;
    input_tensor.set_data(const_cast<uint8_t*>(input_bits.data()));
    input_tensor.set_dimensions({input_bits.size()});
    input_tensor.set_element_type(aerial::tensor::ElementType::UINT8);
    
    aerial::tensor::TensorInfo output_tensor;
    output_tensor.set_data(output_symbols.data());
    output_tensor.set_dimensions({num_symbols});
    output_tensor.set_element_type(aerial::tensor::ElementType::COMPLEX_FLOAT32);
    
    std::array<aerial::tensor::TensorInfo, 1> inputs = {input_tensor};
    std::array<aerial::tensor::TensorInfo, 1> outputs = {output_tensor};
    
    return execute_pipeline(inputs, outputs, token);
}

framework::task::TaskResult ModulationPipeline::validate_inputs(
    std::span<const aerial::tensor::TensorInfo> inputs) const {
    
    if (inputs.size() != 1) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Expected exactly 1 input tensor");
    }
    
    const auto& input = inputs[0];
    if (input.element_type() != aerial::tensor::ElementType::UINT8) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Input tensor must be UINT8 type");
    }
    
    if (input.dimensions().size() != 1) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Input tensor must be 1-dimensional");
    }
    
    return framework::task::TaskResult(framework::task::TaskStatus::Completed);
}

framework::task::TaskResult ModulationPipeline::validate_outputs(
    std::span<aerial::tensor::TensorInfo> outputs) const {
    
    if (outputs.size() != 1) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Expected exactly 1 output tensor");
    }
    
    const auto& output = outputs[0];
    if (output.element_type() != aerial::tensor::ElementType::COMPLEX_FLOAT32) {
        return framework::task::TaskResult(framework::task::TaskStatus::Failed,
                                      "Output tensor must be COMPLEX_FLOAT32 type");
    }
    
    return framework::task::TaskResult(framework::task::TaskStatus::Completed);
}

void ModulationPipeline::update_performance_stats(uint64_t execution_time_us, size_t symbols_processed) {
    stats_.total_symbols_processed += symbols_processed;
    stats_.total_batches_processed++;
    stats_.total_execution_time_us += execution_time_us;
    stats_.last_execution_time_us = execution_time_us;
    stats_.min_execution_time_us = std::min(stats_.min_execution_time_us, execution_time_us);
    stats_.max_execution_time_us = std::max(stats_.max_execution_time_us, execution_time_us);
}

framework::pipeline::PipelineStats ModulationPipeline::get_stats() const {
    framework::pipeline::PipelineStats base_stats;
    base_stats.pipeline_id = pipeline_id_;
    base_stats.total_executions = stats_.total_batches_processed;
    base_stats.successful_executions = stats_.total_batches_processed; // Assuming all successful for now
    base_stats.total_execution_time_us = stats_.total_execution_time_us;
    base_stats.last_execution_time_us = stats_.last_execution_time_us;
    base_stats.min_execution_time_us = stats_.min_execution_time_us;
    base_stats.max_execution_time_us = stats_.max_execution_time_us;
    
    return base_stats;
}

void ModulationPipeline::reset_stats() {
    stats_ = ModulationPipelineStats{};
}

// Factory implementations
std::unique_ptr<ModulationPipeline> ModulationPipelineFactory::create_pipeline(
    const ModulationPipelineConfig& config) {
    return std::make_unique<ModulationPipeline>(config);
}

ModulationPipelineConfig ModulationPipelineFactory::get_default_config(ModulationOrder order) {
    ModulationPipelineConfig config;
    config.modulation_order = order;
    config.max_batch_size = 1024;
    config.max_symbols_per_batch = 10000;
    config.enable_cuda_graphs = true;
    return config;
}

ModulationPipelineConfig ModulationPipelineFactory::get_high_performance_config(ModulationOrder order) {
    ModulationPipelineConfig config;
    config.modulation_order = order;
    config.max_batch_size = 4096;
    config.max_symbols_per_batch = 100000;
    config.enable_cuda_graphs = true;
    config.num_cuda_streams = 4;
    config.memory_pool_size = 1024 * 1024 * 1024; // 1GB
    return config;
}

ModulationPipelineConfig ModulationPipelineFactory::get_low_latency_config(ModulationOrder order) {
    ModulationPipelineConfig config;
    config.modulation_order = order;
    config.max_batch_size = 256;
    config.max_symbols_per_batch = 1000;
    config.enable_cuda_graphs = true;
    config.num_cuda_streams = 1;
    config.shared_memory_size = 16 * 1024; // 16KB
    return config;
}

} // namespace modulation