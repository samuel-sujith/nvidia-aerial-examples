#include "fft_pipeline.hpp"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace fft_processing {

// Helper macro for cuFFT error checking
#define CUFFT_CHECK(call) do { \
    cufftResult result = call; \
    if (result != CUFFT_SUCCESS) { \
        throw std::runtime_error("cuFFT error: " + std::to_string(result)); \
    } \
} while(0)

FFTPipeline::FFTPipeline(const FFTPipelineConfig& config)
    : config_(config), pipeline_id_("fft_pipeline_v1") {
    
    // Generate unique pipeline ID
    std::stringstream ss;
    ss << "fft_pipeline_";
    for (size_t size : config_.fft_sizes) {
        ss << size << "_";
    }
    ss << config_.max_batch_size;
    pipeline_id_ = ss.str();
}

FFTPipeline::~FFTPipeline() {
    teardown();
}

bool FFTPipeline::setup(const aerial::pipeline::PipelineSpec& spec) {
    aerial::profiling::NvtxRange range("FFTPipeline::setup");
    
    try {
        // Set GPU device
        CUDA_CHECK(cudaSetDevice(config_.gpu_device_id));
        
        // Create FFT processor
        FFTConfig fft_config;
        fft_config.max_fft_size = *std::max_element(config_.fft_sizes.begin(), config_.fft_sizes.end());
        fft_config.max_batch_size = config_.max_batch_size;
        fft_config.precision = (config_.precision == FFTPrecision::Single) ? 
                              FFTPrecisionMode::Single : FFTPrecisionMode::Double;
        
        fft_processor_ = std::make_unique<FFTProcessor>(fft_config);
        if (!fft_processor_->initialize()) {
            return false;
        }
        
        // Initialize CUDA resources
        if (!initialize_cuda_resources()) {
            return false;
        }
        
        // Create cuFFT plans
        if (!create_fft_plans()) {
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

void FFTPipeline::teardown() {
    aerial::profiling::NvtxRange range("FFTPipeline::teardown");
    
    cleanup_cuda_resources();
    destroy_fft_plans();
    fft_processor_.reset();
    memory_pool_.reset();
    is_initialized_ = false;
}

bool FFTPipeline::initialize_cuda_resources() {
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

void FFTPipeline::cleanup_cuda_resources() {
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
    
    // Destroy streams
    for (int i = 0; i < config_.num_cuda_streams; ++i) {
        if (streams_[i]) {
            cudaStreamDestroy(streams_[i]);
            streams_[i] = nullptr;
        }
    }
    
    // Destroy events
    for (int i = 0; i < 8; ++i) {
        if (events_[i]) {
            cudaEventDestroy(events_[i]);
            events_[i] = nullptr;
        }
    }
    
    // Free device memory
    if (d_input_data_) {
        cudaFree(d_input_data_);
        d_input_data_ = nullptr;
    }
    if (d_output_data_) {
        cudaFree(d_output_data_);
        d_output_data_ = nullptr;
    }
    if (d_temp_buffer_) {
        cudaFree(d_temp_buffer_);
        d_temp_buffer_ = nullptr;
    }
    
    current_buffer_size_ = 0;
}

bool FFTPipeline::create_fft_plans() {
    try {
        for (size_t fft_size : config_.fft_sizes) {
            // Create forward FFT plan
            if (config_.operation_type == FFTType::Forward || config_.operation_type == FFTType::Both) {
                cufftHandle forward_plan;
                CUFFT_CHECK(cufftPlan1d(&forward_plan, fft_size, CUFFT_C2C, config_.max_batch_size));
                fft_plans_[fft_size] = forward_plan;
            }
            
            // Create inverse FFT plan  
            if (config_.operation_type == FFTType::Inverse || config_.operation_type == FFTType::Both) {
                cufftHandle inverse_plan;
                CUFFT_CHECK(cufftPlan1d(&inverse_plan, fft_size, CUFFT_C2C, config_.max_batch_size));
                ifft_plans_[fft_size] = inverse_plan;
            }
        }
        
        return true;
    } catch (const std::exception& e) {
        destroy_fft_plans();
        return false;
    }
}

void FFTPipeline::destroy_fft_plans() {
    for (auto& [size, plan] : fft_plans_) {
        cufftDestroy(plan);
    }
    fft_plans_.clear();
    
    for (auto& [size, plan] : ifft_plans_) {
        cufftDestroy(plan);
    }
    ifft_plans_.clear();
}

bool FFTPipeline::ensure_buffer_capacity(size_t required_samples) {
    size_t sample_bytes = required_samples * sizeof(cufftComplex);
    size_t total_required = sample_bytes * 3; // input + output + temp
    
    if (total_required <= current_buffer_size_) {
        return true;
    }
    
    // Free existing buffers
    if (d_input_data_) cudaFree(d_input_data_);
    if (d_output_data_) cudaFree(d_output_data_);
    if (d_temp_buffer_) cudaFree(d_temp_buffer_);
    
    // Allocate new buffers with extra capacity
    size_t buffer_multiplier = 2;
    size_t new_size = sample_bytes * buffer_multiplier;
    
    CUDA_CHECK(cudaMalloc(&d_input_data_, new_size));
    CUDA_CHECK(cudaMalloc(&d_output_data_, new_size));
    CUDA_CHECK(cudaMalloc(&d_temp_buffer_, new_size));
    
    current_buffer_size_ = new_size * 3;
    
    return true;
}

aerial::task::TaskResult FFTPipeline::execute_pipeline(
    std::span<const aerial::tensor::TensorInfo> inputs,
    std::span<aerial::tensor::TensorInfo> outputs,
    const aerial::task::CancellationToken& token) {
    
    aerial::profiling::NvtxRange range("FFTPipeline::execute_pipeline");
    
    if (!is_initialized_) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed, 
                                      "Pipeline not initialized");
    }
    
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
        const auto& input_tensor = inputs[0];
        auto& output_tensor = outputs[0];
        
        size_t total_samples = input_tensor.num_elements();
        
        // Determine FFT size and batch size from input dimensions
        const auto& dims = input_tensor.dimensions();
        size_t fft_size = dims[dims.size() - 1]; // Last dimension is FFT size
        size_t batch_size = total_samples / fft_size;
        
        if (!is_fft_size_supported(fft_size)) {
            return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                          "Unsupported FFT size");
        }
        
        // Ensure buffer capacity
        if (!ensure_buffer_capacity(total_samples)) {
            return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                          "Failed to allocate GPU memory");
        }
        
        // Copy input data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_input_data_, input_tensor.data(),
                                   total_samples * sizeof(cufftComplex),
                                   cudaMemcpyHostToDevice, streams_[0]));
        
        // Record start event
        CUDA_CHECK(cudaEventRecord(events_[0], streams_[0]));
        
        // Execute FFT
        if (token.is_cancellation_requested()) {
            return aerial::task::TaskResult(aerial::task::TaskStatus::Cancelled);
        }
        
        cufftHandle plan = fft_plans_[fft_size];
        CUFFT_CHECK(cufftSetStream(plan, streams_[0]));
        CUFFT_CHECK(cufftExecC2C(plan, 
                                 static_cast<cufftComplex*>(d_input_data_),
                                 static_cast<cufftComplex*>(d_output_data_),
                                 CUFFT_FORWARD));
        
        // Record end event
        CUDA_CHECK(cudaEventRecord(events_[1], streams_[0]));
        
        // Copy output data back to host
        CUDA_CHECK(cudaMemcpyAsync(output_tensor.data(), d_output_data_,
                                   total_samples * sizeof(cufftComplex),
                                   cudaMemcpyDeviceToHost, streams_[0]));
        
        // Synchronize
        CUDA_CHECK(cudaStreamSynchronize(streams_[0]));
        
        // Calculate execution time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        
        update_performance_stats(duration.count(), total_samples, FFTType::Forward);
        
        return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
        
    } catch (const std::exception& e) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed, e.what());
    }
}

aerial::task::TaskResult FFTPipeline::execute_pipeline_graph(
    std::span<const aerial::tensor::TensorInfo> inputs,
    std::span<aerial::tensor::TensorInfo> outputs,
    const aerial::task::CancellationToken& token) {
    
    if (!config_.enable_cuda_graphs) {
        return execute_pipeline(inputs, outputs, token);
    }
    
    aerial::profiling::NvtxRange range("FFTPipeline::execute_pipeline_graph");
    
    // Implementation similar to regular execute_pipeline but using CUDA graphs
    // This is a simplified version - full implementation would cache graphs
    return execute_pipeline(inputs, outputs, token);
}

bool FFTPipeline::is_fft_size_supported(size_t fft_size) const {
    return std::find(config_.fft_sizes.begin(), config_.fft_sizes.end(), fft_size) != 
           config_.fft_sizes.end();
}

aerial::task::TaskResult FFTPipeline::validate_inputs(
    std::span<const aerial::tensor::TensorInfo> inputs) const {
    
    if (inputs.size() != 1) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Expected exactly 1 input tensor");
    }
    
    const auto& input = inputs[0];
    if (input.element_type() != aerial::tensor::ElementType::COMPLEX_FLOAT32) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Input tensor must be COMPLEX_FLOAT32 type");
    }
    
    return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
}

aerial::task::TaskResult FFTPipeline::validate_outputs(
    std::span<aerial::tensor::TensorInfo> outputs) const {
    
    if (outputs.size() != 1) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Expected exactly 1 output tensor");
    }
    
    const auto& output = outputs[0];
    if (output.element_type() != aerial::tensor::ElementType::COMPLEX_FLOAT32) {
        return aerial::task::TaskResult(aerial::task::TaskStatus::Failed,
                                      "Output tensor must be COMPLEX_FLOAT32 type");
    }
    
    return aerial::task::TaskResult(aerial::task::TaskStatus::Completed);
}

void FFTPipeline::update_performance_stats(uint64_t execution_time_us, size_t samples_processed, FFTType operation) {
    stats_.total_samples_processed += samples_processed;
    stats_.total_batches_processed++;
    stats_.total_execution_time_us += execution_time_us;
    stats_.last_execution_time_us = execution_time_us;
    stats_.min_execution_time_us = std::min(stats_.min_execution_time_us, execution_time_us);
    stats_.max_execution_time_us = std::max(stats_.max_execution_time_us, execution_time_us);
    
    if (operation == FFTType::Forward) {
        stats_.forward_ffts++;
        stats_.forward_time_us += execution_time_us;
    } else if (operation == FFTType::Inverse) {
        stats_.inverse_ffts++;
        stats_.inverse_time_us += execution_time_us;
    }
    
    stats_.total_ffts_processed++;
}

aerial::pipeline::PipelineStats FFTPipeline::get_stats() const {
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

void FFTPipeline::reset_stats() {
    stats_ = FFTPipelineStats{};
}

// Factory implementations
std::unique_ptr<FFTPipeline> FFTPipelineFactory::create_pipeline(
    const FFTPipelineConfig& config) {
    return std::make_unique<FFTPipeline>(config);
}

FFTPipelineConfig FFTPipelineFactory::get_default_config(const std::vector<size_t>& fft_sizes) {
    FFTPipelineConfig config;
    config.fft_sizes = fft_sizes;
    config.max_batch_size = 64;
    config.operation_type = FFTType::Both;
    config.precision = FFTPrecision::Single;
    config.enable_cuda_graphs = true;
    return config;
}

} // namespace fft_processing
    std::size_t get_num_external_inputs() const override { return 1; }
    std::size_t get_num_external_outputs() const override { return 1; }

    task::TaskResult execute_pipeline(
        std::span<const tensor::TensorInfo> inputs,
        std::span<tensor::TensorInfo> outputs,
        const task::CancellationToken& token
    ) override;

    task::TaskResult execute_pipeline_graph(
        std::span<const tensor::TensorInfo> inputs,
        std::span<tensor::TensorInfo> outputs,
        const task::CancellationToken& token
    ) override;

    bool setup(const pipeline::PipelineSpec& spec) override;
    void teardown() override;
    bool is_ready() const override;
    pipeline::PipelineStats get_stats() const override;

private:
    std::string pipeline_id_;
    std::vector<std::unique_ptr<FFTModule>> fft_modules_;
    pipeline::PipelineStats stats_;
    
    // Intermediate buffers for multi-stage processing
    std::vector<cuComplex*> intermediate_buffers_;
    size_t buffer_size_;
    
    void allocate_intermediate_buffers();
    void deallocate_intermediate_buffers();
};

} // namespace aerial::examples