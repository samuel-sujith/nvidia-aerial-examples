#include "fft_processing_pipeline.hpp"

#include <stdexcept>
#include <cuda_runtime.h>

namespace fft_processing {

FFTProcessingPipeline::FFTProcessingPipeline(
    std::string pipeline_id,
    const FFTParams& params
) : pipeline_id_(std::move(pipeline_id)), 
    fft_params_(params) {
    
    // Create the FFT processor module
    fft_processor_ = std::make_unique<FFTProcessor>(pipeline_id_ + "_processor", fft_params_);
    
    allocate_pipeline_memory();
}

void FFTProcessingPipeline::setup() {
    if (!fft_processor_) {
        throw std::runtime_error("FFT processor not initialized");
    }
    
    // Allocate pipeline memory
    allocate_pipeline_memory();
    
    // Setup module memory slice
    framework::pipeline::ModuleMemorySlice memory_slice;
    // In a real implementation, this would provide the proper memory slice
    
    fft_processor_->setup_memory(memory_slice);
    
    // Setup tensor connections
    setup_tensor_connections();
}

void FFTProcessingPipeline::warmup(cudaStream_t stream) {
    if (fft_processor_) {
        fft_processor_->warmup(stream);
    }
}

void FFTProcessingPipeline::configure_io(
    const framework::pipeline::DynamicParams& params,
    std::span<const framework::pipeline::PortInfo> external_inputs,
    std::span<framework::pipeline::PortInfo> external_outputs,
    cudaStream_t stream
) {
    if (external_inputs.size() != 1) {
        throw std::runtime_error("FFT processing pipeline requires exactly 1 external input");
    }
    
    if (external_outputs.size() != 1) {
        throw std::runtime_error("FFT processing pipeline requires exactly 1 external output");
    }
    
    // Set module inputs using PortInfo directly
    fft_processor_->set_inputs(external_inputs);
    
    // Call module's configure_io
    fft_processor_->configure_io(params, stream);
    
    // Get module outputs and update external outputs
    auto module_output_ports = fft_processor_->get_outputs();
    for (size_t i = 0; i < external_outputs.size() && i < module_output_ports.size(); ++i) {
        external_outputs[i] = module_output_ports[i];
    }
}

void FFTProcessingPipeline::execute_stream(cudaStream_t stream) {
    if (fft_processor_) {
        fft_processor_->execute(stream);
    }
}

void FFTProcessingPipeline::execute_graph(cudaStream_t stream) {
    // For this example, graph mode is the same as stream mode
    // In a real implementation, this would use CUDA graphs
    if (fft_processor_) {
        fft_processor_->execute(stream);
    }
}

void FFTProcessingPipeline::allocate_pipeline_memory() {
    // For this example, we'll use a simple fixed memory allocation
    // In a real implementation, you would query module requirements
    constexpr size_t MEMORY_SIZE = 16 * 1024 * 1024; // 16MB
    memory_size_ = MEMORY_SIZE;
    
    cudaError_t err = cudaMalloc(&device_memory_, memory_size_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pipeline device memory");
    }

}

void FFTProcessingPipeline::setup_tensor_connections() {
    // Setup internal tensor connections between modules
    // For this simple pipeline with one module, no internal connections needed
    
    // In a multi-module pipeline, this would connect outputs of one module
    // to inputs of another module
}

} // namespace fft_processing