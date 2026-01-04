#include "fft_processing_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

using namespace framework::tensor;

namespace fft_processing {

// CUDA kernels for FFT preprocessing and postprocessing
__global__ void apply_windowing_kernel(const FFTDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = desc->params->fft_size * desc->params->num_antennas * desc->params->batch_size;
    
    if (idx < total_samples) {
        int sample_idx = idx % desc->params->fft_size;
        float window_val = desc->window_function ? desc->window_function[sample_idx] : 1.0f;
        
        cuComplex input_val = desc->input_data[idx];
        cuComplex windowed_val = make_cuComplex(
            cuCrealf(input_val) * window_val,
            cuCimagf(input_val) * window_val
        );
        
        // Store in output buffer for FFT processing
        desc->output_data[idx] = windowed_val;
    }
}

__global__ void normalize_ifft_kernel(const FFTDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = desc->params->fft_size * desc->params->num_antennas * desc->params->batch_size;
    
    if (idx < total_samples) {
        float norm_factor = 1.0f / desc->params->fft_size;
        cuComplex output_val = desc->output_data[idx];
        
        desc->output_data[idx] = make_cuComplex(
            cuCrealf(output_val) * norm_factor,
            cuCimagf(output_val) * norm_factor
        );
    }
}

__global__ void copy_input_kernel(const FFTDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_samples = desc->params->fft_size * desc->params->num_antennas * desc->params->batch_size;
    
    if (idx < total_samples) {
        desc->output_data[idx] = desc->input_data[idx];
    }
}

FFTProcessor::FFTProcessor(const std::string& module_id, const FFTParams& params)
    : module_id_(module_id), params_(params), fft_plan_(0), d_descriptor_(nullptr) {
    
    // Validate parameters
    if (params_.fft_size <= 0 || (params_.fft_size & (params_.fft_size - 1)) != 0) {
        throw std::invalid_argument("FFT size must be a positive power of 2");
    }
    
    if (params_.num_antennas <= 0 || params_.batch_size <= 0) {
        throw std::invalid_argument("Number of antennas and batch size must be positive");
    }
    
    // Initialize descriptor
    h_descriptor_ = {};
    h_descriptor_.params = &params_;
    h_descriptor_.total_samples = params_.fft_size * params_.num_antennas * params_.batch_size;
    
    allocate_gpu_memory();
    setup_port_info();
    create_fft_plan();
    
    if (params_.enable_windowing) {
        setup_windowing();
    }
}

FFTProcessor::~FFTProcessor() {
    destroy_fft_plan();
    deallocate_gpu_memory();
}

void FFTProcessor::setup_port_info() {
    using namespace framework::tensor;
    
    // Setup input ports
    input_ports_.resize(1);
    
    // Input port: input_signal
    input_ports_[0].name = "input_signal";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.batch_size),
            static_cast<std::size_t>(params_.num_antennas),
            static_cast<std::size_t>(params_.fft_size)
        }
    );
    
    // Setup output ports
    output_ports_.resize(1);
    
    // Output port: output_signal
    output_ports_[0].name = "output_signal";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.batch_size),
            static_cast<std::size_t>(params_.num_antennas),
            static_cast<std::size_t>(params_.fft_size)
        }
    );
}

std::vector<std::string> FFTProcessor::get_input_port_names() const {
    return {"input_signal"};
}

std::vector<std::string> FFTProcessor::get_output_port_names() const {
    return {"output_signal"};
}

std::vector<framework::tensor::TensorInfo>
FFTProcessor::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "input_signal") {
        return {input_ports_[0].tensors[0].tensor_info};
    }
    throw std::invalid_argument("Unknown input port: " + std::string(port_name));
}

std::vector<framework::tensor::TensorInfo>
FFTProcessor::get_output_tensor_info(std::string_view port_name) const {
    if (port_name == "output_signal") {
        return {output_ports_[0].tensors[0].tensor_info};
    }
    throw std::invalid_argument("Unknown output port: " + std::string(port_name));
}

void FFTProcessor::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    // Memory is already allocated in constructor
}

void FFTProcessor::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for FFT processing
}

void FFTProcessor::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t /*stream*/
) {
    // Update descriptor with current tensor pointers
    h_descriptor_.input_data = current_input_;
    h_descriptor_.output_data = current_output_;
    
    // Copy descriptor to GPU
    cudaError_t err = cudaMemcpy(d_descriptor_, &h_descriptor_, sizeof(FFTDescriptor), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy FFT descriptor to GPU");
    }
}

void FFTProcessor::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    if (inputs.size() != 1) {
        throw std::runtime_error("FFT processor requires exactly 1 input port");
    }
    
    // Extract device pointers from input ports
    for (const auto& port : inputs) {
        if (port.name == "input_signal" && !port.tensors.empty()) {
            current_input_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        }
    }
}

std::vector<framework::pipeline::PortInfo> FFTProcessor::get_outputs() const {
    // Return a copy of output_ports_ with updated device pointers
    std::vector<framework::pipeline::PortInfo> outputs = output_ports_;
    
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_output_data_;
    }
    
    return outputs;
}

void FFTProcessor::execute(cudaStream_t stream) {
    // Launch preprocessing if needed
    if (params_.enable_windowing || !current_input_) {
        cudaError_t err = launch_preprocessing_kernel(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("FFT preprocessing kernel launch failed");
        }
    }
    
    // Execute CUFFT
    cufftResult result;
    if (params_.direction == FFTDirection::FORWARD) {
        result = cufftExecC2C(fft_plan_, 
                             const_cast<cuComplex*>(current_input_ ? current_input_ : d_output_data_), 
                             d_output_data_, 
                             CUFFT_FORWARD);
    } else {
        result = cufftExecC2C(fft_plan_, 
                             const_cast<cuComplex*>(current_input_ ? current_input_ : d_output_data_), 
                             d_output_data_, 
                             CUFFT_INVERSE);
    }
    
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT execution failed");
    }
    
    // Launch postprocessing if needed
    if (params_.direction == FFTDirection::INVERSE && params_.normalize) {
        cudaError_t err = launch_postprocessing_kernel(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("FFT postprocessing kernel launch failed");
        }
    }
}

void FFTProcessor::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(FFTDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for descriptor");
    }
    
    // Allocate memory for input/output data
    size_t data_size = h_descriptor_.total_samples * sizeof(cuComplex);
    
    err = cudaMalloc(&d_input_data_, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for input data");
    }
    
    err = cudaMalloc(&d_output_data_, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for output data");
    }
    
    // Set current output pointer for get_outputs()
    current_output_ = d_output_data_;
}

void FFTProcessor::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
    if (d_input_data_) {
        cudaFree(d_input_data_);
        d_input_data_ = nullptr;
    }
    if (d_output_data_) {
        cudaFree(d_output_data_);
        d_output_data_ = nullptr;
    }
    if (d_window_function_) {
        cudaFree(d_window_function_);
        d_window_function_ = nullptr;
    }
}

void FFTProcessor::create_fft_plan() {
    cufftResult result = cufftPlan1d(&fft_plan_, params_.fft_size, CUFFT_C2C, 
                                     params_.num_antennas * params_.batch_size);
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("Failed to create CUFFT plan");
    }
}

void FFTProcessor::destroy_fft_plan() {
    if (fft_plan_) {
        cufftDestroy(fft_plan_);
        fft_plan_ = 0;
    }
}

void FFTProcessor::setup_windowing() {
    if (!params_.enable_windowing) return;
    
    // Allocate memory for window function
    cudaError_t err = cudaMalloc(&d_window_function_, params_.fft_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for window function");
    }
    
    // Generate Hann window on host
    std::vector<float> hann_window(params_.fft_size);
    for (int i = 0; i < params_.fft_size; ++i) {
        hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (params_.fft_size - 1)));
    }
    
    // Copy window to GPU
    err = cudaMemcpy(d_window_function_, hann_window.data(), 
                     params_.fft_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy window function to GPU");
    }
    
    h_descriptor_.window_function = d_window_function_;
}

framework::pipeline::ModuleMemoryRequirements FFTProcessor::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    // Calculate memory requirements based on parameters
    size_t total_bytes = 0;
    
    // Memory for input/output data
    total_bytes += 2 * h_descriptor_.total_samples * sizeof(cuComplex);
    
    // Memory for window function if enabled
    if (params_.enable_windowing) {
        total_bytes += params_.fft_size * sizeof(float);
    }
    
    // Memory for descriptor
    total_bytes += sizeof(FFTDescriptor);
    
    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = 256; // CUDA memory alignment
    
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics
FFTProcessor::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    
    if (port_name == "output_signal") {
        chars.provides_fixed_address_for_zero_copy = true;
    }
    
    return chars;
}

cudaError_t FFTProcessor::launch_preprocessing_kernel(cudaStream_t stream) {
    int total_samples = h_descriptor_.total_samples;
    
    dim3 blockSize(256);
    dim3 gridSize((total_samples + blockSize.x - 1) / blockSize.x);
    
    if (params_.enable_windowing) {
        apply_windowing_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    } else {
        copy_input_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    }
    
    return cudaGetLastError();
}

cudaError_t FFTProcessor::launch_postprocessing_kernel(cudaStream_t stream) {
    int total_samples = h_descriptor_.total_samples;
    
    dim3 blockSize(256);
    dim3 gridSize((total_samples + blockSize.x - 1) / blockSize.x);
    
    normalize_ifft_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    
    return cudaGetLastError();
}

} // namespace fft_processing