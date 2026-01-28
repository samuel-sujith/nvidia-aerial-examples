#include "fft_processing_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>

using namespace framework::tensor;

namespace fft_processing {

__global__ void apply_windowing_kernel(const FFTDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < desc->total_samples) {
        int sample_idx = idx % desc->params->fft_size;
        float window_val = desc->window_function ? desc->window_function[sample_idx] : 1.0f;

        cuComplex input_val = desc->input_data[idx];
        cuComplex windowed_val = make_cuComplex(
            cuCrealf(input_val) * window_val,
            cuCimagf(input_val) * window_val
        );

        desc->output_data[idx] = windowed_val;
    }
}

__global__ void normalize_ifft_kernel(const FFTDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < desc->total_samples) {
        float norm_factor = 1.0f / desc->params->fft_size;
        cuComplex output_val = desc->output_data[idx];

        desc->output_data[idx] = make_cuComplex(
            cuCrealf(output_val) * norm_factor,
            cuCimagf(output_val) * norm_factor
        );
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
    h_descriptor_.total_samples = params_.fft_size * params_.num_antennas * params_.batch_size;
    
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

void FFTProcessor::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    d_output_data_ = reinterpret_cast<cuComplex*>(mem_slice_.device_tensor_ptr);
    allocate_gpu_memory();
    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<FFTDescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<FFTDescriptor>(0);
    d_descriptor_ = dynamic_params_gpu_ptr_;

    int total_samples = h_descriptor_.total_samples;
    dim3 blockSize(256);
    dim3 gridSize((total_samples + blockSize.x - 1) / blockSize.x);
    window_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(apply_windowing_kernel));
    window_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(window_kernel_config_, *dynamic_params_gpu_ptr_);

    norm_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(normalize_ifft_kernel));
    norm_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(norm_kernel_config_, *dynamic_params_gpu_ptr_);
}

void FFTProcessor::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for FFT processing
}

void FFTProcessor::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {
    if (dynamic_params_cpu_ptr_) {
        dynamic_params_cpu_ptr_->input_data = d_input_data_;
        dynamic_params_cpu_ptr_->output_data = d_output_data_;
        dynamic_params_cpu_ptr_->window_function = d_window_function_;
        dynamic_params_cpu_ptr_->params = d_params_;
        dynamic_params_cpu_ptr_->total_samples = h_descriptor_.total_samples;
        kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);
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
        // Always use the module's internal buffer
        outputs[0].tensors[0].device_ptr = d_output_data_;
    }
    
    return outputs;
}

void FFTProcessor::execute(cudaStream_t stream) {
    if (!current_input_) {
        throw std::runtime_error("Input not set - call set_inputs first");
    }
    
    // Copy input data to internal buffer for processing
    cudaError_t err = cudaMemcpy(d_input_data_, current_input_, 
                               h_descriptor_.total_samples * sizeof(cuComplex), 
                               cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy input data to internal buffer");
    }
    
    cuComplex* fft_input = d_input_data_;
    cuComplex* fft_output = d_output_data_;
    
    if (params_.enable_windowing) {
        // Apply windowing using simplified kernel
        const CUresult window_err = window_kernel_config_.launch(stream);
        if (window_err != CUDA_SUCCESS) {
            throw std::runtime_error("FFT windowing kernel launch failed");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("FFT windowing kernel launch failed");
        }
    }
    
    // Execute CUFFT
    cufftResult result;
    if (params_.direction == FFTDirection::FORWARD) {
        result = cufftExecC2C(fft_plan_, fft_input, fft_output, CUFFT_FORWARD);
    } else {
        result = cufftExecC2C(fft_plan_, fft_input, fft_output, CUFFT_INVERSE);
    }
    
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error("CUFFT execution failed");
    }
    
    // Apply normalization for inverse FFT if needed
    if (params_.direction == FFTDirection::INVERSE && params_.normalize) {
        const CUresult norm_err = norm_kernel_config_.launch(stream);
        if (norm_err != CUDA_SUCCESS) {
            throw std::runtime_error("FFT normalization kernel launch failed");
        }
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("FFT normalization kernel launch failed");
        }
    }
}

std::span<const CUgraphNode> FFTProcessor::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    std::span<const CUgraphNode> last = deps;
    if (params_.enable_windowing) {
        window_node_ = graph->add_kernel_node(last, window_kernel_config_.get_kernel_params());
        last = {&window_node_, 1};
    }
    if (params_.direction == FFTDirection::INVERSE && params_.normalize) {
        norm_node_ = graph->add_kernel_node(last, norm_kernel_config_.get_kernel_params());
        return {&norm_node_, 1};
    }
    return last;
}

void FFTProcessor::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams& /*params*/) {
    if (params_.enable_windowing) {
        auto window_params = window_kernel_config_.get_kernel_params();
        cuGraphExecKernelNodeSetParams(exec, window_node_, &window_params);
    }
    if (params_.direction == FFTDirection::INVERSE && params_.normalize) {
        auto norm_params = norm_kernel_config_.get_kernel_params();
        cuGraphExecKernelNodeSetParams(exec, norm_node_, &norm_params);
    }
}

void FFTProcessor::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err;
    err = cudaMalloc(&d_params_, sizeof(FFTParams));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for params");
    }
    err = cudaMemcpy(d_params_, &params_, sizeof(FFTParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy params to device");
    }
    
    // Allocate memory for input/output data
    size_t data_size = h_descriptor_.total_samples * sizeof(cuComplex);
    
    err = cudaMalloc(&d_input_data_, data_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for input data");
    }
    
    (void)data_size;
}

void FFTProcessor::deallocate_gpu_memory() {
    if (d_params_) {
        cudaFree(d_params_);
        d_params_ = nullptr;
    }
    d_descriptor_ = nullptr;
    if (d_input_data_) {
        cudaFree(d_input_data_);
        d_input_data_ = nullptr;
    }
    d_output_data_ = nullptr;
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
}

framework::pipeline::ModuleMemoryRequirements FFTProcessor::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    size_t output_bytes = h_descriptor_.total_samples * sizeof(cuComplex);
    reqs.device_tensor_bytes = output_bytes;
    reqs.dynamic_kernel_descriptor_bytes = sizeof(FFTDescriptor);
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

} // namespace fft_processing