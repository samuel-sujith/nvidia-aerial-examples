#include "mimo_detection_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdlib>
#include <cublas_v2.h>

namespace mimo_detection {

namespace {
bool debug_enabled() {
    const char* value = std::getenv("AERIAL_DEBUG");
    return value && value[0] != '0';
}
} // namespace

// CUDA kernels for MIMO detection algorithms
__global__ void zero_forcing_detection_kernel(const MIMODescriptor* desc) {
    if (!desc || !desc->params || !desc->received_symbols || !desc->channel_matrix || !desc->detected_symbols) {
        return;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = desc->params->num_subcarriers * desc->params->num_ofdm_symbols;
    
    if (idx < total_elements) {
        int subcarrier = idx % desc->params->num_subcarriers;
        int symbol = idx / desc->params->num_subcarriers;
        
        // For each subcarrier, solve H * x = y using pseudo-inverse
        // This is a simplified ZF for demonstration - real implementation would use matrix inversion
        for (int tx = 0; tx < desc->params->num_tx_antennas; ++tx) {
            cuComplex sum = make_cuComplex(0.0f, 0.0f);
            float norm = 0.0f;
            
            // Simplified ZF: use matched filter as approximation
            for (int rx = 0; rx < desc->params->num_rx_antennas; ++rx) {
                int h_idx = (rx * desc->params->num_tx_antennas + tx) * desc->params->num_subcarriers + subcarrier;
                int y_idx = (rx * desc->params->num_subcarriers + subcarrier) * desc->params->num_ofdm_symbols + symbol;
                
                cuComplex h_conj = cuConjf(desc->channel_matrix[h_idx]);
                sum = cuCaddf(sum, cuCmulf(h_conj, desc->received_symbols[y_idx]));
                norm += cuCrealf(cuCmulf(h_conj, desc->channel_matrix[h_idx]));
            }
            
            if (norm > 1e-6f) {
                sum = make_cuComplex(cuCrealf(sum) / norm, cuCimagf(sum) / norm);
            }
            
            int out_idx = (tx * desc->params->num_subcarriers + subcarrier) * desc->params->num_ofdm_symbols + symbol;
            desc->detected_symbols[out_idx] = sum;
        }
    }
}

__global__ void mmse_detection_kernel(const MIMODescriptor* desc) {
    if (!desc || !desc->params || !desc->received_symbols || !desc->channel_matrix || !desc->detected_symbols) {
        return;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = desc->params->num_subcarriers * desc->params->num_ofdm_symbols;
    
    if (idx < total_elements) {
        int subcarrier = idx % desc->params->num_subcarriers;
        int symbol = idx / desc->params->num_subcarriers;
        
        // MMSE detection: (H^H * H + sigma^2 * I)^-1 * H^H * y
        // Simplified implementation for demonstration
        for (int tx = 0; tx < desc->params->num_tx_antennas; ++tx) {
            cuComplex sum = make_cuComplex(0.0f, 0.0f);
            float norm = desc->params->noise_variance; // Add noise regularization
            
            for (int rx = 0; rx < desc->params->num_rx_antennas; ++rx) {
                int h_idx = (rx * desc->params->num_tx_antennas + tx) * desc->params->num_subcarriers + subcarrier;
                int y_idx = (rx * desc->params->num_subcarriers + subcarrier) * desc->params->num_ofdm_symbols + symbol;
                
                cuComplex h_conj = cuConjf(desc->channel_matrix[h_idx]);
                sum = cuCaddf(sum, cuCmulf(h_conj, desc->received_symbols[y_idx]));
                norm += cuCrealf(cuCmulf(h_conj, desc->channel_matrix[h_idx]));
            }
            
            if (norm > 1e-6f) {
                sum = make_cuComplex(cuCrealf(sum) / norm, cuCimagf(sum) / norm);
            }
            
            int out_idx = (tx * desc->params->num_subcarriers + subcarrier) * desc->params->num_ofdm_symbols + symbol;
            desc->detected_symbols[out_idx] = sum;
        }
    }
}

__global__ void qpsk_hard_decision_kernel(const MIMODescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < desc->total_resource_elements) {
        cuComplex sym = desc->detected_symbols[idx];
        float real_part = cuCrealf(sym) > 0.0f ? 1.0f : -1.0f;
        float imag_part = cuCimagf(sym) > 0.0f ? 1.0f : -1.0f;
        
        desc->detected_symbols[idx] = make_cuComplex(real_part, imag_part);
    }
}

MIMODetector::MIMODetector(const std::string& module_id, const MIMOParams& params)
    : module_id_(module_id), params_(params), d_descriptor_(nullptr) {
    
    // Validate parameters
    if (params_.num_tx_antennas <= 0 || params_.num_rx_antennas <= 0) {
        throw std::invalid_argument("Number of antennas must be positive");
    }
    
    if (params_.num_tx_antennas > params_.num_rx_antennas) {
        throw std::invalid_argument("Number of TX antennas cannot exceed RX antennas");
    }
    
    if (params_.num_subcarriers <= 0 || params_.num_ofdm_symbols <= 0) {
        throw std::invalid_argument("Number of subcarriers and OFDM symbols must be positive");
    }
    
    // Initialize descriptor
    h_descriptor_ = {};
    h_descriptor_.params = &params_;
    h_descriptor_.total_resource_elements = calculate_total_elements();
    
    setup_port_info();
}

MIMODetector::~MIMODetector() {
    deallocate_gpu_memory();
}

void MIMODetector::setup_port_info() {
    // Setup input ports
    input_ports_.resize(2);
    
    // Input port 0: received_symbols
    input_ports_[0].name = "received_symbols";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_rx_antennas),
            static_cast<std::size_t>(params_.num_subcarriers),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
    
    // Input port 1: channel_matrix
    input_ports_[1].name = "channel_matrix";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_rx_antennas),
            static_cast<std::size_t>(params_.num_tx_antennas),
            static_cast<std::size_t>(params_.num_subcarriers)
        }
    );
    
    // Setup output ports
    output_ports_.resize(1);
    
    // Output port 0: detected_symbols
    output_ports_[0].name = "detected_symbols";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_tx_antennas),
            static_cast<std::size_t>(params_.num_subcarriers),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
}

std::vector<std::string> MIMODetector::get_input_port_names() const {
    return {"received_symbols", "channel_matrix"};
}

std::vector<std::string> MIMODetector::get_output_port_names() const {
    return {"detected_symbols"};
}

std::vector<framework::tensor::TensorInfo>
MIMODetector::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "received_symbols") {
        return {input_ports_[0].tensors[0].tensor_info};
    } else if (port_name == "channel_matrix") {
        return {input_ports_[1].tensors[0].tensor_info};
    }
    throw std::invalid_argument("Unknown input port: " + std::string(port_name));
}

std::vector<framework::tensor::TensorInfo>
MIMODetector::get_output_tensor_info(std::string_view port_name) const {
    if (port_name == "detected_symbols") {
        return {output_ports_[0].tensors[0].tensor_info};
    }
    throw std::invalid_argument("Unknown output port: " + std::string(port_name));
}

void MIMODetector::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    if (!mem_slice_.device_tensor_ptr) {
        throw std::runtime_error("MIMO detected symbols buffer not allocated");
    }
    std::byte* base = mem_slice_.device_tensor_ptr;
    size_t offset = 0;

    d_detected_symbols_ = reinterpret_cast<cuComplex*>(base + offset);
    offset += h_descriptor_.total_resource_elements * sizeof(cuComplex);

    if (params_.soft_output) {
        d_soft_bits_ = reinterpret_cast<float*>(base + offset);
        offset += h_descriptor_.total_resource_elements * sizeof(float);
    }

    allocate_gpu_memory();
    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<MIMODescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<MIMODescriptor>(0);
    if (!dynamic_params_gpu_ptr_) {
        throw std::runtime_error("MIMO dynamic descriptor device pointer not allocated");
    }
    d_descriptor_ = dynamic_params_gpu_ptr_;

    int total_elements = params_.num_subcarriers * params_.num_ofdm_symbols;
    dim3 blockSize(256);
    dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);
    if (params_.algorithm == MIMODetectionAlgorithm::ZERO_FORCING) {
        detect_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(zero_forcing_detection_kernel));
    } else {
        detect_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(mmse_detection_kernel));
    }
    detect_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(detect_kernel_config_, *dynamic_params_gpu_ptr_);

    if (params_.constellation_size == 4) {
        int total_symbols = params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols;
        dim3 gridSizeHard((total_symbols + blockSize.x - 1) / blockSize.x);
        hard_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(qpsk_hard_decision_kernel));
        hard_kernel_config_.setup_kernel_dimensions(gridSizeHard, blockSize);
        framework::pipeline::setup_kernel_arguments(hard_kernel_config_, *dynamic_params_gpu_ptr_);
    }
}

void MIMODetector::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for MIMO detection
}

void MIMODetector::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {
    if (!dynamic_params_cpu_ptr_) {
        throw std::runtime_error("MIMO dynamic descriptor not initialized");
    }
    if (!kernel_desc_mgr_) {
        throw std::runtime_error("MIMO kernel descriptor manager not initialized");
    }
    if (!d_received_symbols_ || !d_channel_matrix_ || !d_detected_symbols_ || !d_params_) {
        throw std::runtime_error("MIMO device buffers not initialized");
    }
    dynamic_params_cpu_ptr_->received_symbols = d_received_symbols_;
    dynamic_params_cpu_ptr_->channel_matrix = d_channel_matrix_;
    dynamic_params_cpu_ptr_->detected_symbols = d_detected_symbols_;
    dynamic_params_cpu_ptr_->soft_bits = d_soft_bits_;
    dynamic_params_cpu_ptr_->params = d_params_;
    dynamic_params_cpu_ptr_->total_resource_elements = calculate_total_elements();
    kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);
    if (debug_enabled()) {
        std::cerr << "[DEBUG] MIMODetector::configure_io rx=" << d_received_symbols_
                  << " h=" << d_channel_matrix_
                  << " out=" << d_detected_symbols_
                  << " params=" << d_params_
                  << " dyn_cpu=" << dynamic_params_cpu_ptr_
                  << " dyn_gpu=" << dynamic_params_gpu_ptr_
                  << std::endl;
    }
}

void MIMODetector::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    if (inputs.size() != 2) {
        throw std::runtime_error("MIMO detector requires exactly 2 input ports");
    }
    
    // Extract device pointers from input ports
    for (const auto& port : inputs) {
        if (port.name == "received_symbols" && !port.tensors.empty()) {
            current_received_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "channel_matrix" && !port.tensors.empty()) {
            current_channel_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        }
    }
}

std::vector<framework::pipeline::PortInfo> MIMODetector::get_outputs() const {
    // Return a copy of output_ports_ with updated device pointers
    std::vector<framework::pipeline::PortInfo> outputs = output_ports_;
    
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_detected_symbols_;
    }
    
    return outputs;
}

void MIMODetector::execute(cudaStream_t stream) {
    if (!current_received_ || !current_channel_) {
        throw std::runtime_error("Input tensors not set - call set_inputs first");
    }
    if (!d_received_symbols_ || !d_channel_matrix_ || !d_detected_symbols_) {
        throw std::runtime_error("MIMO device buffers not initialized");
    }
    
    // Copy input data to internal buffers for processing
    size_t received_size = params_.num_rx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    size_t channel_size = params_.num_rx_antennas * params_.num_tx_antennas * params_.num_subcarriers * sizeof(cuComplex);
    
    cudaError_t err = cudaMemcpy(d_received_symbols_, current_received_, received_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy received symbols to internal buffer");
    }
    
    err = cudaMemcpy(d_channel_matrix_, current_channel_, channel_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy channel matrix to internal buffer");
    }
    
    // Launch MIMO detection kernel
    err = launch_mimo_detection_kernel(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("MIMO detection kernel launch failed");
    }

    if (debug_enabled()) {
        err = cudaPeekAtLastError();
        std::cerr << "[DEBUG] MIMODetector::execute cudaPeekAtLastError="
                  << cudaGetErrorString(err) << std::endl;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Stream sync failed: ") + cudaGetErrorString(err));
        }
    }
}

void MIMODetector::allocate_gpu_memory() {
    cudaError_t err;
    err = cudaMalloc(&d_params_, sizeof(MIMOParams));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for params");
    }
    err = cudaMemcpy(d_params_, &params_, sizeof(MIMOParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy params to device");
    }
    
    // Calculate memory sizes
    size_t received_size = params_.num_rx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    size_t channel_size = params_.num_rx_antennas * params_.num_tx_antennas * params_.num_subcarriers * sizeof(cuComplex);
    size_t detected_size = params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    size_t temp_matrix_size = params_.num_tx_antennas * params_.num_tx_antennas * params_.num_subcarriers * sizeof(cuComplex);
    
    // Allocate memory for received symbols
    err = cudaMalloc(&d_received_symbols_, received_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for received symbols");
    }
    
    // Allocate memory for channel matrix
    err = cudaMalloc(&d_channel_matrix_, channel_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for channel matrix");
    }
    
    (void)detected_size;
    
    // Allocate memory for temporary matrix operations
    err = cudaMalloc(&d_temp_matrix_, temp_matrix_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for temporary matrix");
    }
    
    // Allocate memory for soft bits if needed
    if (params_.soft_output) {
        size_t soft_bits_size = params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * 
                               (params_.constellation_size / 4) * sizeof(float); // bits per symbol
        (void)soft_bits_size;
    }
}

void MIMODetector::deallocate_gpu_memory() {
    if (d_params_) {
        cudaFree(d_params_);
        d_params_ = nullptr;
    }
    d_descriptor_ = nullptr;
    if (d_received_symbols_) {
        cudaFree(d_received_symbols_);
        d_received_symbols_ = nullptr;
    }
    if (d_channel_matrix_) {
        cudaFree(d_channel_matrix_);
        d_channel_matrix_ = nullptr;
    }
    d_detected_symbols_ = nullptr;
    if (d_temp_matrix_) {
        cudaFree(d_temp_matrix_);
        d_temp_matrix_ = nullptr;
    }
    d_soft_bits_ = nullptr;
}

framework::pipeline::ModuleMemoryRequirements MIMODetector::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    size_t total_bytes = 0;
    total_bytes += params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    if (params_.soft_output) {
        total_bytes += params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols *
                      (params_.constellation_size / 4) * sizeof(float);
    }
    reqs.device_tensor_bytes = total_bytes;
    reqs.dynamic_kernel_descriptor_bytes = sizeof(MIMODescriptor);
    reqs.alignment = 256; // CUDA memory alignment
    
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics
MIMODetector::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    
    if (port_name == "detected_symbols") {
        chars.provides_fixed_address_for_zero_copy = true;
    }
    
    return chars;
}

cudaError_t MIMODetector::launch_mimo_detection_kernel(cudaStream_t stream) {
    // Launch detection kernel based on algorithm
    const CUresult detect_err = detect_kernel_config_.launch(stream);
    
    cudaError_t err = (detect_err == CUDA_SUCCESS) ? cudaGetLastError() : cudaErrorLaunchFailure;
    if (err != cudaSuccess) {
        return err;
    }
    
    // Apply hard decision if constellation is QPSK
    if (params_.constellation_size == 4) {
        const CUresult hard_err = hard_kernel_config_.launch(stream);
        if (hard_err != CUDA_SUCCESS) {
            return cudaErrorLaunchFailure;
        }
    }
    
    return cudaGetLastError();
}

std::span<const CUgraphNode> MIMODetector::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    detect_node_ = graph->add_kernel_node(deps, detect_kernel_config_.get_kernel_params());
    if (params_.constellation_size == 4) {
        hard_node_ = graph->add_kernel_node({&detect_node_, 1}, hard_kernel_config_.get_kernel_params());
        return {&hard_node_, 1};
    }
    return {&detect_node_, 1};
}

void MIMODetector::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams& /*params*/) {
    if (!detect_node_) {
        throw std::runtime_error("MIMO detect graph node not initialized");
    }
    auto detect_params = detect_kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, detect_node_, &detect_params);
    if (params_.constellation_size == 4) {
        if (!hard_node_) {
            throw std::runtime_error("MIMO hard-decision graph node not initialized");
        }
        auto hard_params = hard_kernel_config_.get_kernel_params();
        cuGraphExecKernelNodeSetParams(exec, hard_node_, &hard_params);
    }
}

size_t MIMODetector::calculate_total_elements() const {
    return params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols;
}

} // namespace mimo_detection