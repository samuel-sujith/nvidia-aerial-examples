#include "mimo_detection_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>

namespace mimo_detection {

// CUDA kernels for MIMO detection algorithms
__global__ void zero_forcing_detection_kernel(
    const cuComplex* received_symbols,
    const cuComplex* channel_matrix,
    cuComplex* detected_symbols,
    int num_tx, int num_rx, int num_subcarriers, int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_subcarriers * num_symbols;
    
    if (idx < total_elements) {
        int subcarrier = idx % num_subcarriers;
        int symbol = idx / num_subcarriers;
        
        // For each subcarrier, solve H * x = y using pseudo-inverse
        // This is a simplified ZF for demonstration - real implementation would use matrix inversion
        for (int tx = 0; tx < num_tx; ++tx) {
            cuComplex sum = make_cuComplex(0.0f, 0.0f);
            float norm = 0.0f;
            
            // Simplified ZF: use matched filter as approximation
            for (int rx = 0; rx < num_rx; ++rx) {
                int h_idx = (rx * num_tx + tx) * num_subcarriers + subcarrier;
                int y_idx = (rx * num_subcarriers + subcarrier) * num_symbols + symbol;
                
                cuComplex h_conj = cuConjf(channel_matrix[h_idx]);
                sum = cuCaddf(sum, cuCmulf(h_conj, received_symbols[y_idx]));
                norm += cuCrealf(cuCmulf(h_conj, channel_matrix[h_idx]));
            }
            
            if (norm > 1e-6f) {
                sum = make_cuComplex(cuCrealf(sum) / norm, cuCimagf(sum) / norm);
            }
            
            int out_idx = (tx * num_subcarriers + subcarrier) * num_symbols + symbol;
            detected_symbols[out_idx] = sum;
        }
    }
}

__global__ void mmse_detection_kernel(
    const cuComplex* received_symbols,
    const cuComplex* channel_matrix,
    cuComplex* detected_symbols,
    float noise_variance,
    int num_tx, int num_rx, int num_subcarriers, int num_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_subcarriers * num_symbols;
    
    if (idx < total_elements) {
        int subcarrier = idx % num_subcarriers;
        int symbol = idx / num_subcarriers;
        
        // MMSE detection: (H^H * H + sigma^2 * I)^-1 * H^H * y
        // Simplified implementation for demonstration
        for (int tx = 0; tx < num_tx; ++tx) {
            cuComplex sum = make_cuComplex(0.0f, 0.0f);
            float norm = noise_variance; // Add noise regularization
            
            for (int rx = 0; rx < num_rx; ++rx) {
                int h_idx = (rx * num_tx + tx) * num_subcarriers + subcarrier;
                int y_idx = (rx * num_subcarriers + subcarrier) * num_symbols + symbol;
                
                cuComplex h_conj = cuConjf(channel_matrix[h_idx]);
                sum = cuCaddf(sum, cuCmulf(h_conj, received_symbols[y_idx]));
                norm += cuCrealf(cuCmulf(h_conj, channel_matrix[h_idx]));
            }
            
            if (norm > 1e-6f) {
                sum = make_cuComplex(cuCrealf(sum) / norm, cuCimagf(sum) / norm);
            }
            
            int out_idx = (tx * num_subcarriers + subcarrier) * num_symbols + symbol;
            detected_symbols[out_idx] = sum;
        }
    }
}

__global__ void qpsk_hard_decision_kernel(
    cuComplex* symbols, 
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_symbols) {
        cuComplex sym = symbols[idx];
        float real_part = cuCrealf(sym) > 0.0f ? 1.0f : -1.0f;
        float imag_part = cuCimagf(sym) > 0.0f ? 1.0f : -1.0f;
        
        symbols[idx] = make_cuComplex(real_part, imag_part);
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
    
    allocate_gpu_memory();
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

void MIMODetector::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    // Memory is already allocated in constructor
}

void MIMODetector::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for MIMO detection
}

void MIMODetector::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t /*stream*/
) {
    // Update descriptor with current tensor pointers
    h_descriptor_.received_symbols = current_received_;
    h_descriptor_.channel_matrix = current_channel_;
    h_descriptor_.detected_symbols = d_detected_symbols_;
    h_descriptor_.soft_bits = d_soft_bits_;
    
    // Copy descriptor to GPU
    cudaError_t err = cudaMemcpy(d_descriptor_, &h_descriptor_, sizeof(MIMODescriptor), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy MIMO descriptor to GPU");
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
}

void MIMODetector::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(MIMODescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for descriptor");
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
    
    // Allocate memory for detected symbols
    err = cudaMalloc(&d_detected_symbols_, detected_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for detected symbols");
    }
    
    // Allocate memory for temporary matrix operations
    err = cudaMalloc(&d_temp_matrix_, temp_matrix_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for temporary matrix");
    }
    
    // Allocate memory for soft bits if needed
    if (params_.soft_output) {
        size_t soft_bits_size = params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * 
                               (params_.constellation_size / 4) * sizeof(float); // bits per symbol
        err = cudaMalloc(&d_soft_bits_, soft_bits_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for soft bits");
        }
    }
}

void MIMODetector::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
    if (d_received_symbols_) {
        cudaFree(d_received_symbols_);
        d_received_symbols_ = nullptr;
    }
    if (d_channel_matrix_) {
        cudaFree(d_channel_matrix_);
        d_channel_matrix_ = nullptr;
    }
    if (d_detected_symbols_) {
        cudaFree(d_detected_symbols_);
        d_detected_symbols_ = nullptr;
    }
    if (d_temp_matrix_) {
        cudaFree(d_temp_matrix_);
        d_temp_matrix_ = nullptr;
    }
    if (d_soft_bits_) {
        cudaFree(d_soft_bits_);
        d_soft_bits_ = nullptr;
    }
}

framework::pipeline::ModuleMemoryRequirements MIMODetector::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    // Calculate memory requirements based on parameters
    size_t total_bytes = 0;
    
    // Memory for received symbols
    total_bytes += params_.num_rx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    
    // Memory for channel matrix
    total_bytes += params_.num_rx_antennas * params_.num_tx_antennas * params_.num_subcarriers * sizeof(cuComplex);
    
    // Memory for detected symbols
    total_bytes += params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * sizeof(cuComplex);
    
    // Memory for temporary matrix operations
    total_bytes += params_.num_tx_antennas * params_.num_tx_antennas * params_.num_subcarriers * sizeof(cuComplex);
    
    // Memory for soft bits if enabled
    if (params_.soft_output) {
        total_bytes += params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols * 
                      (params_.constellation_size / 4) * sizeof(float);
    }
    
    // Memory for descriptor
    total_bytes += sizeof(MIMODescriptor);
    
    reqs.device_tensor_bytes = total_bytes;
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
    int total_elements = params_.num_subcarriers * params_.num_ofdm_symbols;
    
    dim3 blockSize(256);
    dim3 gridSize((total_elements + blockSize.x - 1) / blockSize.x);
    
    // Launch detection kernel based on algorithm
    if (params_.algorithm == MIMODetectionAlgorithm::ZERO_FORCING) {
        zero_forcing_detection_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_received_symbols_,
            d_channel_matrix_,
            d_detected_symbols_,
            params_.num_tx_antennas,
            params_.num_rx_antennas,
            params_.num_subcarriers,
            params_.num_ofdm_symbols
        );
    } else if (params_.algorithm == MIMODetectionAlgorithm::MMSE) {
        mmse_detection_kernel<<<gridSize, blockSize, 0, stream>>>(
            d_received_symbols_,
            d_channel_matrix_,
            d_detected_symbols_,
            params_.noise_variance,
            params_.num_tx_antennas,
            params_.num_rx_antennas,
            params_.num_subcarriers,
            params_.num_ofdm_symbols
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Apply hard decision if constellation is QPSK
    if (params_.constellation_size == 4) {
        int total_symbols = params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols;
        dim3 gridSizeHard((total_symbols + blockSize.x - 1) / blockSize.x);
        
        qpsk_hard_decision_kernel<<<gridSizeHard, blockSize, 0, stream>>>(
            d_detected_symbols_,
            total_symbols
        );
    }
    
    return cudaGetLastError();
}

size_t MIMODetector::calculate_total_elements() const {
    return params_.num_tx_antennas * params_.num_subcarriers * params_.num_ofdm_symbols;
}

} // namespace mimo_detection