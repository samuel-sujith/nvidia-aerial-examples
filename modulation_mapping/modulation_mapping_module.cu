#include "modulation_mapping_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace modulation_mapping {

// CUDA kernels for modulation mapping

__global__ void qpsk_modulation_kernel(
    const uint8_t* input_bits,
    cuComplex* output_symbols,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_symbols) {
        // QPSK constellation: (1+1j, 1-1j, -1+1j, -1-1j) / sqrt(2)
        int bit_idx = idx * 2;
        uint8_t bit0 = input_bits[bit_idx];
        uint8_t bit1 = input_bits[bit_idx + 1];
        
        float real_part = (bit0 == 0) ? 1.0f : -1.0f;
        float imag_part = (bit1 == 0) ? 1.0f : -1.0f;
        
        // Normalize for unit energy
        float norm_factor = 1.0f / sqrtf(2.0f);
        output_symbols[idx] = make_cuComplex(real_part * norm_factor, imag_part * norm_factor);
    }
}

__global__ void qam16_modulation_kernel(
    const uint8_t* input_bits,
    cuComplex* output_symbols,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_symbols) {
        // 16-QAM constellation points
        int bit_idx = idx * 4;
        
        // Gray mapping for 16-QAM
        uint8_t bits = (input_bits[bit_idx] << 3) | (input_bits[bit_idx+1] << 2) |
                       (input_bits[bit_idx+2] << 1) | input_bits[bit_idx+3];
        
        // Map 4 bits to I/Q coordinates
        float i_coord, q_coord;
        switch (bits) {
            case 0:  i_coord = -3.0f; q_coord = -3.0f; break; // 0000
            case 1:  i_coord = -3.0f; q_coord = -1.0f; break; // 0001
            case 2:  i_coord = -3.0f; q_coord =  3.0f; break; // 0010
            case 3:  i_coord = -3.0f; q_coord =  1.0f; break; // 0011
            case 4:  i_coord = -1.0f; q_coord = -3.0f; break; // 0100
            case 5:  i_coord = -1.0f; q_coord = -1.0f; break; // 0101
            case 6:  i_coord = -1.0f; q_coord =  3.0f; break; // 0110
            case 7:  i_coord = -1.0f; q_coord =  1.0f; break; // 0111
            case 8:  i_coord =  3.0f; q_coord = -3.0f; break; // 1000
            case 9:  i_coord =  3.0f; q_coord = -1.0f; break; // 1001
            case 10: i_coord =  3.0f; q_coord =  3.0f; break; // 1010
            case 11: i_coord =  3.0f; q_coord =  1.0f; break; // 1011
            case 12: i_coord =  1.0f; q_coord = -3.0f; break; // 1100
            case 13: i_coord =  1.0f; q_coord = -1.0f; break; // 1101
            case 14: i_coord =  1.0f; q_coord =  3.0f; break; // 1110
            case 15: i_coord =  1.0f; q_coord =  1.0f; break; // 1111
        }
        
        // Normalize for unit average energy
        float norm_factor = 1.0f / sqrtf(10.0f);
        output_symbols[idx] = make_cuComplex(i_coord * norm_factor, q_coord * norm_factor);
    }
}

__global__ void qpsk_demodulation_kernel(
    const cuComplex* input_symbols,
    uint8_t* output_bits,
    float* soft_bits,
    float noise_variance,
    bool generate_soft,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_symbols) {
        cuComplex symbol = input_symbols[idx];
        float real_part = cuCrealf(symbol);
        float imag_part = cuCimagf(symbol);
        
        int bit_idx = idx * 2;
        
        // Ensure we don't write out of bounds
        if (bit_idx + 1 < total_symbols * 2) {
            if (generate_soft && soft_bits) {
                // Generate Log-Likelihood Ratios (LLRs)
                float sigma_sq = noise_variance;
                float norm_factor = sqrtf(2.0f);
                
                // LLR for bit 0 (real part)
                soft_bits[bit_idx] = 4.0f * real_part * norm_factor / sigma_sq;
                
                // LLR for bit 1 (imaginary part)  
                soft_bits[bit_idx + 1] = 4.0f * imag_part * norm_factor / sigma_sq;
            }
            
            // Hard decision with explicit casting
            output_bits[bit_idx] = (uint8_t)((real_part > 0.0f) ? 0 : 1);
            output_bits[bit_idx + 1] = (uint8_t)((imag_part > 0.0f) ? 0 : 1);
        }
    }
}

__global__ void calculate_evm_kernel(
    const cuComplex* received_symbols,
    const cuComplex* ideal_symbols,
    float* evm_values,
    int total_symbols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_symbols) {
        cuComplex received = received_symbols[idx];
        cuComplex ideal = ideal_symbols[idx];
        
        cuComplex error = cuCsubf(received, ideal);
        float error_magnitude = cuCabsf(error);
        float ideal_magnitude = cuCabsf(ideal);
        
        // EVM as percentage
        evm_values[idx] = (ideal_magnitude > 1e-6f) ? 
                         (error_magnitude / ideal_magnitude) * 100.0f : 0.0f;
    }
}

ModulationMapper::ModulationMapper(const std::string& module_id, const ModulationParams& params)
    : module_id_(module_id), params_(params), d_descriptor_(nullptr) {
    
    // Validate parameters
    if (params_.num_subcarriers <= 0 || params_.num_ofdm_symbols <= 0) {
        throw std::invalid_argument("Number of subcarriers and OFDM symbols must be positive");
    }
    
    // Initialize descriptor
    h_descriptor_ = {};
    h_descriptor_.params = &params_;
    h_descriptor_.total_symbols = calculate_total_symbols();
    h_descriptor_.total_bits = calculate_total_bits();
    h_descriptor_.bits_per_symbol = get_bits_per_symbol();
    h_descriptor_.constellation_size = get_constellation_size();
    
    allocate_gpu_memory();
    setup_port_info();
    initialize_constellation();
}

ModulationMapper::~ModulationMapper() {
    deallocate_gpu_memory();
}

void ModulationMapper::setup_port_info() {
    // Setup input ports based on processing mode
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        input_ports_.resize(input_ports_.size() + 1);
        size_t idx = input_ports_.size() - 1;
        
        // Input port: input_bits
        input_ports_[idx].name = "input_bits";
        input_ports_[idx].tensors.resize(1);
        input_ports_[idx].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorR8U,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.total_bits)}
        );
    }
    
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        input_ports_.resize(input_ports_.size() + 1);
        size_t idx = input_ports_.size() - 1;
        
        // Input port: input_symbols
        input_ports_[idx].name = "input_symbols";
        input_ports_[idx].tensors.resize(1);
        input_ports_[idx].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(params_.num_subcarriers),
                static_cast<std::size_t>(params_.num_ofdm_symbols)
            }
        );
    }
    
    // Setup output ports
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        output_ports_.resize(output_ports_.size() + 1);
        size_t idx = output_ports_.size() - 1;
        
        // Output port: output_symbols
        output_ports_[idx].name = "output_symbols";
        output_ports_[idx].tensors.resize(1);
        output_ports_[idx].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorC32F,
            std::vector<std::size_t>{
                static_cast<std::size_t>(params_.num_subcarriers),
                static_cast<std::size_t>(params_.num_ofdm_symbols)
            }
        );
    }
    
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        output_ports_.resize(output_ports_.size() + 1);
        size_t idx = output_ports_.size() - 1;
        
        // Output port: output_bits
        output_ports_[idx].name = "output_bits";
        output_ports_[idx].tensors.resize(1);
        output_ports_[idx].tensors[0].tensor_info = framework::tensor::TensorInfo(
            framework::tensor::NvDataType::TensorR8U,
            std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.total_bits)}
        );
        
        // Soft bits output (optional)
        if (params_.soft_output) {
            output_ports_.resize(output_ports_.size() + 1);
            size_t soft_idx = output_ports_.size() - 1;
            
            output_ports_[soft_idx].name = "soft_bits";
            output_ports_[soft_idx].tensors.resize(1);
            output_ports_[soft_idx].tensors[0].tensor_info = framework::tensor::TensorInfo(
                framework::tensor::NvDataType::TensorR32F,
                std::vector<std::size_t>{static_cast<std::size_t>(h_descriptor_.total_bits)}
            );
        }
    }
}

std::vector<std::string> ModulationMapper::get_input_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : input_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<std::string> ModulationMapper::get_output_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : output_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<framework::tensor::TensorInfo>
ModulationMapper::get_input_tensor_info(std::string_view port_name) const {
    for (const auto& port : input_ports_) {
        if (port.name == port_name) {
            return {port.tensors[0].tensor_info};
        }
    }
    throw std::invalid_argument("Unknown input port: " + std::string(port_name));
}

std::vector<framework::tensor::TensorInfo>
ModulationMapper::get_output_tensor_info(std::string_view port_name) const {
    for (const auto& port : output_ports_) {
        if (port.name == port_name) {
            return {port.tensors[0].tensor_info};
        }
    }
    throw std::invalid_argument("Unknown output port: " + std::string(port_name));
}

void ModulationMapper::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    // Memory is already allocated in constructor
}

void ModulationMapper::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for modulation mapping
}

void ModulationMapper::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t /*stream*/
) {
    // Update descriptor with current tensor pointers
    h_descriptor_.input_bits = static_cast<const uint8_t*>(current_bits_);
    h_descriptor_.input_symbols = static_cast<const cuComplex*>(current_symbols_);
    h_descriptor_.output_symbols = d_output_symbols_;
    h_descriptor_.output_bits = d_output_bits_;
    h_descriptor_.soft_bits = d_soft_bits_;
    h_descriptor_.evm_values = d_evm_values_;
    h_descriptor_.constellation = d_constellation_;
    
    // Copy descriptor to GPU
    cudaError_t err = cudaMemcpy(d_descriptor_, &h_descriptor_, sizeof(ModulationDescriptor), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy modulation descriptor to GPU");
    }
}

  void ModulationMapper::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    // Reset pointers first
    current_bits_ = nullptr;
    current_symbols_ = nullptr;
    
    // Extract device pointers from input ports
    for (const auto& port : inputs) {
        if (port.name == "input_bits" && !port.tensors.empty()) {
            current_bits_ = port.tensors[0].device_ptr;
        } else if (port.name == "input_symbols" && !port.tensors.empty()) {
            current_symbols_ = port.tensors[0].device_ptr;
        }
    }
}

std::vector<framework::pipeline::PortInfo> ModulationMapper::get_outputs() const {
    // Return a copy of output_ports_ with updated device pointers
    std::vector<framework::pipeline::PortInfo> outputs = output_ports_;
    
    for (auto& port : outputs) {
        if (port.name == "output_symbols" && !port.tensors.empty()) {
            port.tensors[0].device_ptr = d_output_symbols_;
        } else if (port.name == "output_bits" && !port.tensors.empty()) {
            port.tensors[0].device_ptr = d_output_bits_;
        } else if (port.name == "soft_bits" && !port.tensors.empty()) {
            port.tensors[0].device_ptr = d_soft_bits_;
        }
    }
    
    return outputs;
}

void ModulationMapper::execute(cudaStream_t stream) {
    // Validate inputs based on processing mode
    if ((params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) && 
        !current_bits_) {
        throw std::runtime_error("Input bits not set for modulation");
    }
    
    if ((params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) && 
        !current_symbols_) {
        throw std::runtime_error("Input symbols not set for demodulation");
    }
    
    // Clear output buffers first
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        cudaMemset(d_output_bits_, 0, h_descriptor_.total_bits * sizeof(uint8_t));
    }
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        cudaMemset(d_output_symbols_, 0, h_descriptor_.total_symbols * sizeof(cuComplex));
    }
    
    // Copy input data to internal buffers if needed
    if (current_bits_) {
        size_t bits_size = h_descriptor_.total_bits * sizeof(uint8_t);
        cudaError_t err = cudaMemcpyAsync(d_input_bits_, current_bits_, bits_size, 
                                         cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy input bits to internal buffer");
        }
    }
    
    if (current_symbols_) {
        size_t symbols_size = h_descriptor_.total_symbols * sizeof(cuComplex);
        cudaError_t err = cudaMemcpyAsync(d_input_symbols_, current_symbols_, symbols_size,
                                         cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy input symbols to internal buffer");
        }
    }
    
    // Launch modulation/demodulation kernel
    cudaError_t err = launch_modulation_kernel(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Modulation mapping kernel launch failed");
    }
}

void ModulationMapper::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(ModulationDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for descriptor");
    }
    
    // Calculate memory sizes
    size_t bits_size = h_descriptor_.total_bits * sizeof(uint8_t);
    size_t symbols_size = h_descriptor_.total_symbols * sizeof(cuComplex);
    size_t constellation_size = h_descriptor_.constellation_size * sizeof(cuComplex);
    
    // Allocate memory for input bits
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        err = cudaMalloc(&d_input_bits_, bits_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for input bits");
        }
    }
    
    // Allocate memory for input symbols
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        err = cudaMalloc(&d_input_symbols_, symbols_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for input symbols");
        }
    }
    
    // Allocate memory for output symbols
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        err = cudaMalloc(&d_output_symbols_, symbols_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for output symbols");
        }
    }
    
    // Allocate memory for output bits
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        err = cudaMalloc(&d_output_bits_, bits_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for output bits");
        }
        
        // Initialize output bits memory to zero
        err = cudaMemset(d_output_bits_, 0, bits_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize GPU memory for output bits");
        }
        
        // Allocate soft bits if needed
        if (params_.soft_output) {
            size_t soft_bits_size = h_descriptor_.total_bits * sizeof(float);
            err = cudaMalloc(&d_soft_bits_, soft_bits_size);
            if (err != cudaSuccess) {
                throw std::runtime_error("Failed to allocate GPU memory for soft bits");
            }
        }
    }
    
    // Allocate memory for constellation
    err = cudaMalloc(&d_constellation_, constellation_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for constellation");
    }
    
    // Allocate memory for EVM if enabled
    if (params_.enable_evm_calculation) {
        size_t evm_size = h_descriptor_.total_symbols * sizeof(float);
        err = cudaMalloc(&d_evm_values_, evm_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU memory for EVM values");
        }
    }
}

void ModulationMapper::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
    if (d_input_bits_) {
        cudaFree(d_input_bits_);
        d_input_bits_ = nullptr;
    }
    if (d_input_symbols_) {
        cudaFree(d_input_symbols_);
        d_input_symbols_ = nullptr;
    }
    if (d_output_symbols_) {
        cudaFree(d_output_symbols_);
        d_output_symbols_ = nullptr;
    }
    if (d_output_bits_) {
        cudaFree(d_output_bits_);
        d_output_bits_ = nullptr;
    }
    if (d_soft_bits_) {
        cudaFree(d_soft_bits_);
        d_soft_bits_ = nullptr;
    }
    if (d_evm_values_) {
        cudaFree(d_evm_values_);
        d_evm_values_ = nullptr;
    }
    if (d_constellation_) {
        cudaFree(d_constellation_);
        d_constellation_ = nullptr;
    }
}

framework::pipeline::ModuleMemoryRequirements ModulationMapper::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    // Calculate memory requirements based on parameters
    size_t total_bytes = 0;
    
    // Memory for bits and symbols
    total_bytes += h_descriptor_.total_bits * sizeof(uint8_t) * 2; // input and output bits
    total_bytes += h_descriptor_.total_symbols * sizeof(cuComplex) * 2; // input and output symbols
    
    // Memory for soft bits if enabled
    if (params_.soft_output) {
        total_bytes += h_descriptor_.total_bits * sizeof(float);
    }
    
    // Memory for constellation
    total_bytes += h_descriptor_.constellation_size * sizeof(cuComplex);
    
    // Memory for EVM if enabled
    if (params_.enable_evm_calculation) {
        total_bytes += h_descriptor_.total_symbols * sizeof(float);
    }
    
    // Memory for descriptor
    total_bytes += sizeof(ModulationDescriptor);
    
    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = 256; // CUDA memory alignment
    
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics
ModulationMapper::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    chars.provides_fixed_address_for_zero_copy = true;
    return chars;
}

cudaError_t ModulationMapper::launch_modulation_kernel(cudaStream_t stream) {
    dim3 blockSize(256);
    
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        dim3 gridSize((h_descriptor_.total_symbols + blockSize.x - 1) / blockSize.x);
        
        if (params_.scheme == ModulationScheme::QPSK) {
            qpsk_modulation_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_input_bits_,
                d_output_symbols_,
                h_descriptor_.total_symbols
            );
        } else if (params_.scheme == ModulationScheme::QAM16) {
            qam16_modulation_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_input_bits_,
                d_output_symbols_,
                h_descriptor_.total_symbols
            );
        }
        // Add other modulation schemes as needed
    }
    
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        dim3 gridSize((h_descriptor_.total_symbols + blockSize.x - 1) / blockSize.x);
        
        if (params_.scheme == ModulationScheme::QPSK) {
            qpsk_demodulation_kernel<<<gridSize, blockSize, 0, stream>>>(
                d_input_symbols_,
                d_output_bits_,
                d_soft_bits_,
                params_.noise_variance,
                params_.soft_output,
                h_descriptor_.total_symbols
            );
        }
        // Add other demodulation schemes as needed
    }
    
    return cudaGetLastError();
}

void ModulationMapper::initialize_constellation() {
    // Create host constellation points
    std::vector<cuComplex> h_constellation(h_descriptor_.constellation_size);
    
    if (params_.scheme == ModulationScheme::QPSK) {
        float norm = 1.0f / sqrtf(2.0f);
        h_constellation[0] = make_cuComplex( norm,  norm);  // 00
        h_constellation[1] = make_cuComplex( norm, -norm);  // 01
        h_constellation[2] = make_cuComplex(-norm,  norm);  // 10
        h_constellation[3] = make_cuComplex(-norm, -norm);  // 11
    } else if (params_.scheme == ModulationScheme::QAM16) {
        float norm = 1.0f / sqrtf(10.0f);
        // 16-QAM constellation with Gray mapping
        const float coords[] = {-3.0f, -1.0f, 1.0f, 3.0f};
        int idx = 0;
        for (int i = 0; i < 4; ++i) {
            for (int q = 0; q < 4; ++q) {
                h_constellation[idx++] = make_cuComplex(coords[i] * norm, coords[q] * norm);
            }
        }
    }
    // Add other constellation initialization as needed
    
    // Copy to GPU
    size_t constellation_size = h_constellation.size() * sizeof(cuComplex);
    cudaError_t err = cudaMemcpy(d_constellation_, h_constellation.data(), 
                                constellation_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy constellation to GPU");
    }
}

size_t ModulationMapper::calculate_total_symbols() const {
    return params_.num_subcarriers * params_.num_ofdm_symbols;
}

size_t ModulationMapper::calculate_total_bits() const {
    return calculate_total_symbols() * get_bits_per_symbol();
}

int ModulationMapper::get_bits_per_symbol() const {
    switch (params_.scheme) {
        case ModulationScheme::BPSK:  return 1;
        case ModulationScheme::QPSK:  return 2;
        case ModulationScheme::QAM16: return 4;
        case ModulationScheme::QAM64: return 6;
        case ModulationScheme::QAM256: return 8;
        default: return 2;
    }
}

int ModulationMapper::get_constellation_size() const {
    switch (params_.scheme) {
        case ModulationScheme::BPSK:   return 2;
        case ModulationScheme::QPSK:   return 4;
        case ModulationScheme::QAM16:  return 16;
        case ModulationScheme::QAM64:  return 64;
        case ModulationScheme::QAM256: return 256;
        default: return 4;
    }
}

} // namespace modulation_mapping