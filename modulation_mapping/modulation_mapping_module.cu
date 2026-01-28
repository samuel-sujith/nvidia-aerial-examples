#include "modulation_mapping_module.hpp"

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace modulation_mapping {

// CUDA kernels for modulation mapping

__global__ void qpsk_modulation_kernel(const ModulationDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < desc->total_symbols) {
        // QPSK constellation: (1+1j, 1-1j, -1+1j, -1-1j) / sqrt(2)
        int bit_idx = idx * 2;
        uint8_t bit0 = desc->input_bits[bit_idx];
        uint8_t bit1 = desc->input_bits[bit_idx + 1];
        
        float real_part = (bit0 == 0) ? 1.0f : -1.0f;
        float imag_part = (bit1 == 0) ? 1.0f : -1.0f;
        
        // Normalize for unit energy
        float norm_factor = 1.0f / sqrtf(2.0f);
        desc->output_symbols[idx] = make_cuComplex(real_part * norm_factor, imag_part * norm_factor);
    }
}

__global__ void qam16_modulation_kernel(const ModulationDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < desc->total_symbols) {
        // 16-QAM constellation points
        int bit_idx = idx * 4;
        
        // Gray mapping for 16-QAM
        uint8_t bits = (desc->input_bits[bit_idx] << 3) |
                       (desc->input_bits[bit_idx+1] << 2) |
                       (desc->input_bits[bit_idx+2] << 1) |
                       desc->input_bits[bit_idx+3];
        
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
        desc->output_symbols[idx] = make_cuComplex(i_coord * norm_factor, q_coord * norm_factor);
    }
}

__global__ void qpsk_demodulation_kernel(const ModulationDescriptor* desc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < desc->total_symbols) {
        cuComplex symbol = desc->input_symbols[idx];
        float real_part = cuCrealf(symbol);
        float imag_part = cuCimagf(symbol);
        
        int bit_idx = idx * 2;
        
        // Ensure we don't write out of bounds
        if (bit_idx + 1 < desc->total_symbols * 2) {
            if (desc->params->soft_output && desc->soft_bits) {
                // Generate Log-Likelihood Ratios (LLRs)
                float sigma_sq = desc->params->noise_variance;
                float norm_factor = sqrtf(2.0f);
                
                // LLR for bit 0 (real part)
                desc->soft_bits[bit_idx] = 4.0f * real_part * norm_factor / sigma_sq;
                
                // LLR for bit 1 (imaginary part)  
                desc->soft_bits[bit_idx + 1] = 4.0f * imag_part * norm_factor / sigma_sq;
            }
            
            // Hard decision with explicit casting
            desc->output_bits[bit_idx] = (uint8_t)((real_part > 0.0f) ? 0 : 1);
            desc->output_bits[bit_idx + 1] = (uint8_t)((imag_part > 0.0f) ? 0 : 1);
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
    
    setup_port_info();
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

void ModulationMapper::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    if (!mem_slice_.device_tensor_ptr) {
        throw std::runtime_error("Modulation mapping output buffer not allocated");
    }
    std::byte* base = mem_slice_.device_tensor_ptr;
    size_t offset = 0;

    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        d_output_symbols_ = reinterpret_cast<cuComplex*>(base + offset);
        offset += h_descriptor_.total_symbols * sizeof(cuComplex);
    }
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        d_output_bits_ = reinterpret_cast<uint8_t*>(base + offset);
        offset += h_descriptor_.total_bits * sizeof(uint8_t);

        if (params_.soft_output) {
            d_soft_bits_ = reinterpret_cast<float*>(base + offset);
            offset += h_descriptor_.total_bits * sizeof(float);
        }
    }

    allocate_gpu_memory();
    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<ModulationDescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<ModulationDescriptor>(0);
    if (!dynamic_params_gpu_ptr_) {
        throw std::runtime_error("Modulation mapping dynamic descriptor device pointer not allocated");
    }
    d_descriptor_ = dynamic_params_gpu_ptr_;

    dim3 blockSize(256);
    dim3 gridSize((h_descriptor_.total_symbols + blockSize.x - 1) / blockSize.x);
    if (params_.scheme == ModulationScheme::QPSK) {
        mod_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(qpsk_modulation_kernel));
    } else {
        mod_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(qam16_modulation_kernel));
    }
    mod_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(mod_kernel_config_, dynamic_params_gpu_ptr_);

    demod_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(qpsk_demodulation_kernel));
    demod_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(demod_kernel_config_, dynamic_params_gpu_ptr_);

    initialize_constellation();
}

void ModulationMapper::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for modulation mapping
}

void ModulationMapper::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {
    if (dynamic_params_cpu_ptr_) {
        dynamic_params_cpu_ptr_->input_bits = d_input_bits_;
        dynamic_params_cpu_ptr_->input_symbols = d_input_symbols_;
        dynamic_params_cpu_ptr_->output_symbols = d_output_symbols_;
        dynamic_params_cpu_ptr_->output_bits = d_output_bits_;
        dynamic_params_cpu_ptr_->soft_bits = d_soft_bits_;
        dynamic_params_cpu_ptr_->evm_values = d_evm_values_;
        dynamic_params_cpu_ptr_->constellation = d_constellation_;
        dynamic_params_cpu_ptr_->params = d_params_;
        dynamic_params_cpu_ptr_->total_symbols = h_descriptor_.total_symbols;
        dynamic_params_cpu_ptr_->total_bits = h_descriptor_.total_bits;
        dynamic_params_cpu_ptr_->bits_per_symbol = h_descriptor_.bits_per_symbol;
        dynamic_params_cpu_ptr_->constellation_size = h_descriptor_.constellation_size;
        kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);
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
    cudaError_t err;
    err = cudaMalloc(&d_params_, sizeof(ModulationParams));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for params");
    }
    err = cudaMemcpy(d_params_, &params_, sizeof(ModulationParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy params to device");
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
    (void)symbols_size;
    }
    
    // Allocate memory for output bits
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
    (void)bits_size;
        
        // Initialize output bits memory to zero
        err = cudaMemset(d_output_bits_, 0, bits_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to initialize GPU memory for output bits");
        }
        
        // Allocate soft bits if needed
        if (params_.soft_output) {
            size_t soft_bits_size = h_descriptor_.total_bits * sizeof(float);
            (void)soft_bits_size;
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
    if (d_params_) {
        cudaFree(d_params_);
        d_params_ = nullptr;
    }
    d_descriptor_ = nullptr;
    if (d_input_bits_) {
        cudaFree(d_input_bits_);
        d_input_bits_ = nullptr;
    }
    if (d_input_symbols_) {
        cudaFree(d_input_symbols_);
        d_input_symbols_ = nullptr;
    }
    d_output_symbols_ = nullptr;
    d_output_bits_ = nullptr;
    d_soft_bits_ = nullptr;
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
    
    size_t total_bytes = 0;
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        total_bytes += h_descriptor_.total_symbols * sizeof(cuComplex);
    }
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        total_bytes += h_descriptor_.total_bits * sizeof(uint8_t);
        if (params_.soft_output) {
            total_bytes += h_descriptor_.total_bits * sizeof(float);
        }
    }
    reqs.device_tensor_bytes = total_bytes;
    reqs.dynamic_kernel_descriptor_bytes = sizeof(ModulationDescriptor);
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
        const CUresult mod_err = mod_kernel_config_.launch(stream);
        if (mod_err != CUDA_SUCCESS) {
            throw std::runtime_error("Modulation kernel launch failed");
        }
        // Add other modulation schemes as needed
    }
    
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        cudaError_t kernel_err = cudaSuccess;
        if (params_.scheme == ModulationScheme::QPSK) {
            const CUresult demod_err = demod_kernel_config_.launch(stream);
            if (demod_err != CUDA_SUCCESS) {
                throw std::runtime_error("Demodulation kernel launch failed");
            }
            kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                throw std::runtime_error(std::string("QPSK demodulation kernel launch failed: ") + cudaGetErrorString(kernel_err));
            }
            // Synchronize to catch runtime errors
            kernel_err = cudaStreamSynchronize(stream);
            if (kernel_err != cudaSuccess) {
                throw std::runtime_error(std::string("QPSK demodulation kernel execution failed: ") + cudaGetErrorString(kernel_err));
            }
        }
        // Add other demodulation schemes as needed
    }
    
    return cudaSuccess;
}

std::span<const CUgraphNode> ModulationMapper::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    std::span<const CUgraphNode> last = deps;
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        mod_node_ = graph->add_kernel_node(last, mod_kernel_config_.get_kernel_params());
        last = {&mod_node_, 1};
    }
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        demod_node_ = graph->add_kernel_node(last, demod_kernel_config_.get_kernel_params());
        return {&demod_node_, 1};
    }
    return last;
}

void ModulationMapper::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams& /*params*/) {
    if (params_.mode == ProcessingMode::MODULATION || params_.mode == ProcessingMode::BOTH) {
        auto mod_params = mod_kernel_config_.get_kernel_params();
        cuGraphExecKernelNodeSetParams(exec, mod_node_, &mod_params);
    }
    if (params_.mode == ProcessingMode::DEMODULATION || params_.mode == ProcessingMode::BOTH) {
        auto demod_params = demod_kernel_config_.get_kernel_params();
        cuGraphExecKernelNodeSetParams(exec, demod_node_, &demod_params);
    }
}

void ModulationMapper::initialize_constellation() {
    if (!d_constellation_) {
        throw std::runtime_error("Constellation device buffer not allocated");
    }
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
        throw std::runtime_error(std::string("Failed to copy constellation to GPU: ") +
                                 cudaGetErrorString(err));
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