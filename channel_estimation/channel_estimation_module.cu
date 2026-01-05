// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "channel_estimation_module.hpp"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>

namespace channel_estimation {

/// CUDA kernel for least squares channel estimation
__global__ void ls_channel_estimation_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_pilots) return;
    const cuComplex rx_pilot = desc->rx_pilots[tid];
    const cuComplex tx_pilot = desc->tx_pilots[tid];
    cuComplex channel_est = cuCdivf(rx_pilot, tx_pilot);
    channel_est = make_cuComplex(
        cuCrealf(channel_est) * desc->params->beta_scaling,
        cuCimagf(channel_est) * desc->params->beta_scaling
    );
    desc->pilot_estimates[tid] = channel_est; // Write to pilot_estimates
}

/// CUDA kernel for linear interpolation between pilots
__global__ void interpolate_channel_estimates_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_data_subcarriers) return;
    int pilot_spacing = desc->params->pilot_spacing;
    int pilot_before = (tid / pilot_spacing) * pilot_spacing;
    int pilot_after = pilot_before + pilot_spacing;
    if (pilot_before < 0) pilot_before = 0;
    if (pilot_before >= desc->num_pilots) pilot_before = desc->num_pilots - 1;
    if (pilot_after < 0) pilot_after = 0;
    if (pilot_after >= desc->num_pilots) pilot_after = desc->num_pilots - 1;
    if (tid < 10) {
        printf("[INTERP KERNEL] tid=%d, pilot_before=%d, pilot_after=%d\n", tid, pilot_before, pilot_after);
    }
    if (pilot_before == pilot_after) {
        desc->channel_estimates[tid] = desc->pilot_estimates[pilot_before]; // Read from pilot_estimates
    } else {
        float alpha = (float)(tid - pilot_before) / pilot_spacing;
        cuComplex h_before = desc->pilot_estimates[pilot_before];
        cuComplex h_after = desc->pilot_estimates[pilot_after];
        cuComplex interpolated = make_cuComplex(
            (1.0f - alpha) * cuCrealf(h_before) + alpha * cuCrealf(h_after),
            (1.0f - alpha) * cuCimagf(h_before) + alpha * cuCimagf(h_after)
        );
        desc->channel_estimates[tid] = interpolated;
    }
}

ChannelEstimator::ChannelEstimator(
    const std::string& module_id,
    const ChannelEstParams& params
) : module_id_(module_id), params_(params), d_descriptor_(nullptr) {
    
    setup_port_info();
}

ChannelEstimator::~ChannelEstimator() {
    deallocate_gpu_memory();
}

void ChannelEstimator::setup_port_info() {
    using namespace framework::tensor;
    
    // Setup input ports
    input_ports_.resize(2);
    
    // Input port 0: rx_pilots
    input_ports_[0].name = "rx_pilots";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    // Input port 1: tx_pilots  
    input_ports_[1].name = "tx_pilots";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    // Setup output ports
    output_ports_.resize(1);
    
    // Output port 0: channel_estimates
    output_ports_[0].name = "channel_estimates";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_resource_blocks * 12),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
}

void ChannelEstimator::setup_memory(const framework::pipeline::ModuleMemorySlice& /*memory_slice*/) {
    allocate_gpu_memory();
}

void ChannelEstimator::warmup(cudaStream_t /*stream*/) {
    // No specific warmup needed for channel estimation
}

void ChannelEstimator::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {

    std::cout << "[DEBUG] ChannelEstimator::configure_io (before assign): this=" << this
          << ", &h_descriptor_=" << &h_descriptor_
          << ", current_rx_pilots_=" << (const void*)current_rx_pilots_
          << ", current_tx_pilots_=" << (const void*)current_tx_pilots_
          << ", current_channel_estimates_=" << (const void*)current_channel_estimates_ << std::endl;
    // Update descriptor with current tensor pointers
    h_descriptor_.rx_pilots = current_rx_pilots_;
    h_descriptor_.tx_pilots = current_tx_pilots_;
    h_descriptor_.channel_estimates = current_channel_estimates_;
    h_descriptor_.params = &params_;
    h_descriptor_.num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    h_descriptor_.num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    h_descriptor_.pilot_estimates = d_pilot_estimates_; // NEW
    
    std::cout << "[DEBUG] ChannelEstimator::configure_io (after assign): h_descriptor_.rx_pilots=" << (const void*)h_descriptor_.rx_pilots
          << ", h_descriptor_.tx_pilots=" << (const void*)h_descriptor_.tx_pilots
          << ", h_descriptor_.channel_estimates=" << (const void*)h_descriptor_.channel_estimates << std::endl;
    // Copy descriptor to GPU
    cudaError_t err = cudaMemcpyAsync(
        d_descriptor_,
        &h_descriptor_,
        sizeof(ChannelEstDescriptor),
        cudaMemcpyHostToDevice,
        stream
    );
    
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy descriptor to GPU");
    }
}

void ChannelEstimator::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    // Accept both input and output ports in the vector
    for (const auto& port : inputs) {
        if (port.name == "rx_pilots" && !port.tensors.empty()) {
            current_rx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "tx_pilots" && !port.tensors.empty()) {
            current_tx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "channel_estimates" && !port.tensors.empty()) {
            current_channel_estimates_ = static_cast<cuComplex*>(port.tensors[0].device_ptr);
        }
    }

    std::cout << "[DEBUG] ChannelEstimator::set_inputs: this=" << this
          << ", &h_descriptor_=" << &h_descriptor_
          << ", current_rx_pilots_=" << (const void*)current_rx_pilots_
          << ", current_tx_pilots_=" << (const void*)current_tx_pilots_
          << ", current_channel_estimates_=" << (const void*)current_channel_estimates_ << std::endl;
}

std::vector<framework::pipeline::PortInfo> ChannelEstimator::get_outputs() const {
    // Return a copy of output_ports_ with updated device pointers
    std::vector<framework::pipeline::PortInfo> outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        // Use user-provided output buffer if set, else use module's own buffer
        if (current_channel_estimates_) {
            outputs[0].tensors[0].device_ptr = current_channel_estimates_;
        } else {
            outputs[0].tensors[0].device_ptr = d_channel_estimates_;
        }
    }
    return outputs;
}

void ChannelEstimator::execute(cudaStream_t stream) {
    // Zero-initialize the output buffer to avoid uninitialized reads
    cudaError_t err = cudaMemset(current_channel_estimates_, 0, params_.num_resource_blocks * 12 * params_.num_ofdm_symbols * sizeof(cuComplex));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaMemset failed");
    }


    // Defensive checks and debug prints for all device memory
    std::cout << "[DEBUG] ChannelEstimator::execute: d_descriptor_=" << (const void*)d_descriptor_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: d_pilot_symbols_=" << (const void*)d_pilot_symbols_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: d_pilot_estimates_=" << (const void*)d_pilot_estimates_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: d_channel_estimates_=" << (const void*)d_channel_estimates_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: h_descriptor_.pilot_estimates=" << (const void*)h_descriptor_.pilot_estimates << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: current_rx_pilots_=" << (const void*)current_rx_pilots_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: current_tx_pilots_=" << (const void*)current_tx_pilots_ << std::endl;
    std::cout << "[DEBUG] ChannelEstimator::execute: current_channel_estimates_=" << (const void*)current_channel_estimates_ << std::endl;

    if (!d_descriptor_) {
        std::cerr << "[ERROR] d_descriptor_ is nullptr!" << std::endl;
        throw std::runtime_error("d_descriptor_ is nullptr");
    }
    if (!d_pilot_symbols_) {
        std::cerr << "[ERROR] d_pilot_symbols_ is nullptr!" << std::endl;
        throw std::runtime_error("d_pilot_symbols_ is nullptr");
    }
    if (!d_pilot_estimates_) {
        std::cerr << "[ERROR] d_pilot_estimates_ is nullptr!" << std::endl;
        throw std::runtime_error("d_pilot_estimates_ is nullptr");
    }
    if (!d_channel_estimates_) {
        std::cerr << "[ERROR] d_channel_estimates_ is nullptr!" << std::endl;
        throw std::runtime_error("d_channel_estimates_ is nullptr");
    }
    if (!h_descriptor_.pilot_estimates) {
        std::cerr << "[ERROR] h_descriptor_.pilot_estimates is nullptr!" << std::endl;
        throw std::runtime_error("h_descriptor_.pilot_estimates is nullptr");
    }
    if (!current_rx_pilots_) {
        std::cerr << "[ERROR] current_rx_pilots_ is nullptr!" << std::endl;
        throw std::runtime_error("current_rx_pilots_ is nullptr");
    }
    if (!current_tx_pilots_) {
        std::cerr << "[ERROR] current_tx_pilots_ is nullptr!" << std::endl;
        throw std::runtime_error("current_tx_pilots_ is nullptr");
    }
    if (!current_channel_estimates_) {
        std::cerr << "[ERROR] current_channel_estimates_ is nullptr!" << std::endl;
        throw std::runtime_error("current_channel_estimates_ is nullptr");
    }

    err = launch_channel_estimation_kernel(stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Channel estimation kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Channel estimation kernel launch failed");
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] cudaStreamSynchronize failed after kernel: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("cudaStreamSynchronize failed after kernel");
    }
}

void ChannelEstimator::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(ChannelEstDescriptor));
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate GPU memory for descriptor" << std::endl;
        throw std::runtime_error("Failed to allocate GPU memory for descriptor");
    }

    // Allocate memory for pilot symbols and channel estimates
    size_t pilot_size = params_.num_resource_blocks * 12 / params_.pilot_spacing * sizeof(cuComplex);
    size_t estimates_size = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols * sizeof(cuComplex);

    err = cudaMalloc(&d_pilot_symbols_, pilot_size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate GPU memory for pilot symbols" << std::endl;
        throw std::runtime_error("Failed to allocate GPU memory for pilot symbols");
    }
    err = cudaMalloc(&d_pilot_estimates_, pilot_size); // NEW
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate GPU memory for pilot estimates" << std::endl;
        throw std::runtime_error("Failed to allocate GPU memory for pilot estimates");
    }
    err = cudaMalloc(&d_channel_estimates_, estimates_size);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to allocate GPU memory for channel estimates" << std::endl;
        throw std::runtime_error("Failed to allocate GPU memory for channel estimates");
    }

    // Debug prints for all device memory after allocation
    std::cout << "[DEBUG] allocate_gpu_memory: d_descriptor_=" << (const void*)d_descriptor_ << std::endl;
    std::cout << "[DEBUG] allocate_gpu_memory: d_pilot_symbols_=" << (const void*)d_pilot_symbols_ << std::endl;
    std::cout << "[DEBUG] allocate_gpu_memory: d_pilot_estimates_=" << (const void*)d_pilot_estimates_ << std::endl;
    std::cout << "[DEBUG] allocate_gpu_memory: d_channel_estimates_=" << (const void*)d_channel_estimates_ << std::endl;
}

void ChannelEstimator::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
    if (d_pilot_symbols_) {
        cudaFree(d_pilot_symbols_);
        d_pilot_symbols_ = nullptr;
    }
    if (d_pilot_estimates_) { cudaFree(d_pilot_estimates_); d_pilot_estimates_ = nullptr; }
    if (d_channel_estimates_) {
        cudaFree(d_channel_estimates_);
        d_channel_estimates_ = nullptr;
    }
}

cudaError_t ChannelEstimator::launch_channel_estimation_kernel(cudaStream_t stream) {
    int num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    dim3 blockSize(256);
    dim3 gridSize((num_pilots + blockSize.x - 1) / blockSize.x);
    ls_channel_estimation_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    int num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    dim3 gridSizeInterp((num_data_subcarriers + blockSize.x - 1) / blockSize.x);
    interpolate_channel_estimates_kernel<<<gridSizeInterp, blockSize, 0, stream>>>(d_descriptor_);
    return cudaGetLastError();
}

framework::pipeline::ModuleMemoryRequirements ChannelEstimator::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    // Calculate memory requirements based on parameters
    size_t total_bytes = 0;
    
    // Memory for pilot symbols
    total_bytes += params_.num_rx_antennas * params_.num_ofdm_symbols * 
                   params_.num_resource_blocks * sizeof(cuComplex);
    
    // Memory for channel estimates
    total_bytes += params_.num_rx_antennas * params_.num_ofdm_symbols * 
                   params_.num_resource_blocks * 12 * sizeof(cuComplex);
    
    reqs.device_tensor_bytes = total_bytes;
    reqs.alignment = 256; // CUDA memory alignment
    
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics
ChannelEstimator::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    
    if (port_name == "channel_estimates") {
        chars.provides_fixed_address_for_zero_copy = true;
    }
    
    return chars;
}

std::vector<std::string> ChannelEstimator::get_input_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : input_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<std::string> ChannelEstimator::get_output_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : output_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<framework::tensor::TensorInfo> ChannelEstimator::get_input_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : input_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

std::vector<framework::tensor::TensorInfo> ChannelEstimator::get_output_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : output_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

} // namespace channel_estimation