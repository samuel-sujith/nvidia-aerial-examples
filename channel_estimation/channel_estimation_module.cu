// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "channel_estimation_module.hpp"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>  // std::min
#include <cstdio>

namespace channel_estimation {

/// CUDA kernel for least squares channel estimation (SAFE)
__global__ void ls_channel_estimation_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_pilots || !desc->params) return;
    
    const cuComplex rx_pilot = desc->rx_pilots[tid];
    const cuComplex tx_pilot = desc->tx_pilots[tid];
    
    // Safe division
    float tx_real = cuCrealf(tx_pilot);
    float tx_imag = cuCimagf(tx_pilot);
    if (tx_real == 0.0f && tx_imag == 0.0f) {
        desc->pilot_estimates[tid] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    
    cuComplex channel_est = cuCdivf(rx_pilot, tx_pilot);
    channel_est = make_cuComplex(
        cuCrealf(channel_est) * desc->params->beta_scaling,
        cuCimagf(channel_est) * desc->params->beta_scaling
    );
    desc->pilot_estimates[tid] = channel_est;
}

/// CUDA kernel for linear interpolation (SAFE)
__global__ void interpolate_channel_estimates_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_data_subcarriers || !desc->params) return;
    
    int pilot_spacing = desc->params->pilot_spacing;
    if (pilot_spacing <= 0) return;  // Safety
    
    int pilot_idx = tid % (desc->num_pilots * pilot_spacing);
    int pilot_before = (pilot_idx / pilot_spacing) * pilot_spacing;
    int pilot_after = pilot_before + pilot_spacing;
    
    // Clamp safely
    pilot_before = min(pilot_before, desc->num_pilots - 1);
    pilot_after = min(pilot_after, desc->num_pilots - 1);
    
    if (pilot_before < 0 || pilot_after < 0) {
        desc->channel_estimates[tid] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    
    if (pilot_before == pilot_after) {
        desc->channel_estimates[tid] = desc->pilot_estimates[pilot_before];
    } else {
        float alpha = static_cast<float>(pilot_idx - pilot_before) / static_cast<float>(pilot_spacing);
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
) : module_id_(module_id), params_(params), d_descriptor_(nullptr), d_params_(nullptr) {
    setup_port_info();
}

ChannelEstimator::~ChannelEstimator() {
    deallocate_gpu_memory();
}

void ChannelEstimator::setup_port_info() {
    using namespace framework::tensor;
    
    input_ports_.resize(2);
    
    input_ports_[0].name = "rx_pilots";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    input_ports_[1].name = "tx_pilots";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    output_ports_.resize(1);
    
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
    // No warmup needed
}

void ChannelEstimator::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {
    // Update host descriptor
    h_descriptor_.rx_pilots = current_rx_pilots_;
    h_descriptor_.tx_pilots = current_tx_pilots_;
    h_descriptor_.channel_estimates = current_channel_estimates_;
    h_descriptor_.params = d_params_;  // FIXED: Device params pointer
    h_descriptor_.num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    h_descriptor_.num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    h_descriptor_.pilot_estimates = d_pilot_estimates_;
    
    // Copy to device
    cudaError_t err = cudaMemcpyAsync(
        d_descriptor_,
        &h_descriptor_,
        sizeof(ChannelEstDescriptor),
        cudaMemcpyHostToDevice,
        stream
    );
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpyAsync descriptor failed: ") + cudaGetErrorString(err));
    }
}

void ChannelEstimator::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    for (const auto& port : inputs) {
        if (port.name == "rx_pilots" && !port.tensors.empty()) {
            current_rx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "tx_pilots" && !port.tensors.empty()) {
            current_tx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "channel_estimates" && !port.tensors.empty()) {
            current_channel_estimates_ = static_cast<cuComplex*>(port.tensors[0].device_ptr);
        }
    }
}

std::vector<framework::pipeline::PortInfo> ChannelEstimator::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty() && current_channel_estimates_) {
        outputs[0].tensors[0].device_ptr = current_channel_estimates_;
    }
    return outputs;
}

void ChannelEstimator::execute(cudaStream_t stream) {
    // Validate pointers
    if (!current_rx_pilots_ || !current_tx_pilots_ || !current_channel_estimates_) {
        throw std::runtime_error("Missing input/output pointers");
    }
    
    // Zero output
    size_t out_size = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    cudaError_t err = cudaMemsetAsync(current_channel_estimates_, 0, out_size, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(err));
    }
    
    // Configure & copy descriptor
    configure_io({}, stream);
    
    // Launch kernels
    err = launch_channel_estimation_kernel(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Stream sync failed: ") + cudaGetErrorString(err));
    }
}

void ChannelEstimator::allocate_gpu_memory() {
    cudaError_t err;
    
    // Descriptor
    err = cudaMalloc(&d_descriptor_, sizeof(ChannelEstDescriptor));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_descriptor_ failed");
    
    // FIXED: Device params copy
    err = cudaMalloc(&d_params_, sizeof(ChannelEstParams));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_params_ failed");
    err = cudaMemcpy(d_params_, &params_, sizeof(ChannelEstParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy d_params_ failed");
    
    // Buffers
    size_t pilot_size = params_.num_resource_blocks * 12ULL / params_.pilot_spacing * sizeof(cuComplex);
    size_t estimates_size = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    
    err = cudaMalloc(&d_pilot_symbols_, pilot_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_pilot_symbols_ failed");
    
    err = cudaMalloc(&d_pilot_estimates_, pilot_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_pilot_estimates_ failed");
    cudaMemset(d_pilot_estimates_, 0, pilot_size);  // Zero init
    
    err = cudaMalloc(&d_channel_estimates_, estimates_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_channel_estimates_ failed");
    
    std::cout << "[DEBUG] Allocated: d_params_=" << d_params_ 
              << " d_pilot_estimates_=" << d_pilot_estimates_ << std::endl;
}

void ChannelEstimator::deallocate_gpu_memory() {
    // Individual frees to avoid type mismatch
    if (d_params_) { cudaFree(d_params_); d_params_ = nullptr; }
    if (d_descriptor_) { cudaFree(d_descriptor_); d_descriptor_ = nullptr; }
    if (d_pilot_symbols_) { cudaFree(d_pilot_symbols_); d_pilot_symbols_ = nullptr; }
    if (d_pilot_estimates_) { cudaFree(d_pilot_estimates_); d_pilot_estimates_ = nullptr; }
    if (d_channel_estimates_) { cudaFree(d_channel_estimates_); d_channel_estimates_ = nullptr; }
}

cudaError_t ChannelEstimator::launch_channel_estimation_kernel(cudaStream_t stream) {
    int num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    dim3 blockSize(256);
    dim3 gridSize((num_pilots + blockSize.x - 1) / blockSize.x);
    
    ls_channel_estimation_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    int num_data = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    dim3 gridInterp((num_data + blockSize.x - 1) / blockSize.x);
    interpolate_channel_estimates_kernel<<<gridInterp, blockSize, 0, stream>>>(d_descriptor_);
    
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