/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_estimation_module.hpp"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace channel_estimation {

/// CUDA kernel for least squares channel estimation
__global__ void ls_channel_estimation_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= desc->num_pilots) return;
    
    const cuComplex rx_pilot = desc->rx_pilots[tid];
    const cuComplex tx_pilot = desc->tx_pilots[tid];
    
    // Least squares: H = Y / X (element-wise division)
    cuComplex channel_est = cuCdivf(rx_pilot, tx_pilot);
    
    // Apply beta scaling
    channel_est = make_cuComplex(
        cuCrealf(channel_est) * desc->params->beta_scaling,
        cuCimagf(channel_est) * desc->params->beta_scaling
    );
    
    desc->channel_estimates[tid] = channel_est;
}

/// CUDA kernel for linear interpolation between pilots
__global__ void interpolate_channel_estimates_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= desc->num_data_subcarriers) return;
    
    int pilot_spacing = desc->params->pilot_spacing;
    
    // Find surrounding pilot indices
    int pilot_before = (tid / pilot_spacing) * pilot_spacing;
    int pilot_after = pilot_before + pilot_spacing;
    
    // Boundary checks
    if (pilot_after >= desc->num_pilots) {
        pilot_after = pilot_before;
    }
    
    if (pilot_before == pilot_after) {
        // Use nearest pilot
        desc->channel_estimates[tid] = desc->channel_estimates[pilot_before];
    } else {
        // Linear interpolation
        float alpha = (float)(tid - pilot_before) / pilot_spacing;
        
        cuComplex h_before = desc->channel_estimates[pilot_before];
        cuComplex h_after = desc->channel_estimates[pilot_after];
        
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
    
    setup_tensor_info();
}

ChannelEstimator::~ChannelEstimator() {
    deallocate_gpu_memory();
}

void ChannelEstimator::setup_tensor_info() {
    using namespace framework::tensor;
    
    // Input 0: Received pilot symbols [num_pilots]
    input_tensor_info_.emplace_back(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    // Input 1: Reference pilot symbols [num_pilots]  
    input_tensor_info_.emplace_back(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    // Output 0: Channel estimates [num_subcarriers * num_symbols]
    output_tensor_info_.emplace_back(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_resource_blocks * 12),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
}

void ChannelEstimator::setup(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    allocate_gpu_memory();
}

void ChannelEstimator::warmup(cudaStream_t stream) {
    // No specific warmup needed for channel estimation
}

void ChannelEstimator::configure_io(
    const framework::pipeline::DynamicParams& params,
    std::span<const framework::tensor::TensorInfo> inputs,
    std::span<framework::tensor::TensorInfo> outputs,
    cudaStream_t stream
) {
    // Validate tensor count
    if (inputs.size() != 2) {
        throw std::runtime_error("Channel estimator requires exactly 2 inputs");
    }
    if (outputs.size() != 1) {
        throw std::runtime_error("Channel estimator requires exactly 1 output");
    }
    
    // Store pointers to current tensors (these will be set by the pipeline)
    // For now, we assume tensors have device_ptr field or similar
    // In actual implementation, this would get device pointers from TensorInfo
    
    // Update descriptor with current tensor pointers
    h_descriptor_.rx_pilots = current_rx_pilots_;
    h_descriptor_.tx_pilots = current_tx_pilots_;
    h_descriptor_.channel_estimates = current_channel_estimates_;
    h_descriptor_.params = &params_;
    h_descriptor_.num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    h_descriptor_.num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    
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

void ChannelEstimator::execute(cudaStream_t stream) {
    cudaError_t err = launch_channel_estimation_kernel(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Channel estimation kernel launch failed");
    }
}

framework::pipeline::ModuleMemoryRequirements ChannelEstimator::get_memory_requirements() const {
    framework::pipeline::ModuleMemoryRequirements req;
    
    // Static descriptor
    req.static_kernel_descriptor_bytes = sizeof(ChannelEstDescriptor);
    
    // No dynamic descriptor needed for this module
    req.dynamic_kernel_descriptor_bytes = 0;
    
    // Device tensor memory (managed by pipeline)
    req.device_tensor_bytes = 0;
    
    return req;
}

std::span<const framework::tensor::TensorInfo> ChannelEstimator::get_input_tensor_info() const {
    return input_tensor_info_;
}

std::span<const framework::tensor::TensorInfo> ChannelEstimator::get_output_tensor_info() const {
    return output_tensor_info_;
}

void ChannelEstimator::allocate_gpu_memory() {
    // Allocate GPU memory for descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(ChannelEstDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for descriptor");
    }
}

void ChannelEstimator::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
}

cudaError_t ChannelEstimator::launch_channel_estimation_kernel(cudaStream_t stream) {
    int num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    
    // Launch LS estimation kernel
    dim3 blockSize(256);
    dim3 gridSize((num_pilots + blockSize.x - 1) / blockSize.x);
    
    ls_channel_estimation_kernel<<<gridSize, blockSize, 0, stream>>>(d_descriptor_);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Launch interpolation kernel for data subcarriers
    int num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    dim3 gridSizeInterp((num_data_subcarriers + blockSize.x - 1) / blockSize.x);
    
    interpolate_channel_estimates_kernel<<<gridSizeInterp, blockSize, 0, stream>>>(d_descriptor_);
    
    return cudaGetLastError();
}

} // namespace channel_estimation