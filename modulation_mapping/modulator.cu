/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "modulator.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <stdexcept>

namespace framework {\nnamespace examples {

// QAM constellation look-up tables
__device__ __constant__ float qpsk_table[4] = {
    0.707106781f, -0.707106781f, 0.707106781f, -0.707106781f
};

__device__ __constant__ float qam16_table[8] = {
    0.316227766f, -0.316227766f, 0.316227766f, -0.316227766f,
    0.948683298f, -0.948683298f, 0.948683298f, -0.948683298f
};

__device__ __constant__ float qam64_table[16] = {
    0.462910049f, -0.462910049f, 0.77151675f, -0.77151675f,
    0.154303350f, -0.154303350f, 1.08012345f, -1.08012345f,
    0.462910049f, -0.462910049f, 0.77151675f, -0.77151675f,
    0.154303350f, -0.154303350f, 1.08012345f, -1.08012345f
};

__device__ __constant__ float qam256_table[32] = {
    0.383482494f, -0.383482494f, 0.843661488f, -0.843661488f,
    0.230089497f, -0.230089497f, 0.997054486f, -0.997054486f,
    0.536875492f, -0.536875492f, 0.690268490f, -0.690268490f,
    0.076696499f, -0.076696499f, 1.150447483f, -1.150447483f,
    0.383482494f, -0.383482494f, 0.843661488f, -0.843661488f,
    0.230089497f, -0.230089497f, 0.997054486f, -0.997054486f,
    0.536875492f, -0.536875492f, 0.690268490f, -0.690268490f,
    0.076696499f, -0.076696499f, 1.150447483f, -1.150447483f
};

__device__ cuComplex modulate_qpsk(uint32_t bits) {
    cuComplex symbol;
    symbol.x = ((bits & 0x1) == 0) ? qpsk_table[0] : qpsk_table[1];
    symbol.y = ((bits & 0x2) == 0) ? qpsk_table[0] : qpsk_table[1];
    return symbol;
}

__device__ cuComplex modulate_qam16(uint32_t bits) {
    cuComplex symbol;
    symbol.x = qam16_table[bits & 0x05];
    symbol.y = qam16_table[(bits >> 1) & 0x05];
    return symbol;
}

__device__ cuComplex modulate_qam64(uint32_t bits) {
    int x_index = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2);
    int y_index = ((bits & 0x2) >> 1) | ((bits & 0x8) >> 2) | ((bits & 0x20) >> 3);
    
    cuComplex symbol;
    symbol.x = qam64_table[x_index];
    symbol.y = qam64_table[y_index + 8];
    return symbol;
}

__device__ cuComplex modulate_qam256(uint32_t bits) {
    int x_index = (bits & 0x1) | ((bits & 0x4) >> 1) | ((bits & 0x10) >> 2) | ((bits & 0x40) >> 3);
    int y_index = ((bits & 0x2) >> 1) | ((bits & 0x8) >> 2) | ((bits & 0x20) >> 3) | ((bits & 0x80) >> 4);
    
    cuComplex symbol;
    symbol.x = qam256_table[x_index];
    symbol.y = qam256_table[y_index + 16];
    return symbol;
}

// CUDA kernel for QAM modulation
__global__ void qam_modulation_kernel(ModulationDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= desc->total_symbols) {
        return;
    }
    
    // Extract bits for this symbol based on modulation scheme
    int bits_per_symbol;
    switch (desc->params->scheme) {
        case ModulationScheme::QPSK:    bits_per_symbol = 2; break;
        case ModulationScheme::QAM_16:  bits_per_symbol = 4; break;
        case ModulationScheme::QAM_64:  bits_per_symbol = 6; break;
        case ModulationScheme::QAM_256: bits_per_symbol = 8; break;
        default: return;
    }
    
    int bit_offset = tid * bits_per_symbol;
    int word_index = bit_offset / 32;
    int bit_shift = bit_offset % 32;
    
    uint32_t symbol_bits = (desc->input_bits[word_index] >> bit_shift) & ((1U << bits_per_symbol) - 1);
    
    // Handle bits spanning across word boundaries
    if (bit_shift + bits_per_symbol > 32) {
        int bits_from_next = (bit_shift + bits_per_symbol) - 32;
        uint32_t next_bits = desc->input_bits[word_index + 1] & ((1U << bits_from_next) - 1);
        symbol_bits |= (next_bits << (32 - bit_shift));
    }
    
    // Modulate based on scheme
    cuComplex symbol;
    switch (desc->params->scheme) {
        case ModulationScheme::QPSK:
            symbol = modulate_qpsk(symbol_bits);
            break;
        case ModulationScheme::QAM_16:
            symbol = modulate_qam16(symbol_bits);
            break;
        case ModulationScheme::QAM_64:
            symbol = modulate_qam64(symbol_bits);
            break;
        case ModulationScheme::QAM_256:
            symbol = modulate_qam256(symbol_bits);
            break;
    }
    
    // Apply scaling
    symbol.x *= desc->params->scaling_factor;
    symbol.y *= desc->params->scaling_factor;
    
    desc->output_symbols[tid] = symbol;
}

QAMModulator::QAMModulator(
    const std::string& module_id,
    const ModulationParams& params
) : module_id_(module_id), params_(params), d_descriptor_(nullptr) {
    allocate_gpu_memory();
}
QAMModulator::~QAMModulator() {
    deallocate_gpu_memory();
}

void QAMModulator::allocate_gpu_memory() {
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(ModulationDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for modulation descriptor");
    }
}

void QAMModulator::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
}

cudaError_t QAMModulator::setup_modulation_kernel(cudaStream_t stream) {
    // Copy descriptor to device
    return cudaMemcpyAsync(
        d_descriptor_, &h_descriptor_,
        sizeof(ModulationDescriptor),
        cudaMemcpyHostToDevice, stream
    );
}

cudaError_t QAMModulator::launch_modulation_kernel(cudaStream_t stream) {
    int threads_per_block = 256;
    int blocks = (params_.num_symbols + threads_per_block - 1) / threads_per_block;
    
    qam_modulation_kernel<<<blocks, threads_per_block, 0, stream>>>(d_descriptor_);
    
    return cudaGetLastError();
}

task::TaskResult QAMModulator::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token
) {
    if (token.is_cancellation_requested()) {
        return task::TaskResult(task::TaskStatus::Cancelled, "Task cancelled");
    }
    
    try {
        if (inputs.empty() || outputs.empty()) {
            return task::TaskResult(task::TaskStatus::Failed, "Invalid inputs/outputs");
        }
        
        // Setup descriptor
        h_descriptor_.input_bits = reinterpret_cast<const uint32_t*>(inputs[0].data());
        h_descriptor_.output_symbols = reinterpret_cast<cuComplex*>(outputs[0].data());
        h_descriptor_.params = const_cast<ModulationParams*>(&params_);
        h_descriptor_.total_symbols = params_.num_symbols;
        
        cudaStream_t stream = 0; // Use default stream
        
        // Setup and launch kernel
        cudaError_t err = setup_modulation_kernel(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Setup failed");
        }
        
        err = launch_modulation_kernel(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Kernel launch failed");
        }
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Synchronization failed");
        }
        
        return task::TaskResult(task::TaskStatus::Completed, "Modulation completed");
        
    } catch (const std::exception& e) {
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}

} // namespace examples\n} // namespace framework