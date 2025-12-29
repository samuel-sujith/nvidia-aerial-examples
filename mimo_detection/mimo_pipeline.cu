/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mimo_detector.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <stdexcept>

// MIMO detection CUDA kernels
__device__ void matrix_2x2_inverse(
    const cuComplex* h_matrix,
    cuComplex* h_inv,
    int sc_idx
) {
    // For 2x2 matrix inversion: inv(H) = (1/det) * adj(H)
    int offset = sc_idx * 4; // 2x2 matrix per subcarrier
    
    cuComplex h00 = h_matrix[offset + 0];
    cuComplex h01 = h_matrix[offset + 1]; 
    cuComplex h10 = h_matrix[offset + 2];
    cuComplex h11 = h_matrix[offset + 3];
    
    // Calculate determinant: det = h00*h11 - h01*h10
    cuComplex det = cuCsubf(cuCmulf(h00, h11), cuCmulf(h01, h10));
    
    // Calculate 1/det
    float det_mag_sq = det.x * det.x + det.y * det.y;
    if (det_mag_sq < 1e-10f) {
        // Singular matrix - set to zero
        h_inv[offset + 0] = make_cuComplex(0.0f, 0.0f);
        h_inv[offset + 1] = make_cuComplex(0.0f, 0.0f);
        h_inv[offset + 2] = make_cuComplex(0.0f, 0.0f);
        h_inv[offset + 3] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    
    cuComplex det_inv = make_cuComplex(det.x / det_mag_sq, -det.y / det_mag_sq);
    
    // Adjugate matrix: [h11, -h01; -h10, h00]
    h_inv[offset + 0] = cuCmulf(det_inv, h11);           // h11/det
    h_inv[offset + 1] = cuCmulf(det_inv, cuCsubf(make_cuComplex(0,0), h01)); // -h01/det
    h_inv[offset + 2] = cuCmulf(det_inv, cuCsubf(make_cuComplex(0,0), h10)); // -h10/det  
    h_inv[offset + 3] = cuCmulf(det_inv, h00);           // h00/det
}

__global__ void mimo_zf_detection_kernel(MIMODetectionDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_subcarriers = desc->params->num_subcarriers;
    int num_symbols = desc->params->num_symbols;
    int num_tx = desc->params->num_tx_antennas;
    int num_rx = desc->params->num_rx_antennas;
    
    if (tid >= num_subcarriers * num_symbols) {
        return;
    }
    
    int sc_idx = tid % num_subcarriers;
    int sym_idx = tid / num_subcarriers;
    
    // For 2x2 MIMO system - extend for larger systems
    if (num_tx == 2 && num_rx == 2) {
        // Get channel matrix for this subcarrier/symbol
        int h_offset = (sym_idx * num_subcarriers + sc_idx) * num_rx * num_tx;
        const cuComplex* h_matrix = &desc->channel_matrix[h_offset];
        
        // Calculate H^(-1) using shared memory for efficiency
        __shared__ cuComplex h_inv_shared[256 * 4]; // Assuming max 256 threads per block
        cuComplex* h_inv = &h_inv_shared[threadIdx.x * 4];
        
        matrix_2x2_inverse(h_matrix, h_inv, 0);
        
        // Get received signal
        int rx_offset = sym_idx * num_subcarriers * num_rx + sc_idx * num_rx;
        cuComplex y0 = desc->received_signal[rx_offset + 0];
        cuComplex y1 = desc->received_signal[rx_offset + 1];
        
        // Zero-forcing detection: x = H^(-1) * y
        cuComplex x0 = cuCaddf(cuCmulf(h_inv[0], y0), cuCmulf(h_inv[1], y1));
        cuComplex x1 = cuCaddf(cuCmulf(h_inv[2], y0), cuCmulf(h_inv[3], y1));
        
        // Store detected symbols
        int out_offset = sym_idx * num_subcarriers * num_tx + sc_idx * num_tx;
        desc->detected_symbols[out_offset + 0] = x0;
        desc->detected_symbols[out_offset + 1] = x1;
    }
}

__global__ void mimo_mmse_detection_kernel(MIMODetectionDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_subcarriers = desc->params->num_subcarriers;
    int num_symbols = desc->params->num_symbols;
    int num_tx = desc->params->num_tx_antennas;
    int num_rx = desc->params->num_rx_antennas;
    
    if (tid >= num_subcarriers * num_symbols) {
        return;
    }
    
    int sc_idx = tid % num_subcarriers;
    int sym_idx = tid / num_subcarriers;
    float noise_var = desc->params->noise_variance;
    
    // For 2x2 MIMO: MMSE filter = H^H * (H*H^H + σ²I)^(-1)
    if (num_tx == 2 && num_rx == 2) {
        int h_offset = (sym_idx * num_subcarriers + sc_idx) * num_rx * num_tx;
        const cuComplex* h = &desc->channel_matrix[h_offset];
        
        // Calculate Gram matrix: G = H*H^H + σ²I
        cuComplex g00 = cuCaddf(
            cuCaddf(cuCmulf(h[0], cuConjf(h[0])), cuCmulf(h[1], cuConjf(h[1]))),
            make_cuComplex(noise_var, 0.0f)
        );
        cuComplex g01 = cuCaddf(cuCmulf(h[0], cuConjf(h[2])), cuCmulf(h[1], cuConjf(h[3])));
        cuComplex g10 = cuConjf(g01);
        cuComplex g11 = cuCaddf(
            cuCaddf(cuCmulf(h[2], cuConjf(h[2])), cuCmulf(h[3], cuConjf(h[3]))),
            make_cuComplex(noise_var, 0.0f)
        );
        
        // Invert Gram matrix
        __shared__ cuComplex g_matrix[256 * 4];
        cuComplex* g_local = &g_matrix[threadIdx.x * 4];
        g_local[0] = g00; g_local[1] = g01;
        g_local[2] = g10; g_local[3] = g11;
        
        cuComplex g_inv[4];
        matrix_2x2_inverse(g_local, g_inv, 0);
        
        // Calculate MMSE filter: W = H^H * G^(-1)
        cuComplex w00 = cuCaddf(cuCmulf(cuConjf(h[0]), g_inv[0]), cuCmulf(cuConjf(h[2]), g_inv[2]));
        cuComplex w01 = cuCaddf(cuCmulf(cuConjf(h[0]), g_inv[1]), cuCmulf(cuConjf(h[2]), g_inv[3]));
        cuComplex w10 = cuCaddf(cuCmulf(cuConjf(h[1]), g_inv[0]), cuCmulf(cuConjf(h[3]), g_inv[2]));
        cuComplex w11 = cuCaddf(cuCmulf(cuConjf(h[1]), g_inv[1]), cuCmulf(cuConjf(h[3]), g_inv[3]));
        
        // MMSE detection: x = W * y
        int rx_offset = sym_idx * num_subcarriers * num_rx + sc_idx * num_rx;
        cuComplex y0 = desc->received_signal[rx_offset + 0];
        cuComplex y1 = desc->received_signal[rx_offset + 1];
        
        cuComplex x0 = cuCaddf(cuCmulf(w00, y0), cuCmulf(w01, y1));
        cuComplex x1 = cuCaddf(cuCmulf(w10, y0), cuCmulf(w11, y1));
        
        // Store detected symbols
        int out_offset = sym_idx * num_subcarriers * num_tx + sc_idx * num_tx;
        desc->detected_symbols[out_offset + 0] = x0;
        desc->detected_symbols[out_offset + 1] = x1;
    }
}

__global__ void mimo_ml_detection_kernel(MIMODetectionDescriptor* desc) {
    // Simplified ML detection for demonstration
    // In practice, this would implement sphere decoding or other efficient ML algorithms
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_subcarriers = desc->params->num_subcarriers;
    int num_symbols = desc->params->num_symbols;
    
    if (tid >= num_subcarriers * num_symbols) {
        return;
    }
    
    // For now, fall back to ZF detection
    // TODO: Implement proper ML detection with constellation search
    mimo_zf_detection_kernel<<<gridDim, blockDim>>>(desc);
}

namespace aerial::examples {

MIMODetector::MIMODetector(
    const std::string& module_id,
    const MIMOParams& params
) : module_id_(module_id), params_(params), d_descriptor_(nullptr),
    d_work_buffer_(nullptr), d_channel_inv_(nullptr), d_channel_gram_(nullptr) {
    
    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    
    allocate_gpu_memory();
}

MIMODetector::~MIMODetector() {
    deallocate_gpu_memory();
    
    if (cublas_handle_) {
        cublasDestroy(cublas_handle_);
    }
}

void MIMODetector::allocate_gpu_memory() {
    // Allocate descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(MIMODetectionDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate MIMO descriptor");
    }
    
    // Allocate working buffers
    size_t matrix_size = params_.num_tx_antennas * params_.num_rx_antennas * 
                        params_.num_subcarriers * params_.num_symbols;
    
    err = cudaMalloc(&d_work_buffer_, matrix_size * sizeof(cuComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate work buffer");
    }
    
    err = cudaMalloc(&d_channel_inv_, matrix_size * sizeof(cuComplex));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate channel inverse buffer");
    }
    
    err = cudaMalloc(&d_channel_gram_, matrix_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate channel Gram buffer");
    }
}

void MIMODetector::deallocate_gpu_memory() {
    if (d_descriptor_) { cudaFree(d_descriptor_); d_descriptor_ = nullptr; }
    if (d_work_buffer_) { cudaFree(d_work_buffer_); d_work_buffer_ = nullptr; }
    if (d_channel_inv_) { cudaFree(d_channel_inv_); d_channel_inv_ = nullptr; }
    if (d_channel_gram_) { cudaFree(d_channel_gram_); d_channel_gram_ = nullptr; }
}

cudaError_t MIMODetector::setup_detection_kernel(cudaStream_t stream) {
    // Copy descriptor to device
    return cudaMemcpyAsync(
        d_descriptor_, &h_descriptor_,
        sizeof(MIMODetectionDescriptor),
        cudaMemcpyHostToDevice, stream
    );
}

cudaError_t MIMODetector::launch_detection_kernel(cudaStream_t stream) {
    int total_elements = params_.num_subcarriers * params_.num_symbols;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    switch (params_.algorithm) {
        case MIMOAlgorithm::ZERO_FORCING:
            mimo_zf_detection_kernel<<<blocks, threads_per_block, 0, stream>>>(d_descriptor_);
            break;
        case MIMOAlgorithm::MMSE:
            mimo_mmse_detection_kernel<<<blocks, threads_per_block, 0, stream>>>(d_descriptor_);
            break;
        case MIMOAlgorithm::MAXIMUM_LIKELIHOOD:
            mimo_ml_detection_kernel<<<blocks, threads_per_block, 0, stream>>>(d_descriptor_);
            break;
        default:
            return cudaErrorInvalidValue;
    }
    
    return cudaGetLastError();
}

task::TaskResult MIMODetector::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token
) {
    if (token.is_cancellation_requested()) {
        return task::TaskResult(task::TaskStatus::Cancelled, "Task cancelled");
    }
    
    try {
        if (inputs.size() < 2 || outputs.empty()) {
            return task::TaskResult(task::TaskStatus::Failed, "Invalid inputs/outputs");
        }
        
        // Setup descriptor
        h_descriptor_.received_signal = reinterpret_cast<const cuComplex*>(inputs[0].data());
        h_descriptor_.channel_matrix = reinterpret_cast<const cuComplex*>(inputs[1].data());
        h_descriptor_.detected_symbols = reinterpret_cast<cuComplex*>(outputs[0].data());
        h_descriptor_.llr_output = (outputs.size() > 1) ? 
            reinterpret_cast<float*>(outputs[1].data()) : nullptr;
        h_descriptor_.params = &params_;
        h_descriptor_.total_resource_elements = params_.num_subcarriers * params_.num_symbols;
        
        cudaStream_t stream = 0; // Use default stream
        
        // Setup and launch kernel
        cudaError_t err = setup_detection_kernel(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Setup failed");
        }
        
        err = launch_detection_kernel(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Kernel launch failed");
        }
        
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            return task::TaskResult(task::TaskStatus::Failed, "Synchronization failed");
        }
        
        return task::TaskResult(task::TaskStatus::Completed, "MIMO detection completed");
        
    } catch (const std::exception& e) {
        return task::TaskResult(task::TaskStatus::Failed, e.what());
    }
}

bool MIMODetector::is_input_ready(std::size_t input_index) const {
    return input_index < 2; // received_signal and channel_matrix
}

bool MIMODetector::is_output_ready(std::size_t output_index) const {
    return input_index < 2; // detected_symbols and optional LLRs
}

} // namespace aerial::examples