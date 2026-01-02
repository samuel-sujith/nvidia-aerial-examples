/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AERIAL_EXAMPLES_MIMO_DETECTOR_HPP
#define AERIAL_EXAMPLES_MIMO_DETECTOR_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <memory>
#include <string>

#include "task/task.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/imodule.hpp"

namespace aerial::examples {

/// MIMO detection algorithms
enum class MIMOAlgorithm {
    ZERO_FORCING,      ///< Zero Forcing (ZF) detection
    MMSE,              ///< Minimum Mean Square Error detection
    MAXIMUM_LIKELIHOOD ///< Maximum Likelihood detection (ML)
};

/// MIMO system configuration
struct MIMOParams {
    MIMOAlgorithm algorithm{MIMOAlgorithm::ZERO_FORCING};
    int num_tx_antennas{2};        ///< Number of transmit antennas
    int num_rx_antennas{2};        ///< Number of receive antennas
    int num_subcarriers{1200};     ///< Number of subcarriers (e.g., 100 RBs * 12)
    int num_symbols{14};           ///< Number of OFDM symbols
    float noise_variance{0.1f};    ///< Noise variance for MMSE
    int constellation_size{4};     ///< Constellation size (4=QPSK, 16=16QAM, etc.)
};

/// GPU kernel descriptor for MIMO detection
struct MIMODetectionDescriptor {
    const cuComplex* received_signal;     ///< Received signal [num_rx x num_subcarriers x num_symbols]
    const cuComplex* channel_matrix;      ///< Channel matrix [num_rx x num_tx x num_subcarriers x num_symbols] 
    cuComplex* detected_symbols;          ///< Output detected symbols [num_tx x num_subcarriers x num_symbols]
    float* llr_output;                    ///< Soft bit LLRs (for soft detection) [optional]
    MIMOParams* params;                   ///< Detection parameters
    int total_resource_elements;         ///< Total REs to process
};

/// MIMO detection module with multiple algorithms
class MIMODetector {
public:
    explicit MIMODetector(
        const std::string& module_id,
        const MIMOParams& params
    );
    
    ~MIMODetector() override;

    // IModule interface
    std::string_view get_module_id() const override { return module_id_; }
    
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    ) override;

    bool is_input_ready(std::size_t input_index) const override;
    bool is_output_ready(std::size_t output_index) const override;

private:
    std::string module_id_;
    MIMOParams params_;
    
    // GPU resources
    MIMODetectionDescriptor* d_descriptor_;
    MIMODetectionDescriptor h_descriptor_;
    cublasHandle_t cublas_handle_;
    
    // Working buffers for matrix operations
    cuComplex* d_work_buffer_;
    cuComplex* d_channel_inv_;
    float* d_channel_gram_;
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    cudaError_t setup_detection_kernel(cudaStream_t stream);
    cudaError_t launch_detection_kernel(cudaStream_t stream);
    
    // Algorithm-specific implementations
    cudaError_t zero_forcing_detection(cudaStream_t stream);
    cudaError_t mmse_detection(cudaStream_t stream);
    cudaError_t ml_detection(cudaStream_t stream);
};

/// CUDA kernel declarations
extern "C" {
    __global__ void mimo_zf_detection_kernel(MIMODetectionDescriptor* desc);
    __global__ void mimo_mmse_detection_kernel(MIMODetectionDescriptor* desc);
    __global__ void mimo_ml_detection_kernel(MIMODetectionDescriptor* desc);
}

} // namespace aerial::examples

#endif // AERIAL_EXAMPLES_MIMO_DETECTOR_HPP