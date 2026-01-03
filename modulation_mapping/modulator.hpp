/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AERIAL_EXAMPLES_MODULATOR_HPP
#define AERIAL_EXAMPLES_MODULATOR_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <memory>
#include <string>
#include <vector>

#include "task/task.hpp"
#include "tensor/tensor_info.hpp"

namespace framework {
namespace examples {

/// QAM modulation schemes supported
enum class ModulationScheme {
    QPSK,      ///< Quadrature Phase Shift Keying
    QAM_16,    ///< 16-QAM
    QAM_64,    ///< 64-QAM  
    QAM_256    ///< 256-QAM
};

/// Modulation parameters
struct ModulationParams {
    ModulationScheme scheme{ModulationScheme::QPSK};
    float scaling_factor{1.0f};        ///< Symbol scaling factor
    int num_symbols{0};                ///< Total symbols to modulate
    bool normalize_power{true};        ///< Normalize average symbol power
};

/// GPU kernel descriptor for QAM modulation
struct ModulationDescriptor {
    const uint32_t* input_bits;        ///< Input bit stream
    cuComplex* output_symbols;         ///< Output modulated symbols
    ModulationParams* params;          ///< Modulation parameters
    int total_symbols;                 ///< Total number of symbols
};

/// QAM modulator module
class QAMModulator final {
public:
    explicit QAMModulator(
        const std::string& module_id,
        const ModulationParams& params
    );
    
    ~QAMModulator();

    // Module interface implementation  
    std::string_view get_module_id() const { return module_id_; }
    
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    );

private:
    std::string module_id_;
    ModulationParams params_;
    
    // GPU resources
    ModulationDescriptor* d_descriptor_;
    ModulationDescriptor h_descriptor_;
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    cudaError_t setup_modulation_kernel(cudaStream_t stream);
    cudaError_t launch_modulation_kernel(cudaStream_t stream);
};

/// CUDA kernel declarations
__global__ void qam_modulation_kernel(ModulationDescriptor* desc);

} // namespace examples
} // namespace framework

#endif // AERIAL_EXAMPLES_MODULATOR_HPP