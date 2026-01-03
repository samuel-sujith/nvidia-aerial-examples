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
#include <span>

#include "task/task.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/imodule.hpp"
#include "pipeline/types.hpp"

using namespace aerial::examples;

namespace aerial::examples {

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
class QAMModulator final : public framework::pipeline::IModule {
public:
    explicit QAMModulator(
        const std::string& module_id,
        const ModulationParams& params
    );
    
    ~QAMModulator() override;

    // IModule interface
    [[nodiscard]] std::string_view get_type_id() const override { return "QAMModulator"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }
    
    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;
    
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;

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

} // namespace aerial::examples

#endif // AERIAL_EXAMPLES_MODULATOR_HPP