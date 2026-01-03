/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AERIAL_EXAMPLES_FFT_MODULE_HPP
#define AERIAL_EXAMPLES_FFT_MODULE_HPP

#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <memory>
#include <string>

#include "task/task.hpp"
#include "tensor/tensor_info.hpp"
#include "pipeline/imodule.hpp"

namespace aerial::examples {

/// FFT operation types
enum class FFTType {
    FORWARD,    ///< Forward FFT (time to frequency domain)
    INVERSE     ///< Inverse FFT (frequency to time domain)
};

/// FFT processing parameters
struct FFTParams {
    FFTType type{FFTType::FORWARD};
    int fft_size{2048};            ///< FFT size (must be power of 2)
    int batch_size{1};             ///< Number of FFTs to process in parallel
    bool normalize{true};          ///< Apply normalization for inverse FFT
    float scale_factor{1.0f};      ///< Additional scaling factor
};

/// cuFFT-based FFT processing module
class FFTModule final : public pipeline::IModule {
public:
    explicit FFTModule(
        const std::string& module_id,
        const FFTParams& params
    );
    
    ~FFTModule() override;

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
    FFTParams params_;
    
    // cuFFT resources
    cufftHandle fft_plan_;
    cudaStream_t cuda_stream_;
    
    void setup_fft_plan();
    void cleanup_fft_resources();
    cudaError_t execute_fft(cuComplex* input, cuComplex* output);
};

} // namespace aerial::examples

#endif // AERIAL_EXAMPLES_FFT_MODULE_HPP