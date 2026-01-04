/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FRAMEWORK_EXAMPLES_FFT_MODULE_HPP
#define FRAMEWORK_EXAMPLES_FFT_MODULE_HPP

#include <cufft.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <memory>
#include <string>
#include <string_view>
#include <span>

#include "pipeline/imodule.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

namespace framework::examples {

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
        const std::string& instance_id,
        const FFTParams& params
    );
    
    ~FFTModule() override;

    // Non-copyable, non-movable
    FFTModule(const FFTModule&) = delete;
    FFTModule& operator=(const FFTModule&) = delete;
    FFTModule(FFTModule&&) = delete;
    FFTModule& operator=(FFTModule&&) = delete;

    // IModule interface
    [[nodiscard]] std::string_view get_type_id() const override { return "FFTModule"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return instance_id_; }

    void setup_memory(const pipeline::ModuleMemorySlice& memory_slice) override;
    
    pipeline::ModuleMemoryRequirements get_memory_requirements(
        const pipeline::DynamicParams& params
    ) const override;

    std::vector<tensor::TensorInfo> get_input_tensor_info(
        std::size_t port,
        const pipeline::DynamicParams& params
    ) const override;

    std::vector<tensor::TensorInfo> get_output_tensor_info(
        std::size_t port,
        const pipeline::DynamicParams& params
    ) const override;

    void configure_io(
        const pipeline::DynamicParams& params,
        std::span<const pipeline::PortInfo> inputs,
        std::span<pipeline::PortInfo> outputs,
        cudaStream_t stream
    ) override;

    void execute_stream(cudaStream_t stream) override;
    
    void warmup(cudaStream_t stream) override;

    std::size_t get_num_input_ports() const override { return 1; }
    std::size_t get_num_output_ports() const override { return 1; }

private:
    std::string instance_id_;
    FFTParams params_;
    
    // cuFFT resources
    cufftHandle fft_plan_;
    cudaStream_t cuda_stream_;
    
    // Memory management
    pipeline::ModuleMemorySlice memory_slice_;
    void* d_input_buffer_{nullptr};
    void* d_output_buffer_{nullptr};
    void* d_workspace_{nullptr};
    size_t workspace_size_{0};
    
    // State management
    bool is_setup_{false};
    bool is_warmed_up_{false};
    std::vector<pipeline::PortInfo> input_ports_;
    std::vector<pipeline::PortInfo> output_ports_;
    
    void setup_fft_plan();
    void cleanup_fft_resources();
    cudaError_t execute_fft(cuComplex* input, cuComplex* output, cudaStream_t stream);
};

} // namespace framework::examples

#endif // FRAMEWORK_EXAMPLES_FFT_MODULE_HPP