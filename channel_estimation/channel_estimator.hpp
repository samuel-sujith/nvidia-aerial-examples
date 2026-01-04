/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FRAMEWORK_EXAMPLES_CHANNEL_ESTIMATOR_HPP
#define FRAMEWORK_EXAMPLES_CHANNEL_ESTIMATOR_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <memory>
#include <string>

#include <task/task.hpp>
#include "tensor/tensor_info.hpp"
#include "pipeline/imodule.hpp"

namespace framework::examples {

/// Channel estimation algorithm types
enum class ChannelEstAlgorithm {
    LEAST_SQUARES,      ///< Least squares estimation
    MMSE,              ///< Minimum mean square error
    LINEAR_INTERPOLATION ///< Linear interpolation between pilots
};

/// Channel estimation parameters
struct ChannelEstParams {
    ChannelEstAlgorithm algorithm{ChannelEstAlgorithm::LEAST_SQUARES};
    int num_rx_antennas{1};        ///< Number of receive antennas
    int num_tx_layers{1};          ///< Number of transmit layers
    int num_resource_blocks{273};   ///< Number of resource blocks
    int num_ofdm_symbols{14};      ///< Number of OFDM symbols per slot
    int pilot_spacing{4};          ///< Pilot symbol spacing in frequency
    float noise_variance{0.1f};    ///< Estimated noise variance for MMSE
    float beta_scaling{1.0f};      ///< Channel scaling factor
};

/// GPU kernel descriptor for channel estimation
struct ChannelEstDescriptor {
    const cuComplex* rx_pilots;       ///< Received pilot symbols [num_pilots]
    const cuComplex* tx_pilots;       ///< Known transmitted pilots [num_pilots]
    cuComplex* channel_estimates;     ///< Output channel estimates [num_subcarriers * num_symbols]
    ChannelEstParams* params;         ///< Estimation parameters
    int num_pilots;                   ///< Total number of pilot symbols
    int num_data_subcarriers;         ///< Number of data subcarriers to interpolate
};

/// GPU launch configuration for channel estimation
struct ChannelEstLaunchConfig {
    cudaKernelNodeParams kernel_params;
    ChannelEstDescriptor* descriptor;
    void* kernel_args[1];
};

/// Channel estimator class implementing the framework task pattern
class ChannelEstimator final : public pipeline::IModule {
public:
    /**
     * Constructor
     * @param module_id Unique identifier for this module
     * @param params Channel estimation parameters
     */
    explicit ChannelEstimator(
        const std::string& module_id,
        const ChannelEstParams& params
    );
    
    ~ChannelEstimator() override;

    // IModule interface implementation  
    std::string_view get_module_id() const override { return module_id_; }
    
    task::TaskResult execute(
        const std::vector<tensor::TensorInfo>& inputs,
        std::vector<tensor::TensorInfo>& outputs,
        const task::CancellationToken& token
    ) override;

    bool is_input_ready(std::size_t input_index) const;
    bool is_output_ready(std::size_t output_index) const;

    /// Setup GPU kernel for channel estimation
    cudaError_t setup_channel_estimation(
        const cuComplex* rx_pilots,
        const cuComplex* tx_pilots,
        cuComplex* channel_estimates,
        cudaStream_t stream
    );

    /// Launch channel estimation kernel
    cudaError_t launch_channel_estimation(cudaStream_t stream);

private:
    std::string module_id_;
    ChannelEstParams params_;
    
    // GPU memory management
    std::unique_ptr<ChannelEstLaunchConfig> launch_config_;
    ChannelEstDescriptor* d_descriptor_;  ///< Device descriptor
    ChannelEstDescriptor h_descriptor_;   ///< Host descriptor
    
    // Internal methods
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    cudaError_t configure_kernel_launch();
};

/// GPU kernel functions (implemented in .cu file)  
namespace framework {
namespace examples {

/// Main channel estimation kernel
__global__ void channel_estimation_kernel(ChannelEstDescriptor* desc);

/// Helper device functions
__device__ cuComplex least_squares_estimate(
    const cuComplex* rx_pilots, 
    const cuComplex* tx_pilots,
    int pilot_idx
);

__device__ cuComplex mmse_estimate(
    const cuComplex* rx_pilots,
    const cuComplex* tx_pilots, 
    float noise_variance,
    int pilot_idx
);

__device__ cuComplex linear_interpolate(
    const cuComplex* channel_pilots,
    int data_subcarrier_idx,
    int pilot_spacing
);

} // namespace examples
} // namespace framework

} // namespace framework::examples

#endif // FRAMEWORK_EXAMPLES_CHANNEL_ESTIMATOR_HPP