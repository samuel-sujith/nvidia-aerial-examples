/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_estimator.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cmath>
#include <stdexcept>

namespace framework {
namespace examples {

// CUDA kernel implementation
__device__ cuComplex least_squares_estimate(
    const cuComplex* rx_pilots, 
    const cuComplex* tx_pilots,
    int pilot_idx
) {
    cuComplex rx = rx_pilots[pilot_idx];
    cuComplex tx = tx_pilots[pilot_idx];
    
    // H = Y / X (element-wise division)
    // For complex division: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c²+d²)
    float denominator = tx.x * tx.x + tx.y * tx.y;
    if (denominator < 1e-10f) {
        return make_cuComplex(0.0f, 0.0f);
    }
    
    cuComplex estimate;
    estimate.x = (rx.x * tx.x + rx.y * tx.y) / denominator;
    estimate.y = (rx.y * tx.x - rx.x * tx.y) / denominator;
    
    return estimate;
}

__device__ cuComplex mmse_estimate(
    const cuComplex* rx_pilots,
    const cuComplex* tx_pilots, 
    float noise_variance,
    int pilot_idx
) {
    // MMSE: H = (H_LS * |X|²) / (|X|² + σ²)
    cuComplex h_ls = least_squares_estimate(rx_pilots, tx_pilots, pilot_idx);
    cuComplex tx = tx_pilots[pilot_idx];
    
    float tx_power = tx.x * tx.x + tx.y * tx.y;
    float mmse_factor = tx_power / (tx_power + noise_variance);
    
    cuComplex estimate;
    estimate.x = h_ls.x * mmse_factor;
    estimate.y = h_ls.y * mmse_factor;
    
    return estimate;
}

__device__ cuComplex linear_interpolate(
    const cuComplex* channel_pilots,
    int data_subcarrier_idx,
    int pilot_spacing
) {
    // Find surrounding pilot positions
    int pilot_before = (data_subcarrier_idx / pilot_spacing) * pilot_spacing;
    int pilot_after = pilot_before + pilot_spacing;
    
    // Linear interpolation weight
    float alpha = (float)(data_subcarrier_idx - pilot_before) / (float)pilot_spacing;
    
    cuComplex h_before = channel_pilots[pilot_before / pilot_spacing];
    cuComplex h_after = channel_pilots[pilot_after / pilot_spacing];
    
    // Linear interpolation: H = (1-α)*H1 + α*H2
    cuComplex estimate;
    estimate.x = (1.0f - alpha) * h_before.x + alpha * h_after.x;
    estimate.y = (1.0f - alpha) * h_before.y + alpha * h_after.y;
    
    return estimate;
}

__global__ void channel_estimation_kernel(ChannelEstDescriptor* desc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_subcarriers = desc->params->num_resource_blocks * 12; // 12 subcarriers per RB
    
    if (tid >= total_subcarriers) {
        return;
    }
    
    int symbol_idx = blockIdx.y;
    if (symbol_idx >= desc->params->num_ofdm_symbols) {
        return;
    }
    
    int subcarrier_idx = tid;
    int output_idx = symbol_idx * total_subcarriers + subcarrier_idx;
    
    // Check if this is a pilot subcarrier
    bool is_pilot = (subcarrier_idx % desc->params->pilot_spacing) == 0;
    
    cuComplex estimate;
    if (is_pilot) {
        // Direct estimation from pilot
        int pilot_idx = subcarrier_idx / desc->params->pilot_spacing;
        
        switch (desc->params->algorithm) {
            case ChannelEstAlgorithm::LEAST_SQUARES:
                estimate = least_squares_estimate(
                    desc->rx_pilots, desc->tx_pilots, pilot_idx
                );
                break;
                
            case ChannelEstAlgorithm::MMSE:
                estimate = mmse_estimate(
                    desc->rx_pilots, desc->tx_pilots, 
                    desc->params->noise_variance, pilot_idx
                );
                break;
                
            default:
                estimate = least_squares_estimate(
                    desc->rx_pilots, desc->tx_pilots, pilot_idx
                );
                break;
        }
    } else {
        // Interpolation for data subcarriers
        // First, we need pilot estimates for this symbol (stored in shared memory)
        __shared__ cuComplex pilot_estimates[256]; // Assuming max 256 pilots per symbol
        
        int pilots_per_symbol = total_subcarriers / desc->params->pilot_spacing;
        
        // Cooperatively load pilot estimates
        if (threadIdx.x < pilots_per_symbol) {
            int pilot_sc = threadIdx.x * desc->params->pilot_spacing;
            int pilot_idx = threadIdx.x;
            
            switch (desc->params->algorithm) {
                case ChannelEstAlgorithm::LEAST_SQUARES:
                    pilot_estimates[threadIdx.x] = least_squares_estimate(
                        desc->rx_pilots, desc->tx_pilots, pilot_idx
                    );
                    break;
                    
                case ChannelEstAlgorithm::MMSE:
                    pilot_estimates[threadIdx.x] = mmse_estimate(
                        desc->rx_pilots, desc->tx_pilots,
                        desc->params->noise_variance, pilot_idx
                    );
                    break;
                    
                default:
                    pilot_estimates[threadIdx.x] = least_squares_estimate(
                        desc->rx_pilots, desc->tx_pilots, pilot_idx
                    );
                    break;
            }
        }
        __syncthreads();
        
        // Interpolate for data subcarrier
        estimate = linear_interpolate(
            pilot_estimates, subcarrier_idx, desc->params->pilot_spacing
        );
    }
    
    // Apply scaling factor
    estimate.x *= desc->params->beta_scaling;
    estimate.y *= desc->params->beta_scaling;
    
    // Store result
    desc->channel_estimates[output_idx] = estimate;
}

} // namespace examples  
} // namespace framework

// Class member function implementations
namespace framework {
namespace examples {

ChannelEstimator::ChannelEstimator(
    const std::string& module_id,
    const ChannelEstParams& params
) : module_id_(module_id), params_(params), d_descriptor_(nullptr) {
    allocate_gpu_memory();
    configure_kernel_launch();
}

ChannelEstimator::~ChannelEstimator() {
    deallocate_gpu_memory();
}

void ChannelEstimator::allocate_gpu_memory() {
    // Allocate device descriptor
    cudaError_t err = cudaMalloc(&d_descriptor_, sizeof(ChannelEstDescriptor));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for channel estimator descriptor");
    }
    
    // Initialize launch config
    launch_config_ = std::make_unique<ChannelEstLaunchConfig>();
    launch_config_->descriptor = d_descriptor_;
    launch_config_->kernel_args[0] = &(launch_config_->descriptor);
}

void ChannelEstimator::deallocate_gpu_memory() {
    if (d_descriptor_) {
        cudaFree(d_descriptor_);
        d_descriptor_ = nullptr;
    }
}

cudaError_t ChannelEstimator::configure_kernel_launch() {
    // Configure kernel launch parameters
    int total_subcarriers = params_.num_resource_blocks * 12;
    int threads_per_block = 256;
    int blocks_x = (total_subcarriers + threads_per_block - 1) / threads_per_block;
    int blocks_y = params_.num_ofdm_symbols;
    
    launch_config_->kernel_params.blockDim.x = threads_per_block;
    launch_config_->kernel_params.blockDim.y = 1;
    launch_config_->kernel_params.blockDim.z = 1;
    launch_config_->kernel_params.gridDim.x = blocks_x;
    launch_config_->kernel_params.gridDim.y = blocks_y;
    launch_config_->kernel_params.gridDim.z = 1;
    launch_config_->kernel_params.sharedMemBytes = 256 * sizeof(cuComplex); // For pilot estimates
    launch_config_->kernel_params.kernelParams = launch_config_->kernel_args;
    launch_config_->kernel_params.extra = nullptr;
    
    // Get kernel function
    cudaError_t err = cudaGetFuncBySymbol(
        &(launch_config_->kernel_params.func),
        reinterpret_cast<void*>(channel_estimation_kernel)
    );
    
    return err;
}

cudaError_t ChannelEstimator::setup_channel_estimation(
    const cuComplex* rx_pilots,
    const cuComplex* tx_pilots,
    cuComplex* channel_estimates,
    cudaStream_t stream
) {
    // Setup host descriptor
    h_descriptor_.rx_pilots = rx_pilots;
    h_descriptor_.tx_pilots = tx_pilots;
    h_descriptor_.channel_estimates = channel_estimates;
    h_descriptor_.params = &params_;
    h_descriptor_.num_pilots = (params_.num_resource_blocks * 12) / params_.pilot_spacing;
    h_descriptor_.num_data_subcarriers = params_.num_resource_blocks * 12;
    
    // Copy descriptor to device
    return cudaMemcpyAsync(
        d_descriptor_, &h_descriptor_, 
        sizeof(ChannelEstDescriptor),
        cudaMemcpyHostToDevice, stream
    );
}

cudaError_t ChannelEstimator::launch_channel_estimation(cudaStream_t stream) {
    if (!launch_config_) {
        return cudaErrorNotReady;
    }

    // Use simple kernel launch instead of complex API
    dim3 gridDim(launch_config_->kernel_params.gridDim.x, 
                 launch_config_->kernel_params.gridDim.y,
                 launch_config_->kernel_params.gridDim.z);
    dim3 blockDim(launch_config_->kernel_params.blockDim.x,
                  launch_config_->kernel_params.blockDim.y, 
                  launch_config_->kernel_params.blockDim.z);
    
    channel_estimation_kernel<<<gridDim, blockDim, 0, stream>>>(launch_config_->descriptor);
    
    return cudaGetLastError();
}

task::TaskResult ChannelEstimator::execute(
    const std::vector<tensor::TensorInfo>& inputs,
    std::vector<tensor::TensorInfo>& outputs,
    const task::CancellationToken& token
) {
    // Note: Simplified cancellation check - actual interface may differ
    // if (token.is_cancelled()) {
    //     return task::TaskResult(task::TaskStatus::Cancelled, "Task cancelled before execution");
    // }
    
    try {
        // Validate inputs
        if (inputs.size() < 2) {
            return task::TaskResult(task::TaskStatus::Failed, "Insufficient inputs: need rx_pilots and tx_pilots");
        }
        
        // Get input tensors
        const auto& rx_pilots_tensor = inputs[0];
        const auto& tx_pilots_tensor = inputs[1];
        
        // Setup output tensor
        if (outputs.empty()) {
            return task::TaskResult(task::TaskStatus::Failed, "No output tensor provided");
        }
        
        auto& channel_est_tensor = outputs[0];
        
        // Get CUDA stream from tensor context (framework specific)
        cudaStream_t stream = 0; // In real implementation, extract from tensor context
        
        // Setup kernel (simplified - actual tensor data access would be different)
        cudaError_t err = this->setup_channel_estimation(
            nullptr, // reinterpret_cast<const cuComplex*>(rx_pilots_tensor.get_data_ptr()),
            nullptr, // reinterpret_cast<const cuComplex*>(tx_pilots_tensor.get_data_ptr()),
            nullptr, // reinterpret_cast<cuComplex*>(channel_est_tensor.get_data_ptr()),
            stream
        );
        
        if (err != cudaSuccess) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                std::string("Setup failed: ") + cudaGetErrorString(err)
            );
        }
        
        // Launch kernel
        err = this->launch_channel_estimation(stream);
        
        if (err != cudaSuccess) {
            return task::TaskResult(
                task::TaskStatus::Failed, 
                std::string("Kernel launch failed: ") + cudaGetErrorString(err)
            );
        }
        
        // Synchronize stream
        err = cudaStreamSynchronize(stream);
        
        if (err != cudaSuccess) {
            return task::TaskResult(
                task::TaskStatus::Failed,
                std::string("Synchronization failed: ") + cudaGetErrorString(err)
            );
        }
        
        return task::TaskResult(task::TaskStatus::Completed, "Channel estimation completed successfully");
        
    } catch (const std::exception& e) {
        return task::TaskResult(
            task::TaskStatus::Failed, 
            std::string("Exception during execution: ") + e.what()
        );
    }
}

bool ChannelEstimator::is_input_ready(std::size_t input_index) const {
    // In real implementation, check if input tensor is available
    return input_index < 2; // rx_pilots and tx_pilots
}

bool ChannelEstimator::is_output_ready(std::size_t output_index) const {
    // In real implementation, check if output buffer is allocated
    return output_index == 0; // channel_estimates
}

} // namespace examples
} // namespace framework