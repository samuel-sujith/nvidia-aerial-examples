// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "channel_estimation_module.hpp"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>  // std::min
#include <cstdio>
#include <cstdlib>

namespace channel_estimation {

namespace {
bool debug_enabled() {
    const char* value = std::getenv("AERIAL_DEBUG");
    return value && value[0] != '0';
}
} // namespace

/// CUDA kernel for least squares channel estimation (SAFE)
__global__ void ls_channel_estimation_kernel(ChannelEstDescriptor* desc) {
    if (!desc || !desc->params || !desc->rx_pilots || !desc->tx_pilots || !desc->pilot_estimates) {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_pilots) return;
    
    const cuComplex rx_pilot = desc->rx_pilots[tid];
    const cuComplex tx_pilot = desc->tx_pilots[tid];
    
    // Safe division
    float tx_real = cuCrealf(tx_pilot);
    float tx_imag = cuCimagf(tx_pilot);
    if (tx_real == 0.0f && tx_imag == 0.0f) {
        desc->pilot_estimates[tid] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    
    cuComplex channel_est = cuCdivf(rx_pilot, tx_pilot);
    channel_est = make_cuComplex(
        cuCrealf(channel_est) * desc->params->beta_scaling,
        cuCimagf(channel_est) * desc->params->beta_scaling
    );
    desc->pilot_estimates[tid] = channel_est;
}

/// CUDA kernel for linear interpolation (SAFE)
__global__ void interpolate_channel_estimates_kernel(ChannelEstDescriptor* desc) {
    if (!desc || !desc->params || !desc->pilot_estimates || !desc->channel_estimates) {
        return;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= desc->num_data_subcarriers) return;
    
    int pilot_spacing = desc->params->pilot_spacing;
    if (pilot_spacing <= 0) return;  // Safety
    
    int pilot_idx = tid % (desc->num_pilots * pilot_spacing);
    int pilot_before = (pilot_idx / pilot_spacing) * pilot_spacing;
    int pilot_after = pilot_before + pilot_spacing;
    
    // Clamp safely
    pilot_before = min(pilot_before, desc->num_pilots - 1);
    pilot_after = min(pilot_after, desc->num_pilots - 1);
    
    if (pilot_before < 0 || pilot_after < 0) {
        desc->channel_estimates[tid] = make_cuComplex(0.0f, 0.0f);
        return;
    }
    
    if (pilot_before == pilot_after) {
        desc->channel_estimates[tid] = desc->pilot_estimates[pilot_before];
    } else {
        float alpha = static_cast<float>(pilot_idx - pilot_before) / static_cast<float>(pilot_spacing);
        cuComplex h_before = desc->pilot_estimates[pilot_before];
        cuComplex h_after = desc->pilot_estimates[pilot_after];
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
) : module_id_(module_id), params_(params), d_descriptor_(nullptr), d_params_(nullptr) {
    setup_port_info();
}

ChannelEstimator::~ChannelEstimator() {
    deallocate_gpu_memory();
}

void ChannelEstimator::setup_port_info() {
    using namespace framework::tensor;
    
    input_ports_.resize(2);
    
    input_ports_[0].name = "rx_pilots";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    input_ports_[1].name = "tx_pilots";
    input_ports_[1].tensors.resize(1);
    input_ports_[1].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{static_cast<std::size_t>(params_.num_resource_blocks * 12 / params_.pilot_spacing)}
    );
    
    output_ports_.resize(1);
    
    output_ports_[0].name = "channel_estimates";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = TensorInfo(
        NvDataType::TensorC32F,
        std::vector<std::size_t>{
            static_cast<std::size_t>(params_.num_resource_blocks * 12),
            static_cast<std::size_t>(params_.num_ofdm_symbols)
        }
    );
}

void ChannelEstimator::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    if (!mem_slice_.device_tensor_ptr) {
        throw std::runtime_error("Channel estimation output device buffer not allocated");
    }
    d_channel_estimates_ = reinterpret_cast<cuComplex*>(mem_slice_.device_tensor_ptr);
    allocate_gpu_memory();
    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<ChannelEstDescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<ChannelEstDescriptor>(0);
    if (!dynamic_params_gpu_ptr_) {
        throw std::runtime_error("Channel estimation dynamic descriptor device pointer not allocated");
    }
    d_descriptor_ = dynamic_params_gpu_ptr_;

    int num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    dim3 blockSize(256);
    dim3 gridSize((num_pilots + blockSize.x - 1) / blockSize.x);
    ls_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(ls_channel_estimation_kernel));
    ls_kernel_config_.setup_kernel_dimensions(gridSize, blockSize);
    framework::pipeline::setup_kernel_arguments(ls_kernel_config_, dynamic_params_gpu_ptr_);

    int num_data = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    dim3 gridInterp((num_data + blockSize.x - 1) / blockSize.x);
    interp_kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(interpolate_channel_estimates_kernel));
    interp_kernel_config_.setup_kernel_dimensions(gridInterp, blockSize);
    framework::pipeline::setup_kernel_arguments(interp_kernel_config_, dynamic_params_gpu_ptr_);
}

void ChannelEstimator::warmup(cudaStream_t /*stream*/) {
    // No warmup needed
}

void ChannelEstimator::configure_io(
    const framework::pipeline::DynamicParams& /*params*/,
    cudaStream_t stream
) {
    if (!dynamic_params_cpu_ptr_) {
        throw std::runtime_error("Channel estimation dynamic descriptor not initialized");
    }
    if (!kernel_desc_mgr_) {
        throw std::runtime_error("Channel estimation kernel descriptor manager not initialized");
    }
    if (!current_rx_pilots_ || !current_tx_pilots_) {
        throw std::runtime_error("Channel estimation input pilots not set");
    }
    if (!d_channel_estimates_ || !d_pilot_estimates_ || !d_params_) {
        throw std::runtime_error("Channel estimation device buffers not initialized");
    }
    dynamic_params_cpu_ptr_->rx_pilots = current_rx_pilots_;
    dynamic_params_cpu_ptr_->tx_pilots = current_tx_pilots_;
    dynamic_params_cpu_ptr_->channel_estimates = d_channel_estimates_;
    dynamic_params_cpu_ptr_->params = d_params_;
    dynamic_params_cpu_ptr_->num_pilots = params_.num_resource_blocks * 12 / params_.pilot_spacing;
    dynamic_params_cpu_ptr_->num_data_subcarriers = params_.num_resource_blocks * 12 * params_.num_ofdm_symbols;
    dynamic_params_cpu_ptr_->pilot_estimates = d_pilot_estimates_;
    kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);

    if (debug_enabled()) {
        std::cerr << "[DEBUG] ChannelEstimator::configure_io rx=" << current_rx_pilots_
                  << " tx=" << current_tx_pilots_
                  << " out=" << d_channel_estimates_
                  << " pilot_est=" << d_pilot_estimates_
                  << " params=" << d_params_
                  << " dyn_cpu=" << dynamic_params_cpu_ptr_
                  << " dyn_gpu=" << dynamic_params_gpu_ptr_
                  << std::endl;
    }
}

void ChannelEstimator::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    for (const auto& port : inputs) {
        if (port.name == "rx_pilots" && !port.tensors.empty()) {
            current_rx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        } else if (port.name == "tx_pilots" && !port.tensors.empty()) {
            current_tx_pilots_ = static_cast<const cuComplex*>(port.tensors[0].device_ptr);
        }
    }
}

std::vector<framework::pipeline::PortInfo> ChannelEstimator::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_channel_estimates_;
    }
    return outputs;
}

void ChannelEstimator::execute(cudaStream_t stream) {
    // Validate pointers
    if (!current_rx_pilots_ || !current_tx_pilots_ || !d_channel_estimates_) {
        throw std::runtime_error("Missing input/output pointers");
    }
    
    // Zero output
    size_t out_size = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    cudaError_t err = cudaMemsetAsync(d_channel_estimates_, 0, out_size, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(err));
    }
    
    // Configure & copy descriptor
    configure_io({}, stream);
    
    // Launch kernels
    err = launch_channel_estimation_kernel(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    if (debug_enabled()) {
        err = cudaPeekAtLastError();
        std::cerr << "[DEBUG] ChannelEstimator::execute cudaPeekAtLastError="
                  << cudaGetErrorString(err) << std::endl;
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Stream sync failed: ") + cudaGetErrorString(err));
        }
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Stream sync failed: ") + cudaGetErrorString(err));
    }
}

void ChannelEstimator::allocate_gpu_memory() {
    cudaError_t err;
    
    // FIXED: Device params copy
    err = cudaMalloc(&d_params_, sizeof(ChannelEstParams));
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_params_ failed");
    err = cudaMemcpy(d_params_, &params_, sizeof(ChannelEstParams), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error("cudaMemcpy d_params_ failed");
    
    // Buffers
    size_t pilot_size = params_.num_resource_blocks * 12ULL / params_.pilot_spacing * sizeof(cuComplex);
    size_t estimates_size = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    (void)estimates_size;
    
    err = cudaMalloc(&d_pilot_symbols_, pilot_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_pilot_symbols_ failed");
    
    err = cudaMalloc(&d_pilot_estimates_, pilot_size);
    if (err != cudaSuccess) throw std::runtime_error("cudaMalloc d_pilot_estimates_ failed");
    cudaMemset(d_pilot_estimates_, 0, pilot_size);  // Zero init
    
    std::cout << "[DEBUG] Allocated: d_params_=" << d_params_ 
              << " d_pilot_estimates_=" << d_pilot_estimates_ << std::endl;
}

void ChannelEstimator::deallocate_gpu_memory() {
    // Individual frees to avoid type mismatch
    if (d_params_) { cudaFree(d_params_); d_params_ = nullptr; }
    d_descriptor_ = nullptr;
    if (d_pilot_symbols_) { cudaFree(d_pilot_symbols_); d_pilot_symbols_ = nullptr; }
    if (d_pilot_estimates_) { cudaFree(d_pilot_estimates_); d_pilot_estimates_ = nullptr; }
    d_channel_estimates_ = nullptr;
}

cudaError_t ChannelEstimator::launch_channel_estimation_kernel(cudaStream_t stream) {
    const CUresult ls_err = ls_kernel_config_.launch(stream);
    cudaError_t err = (ls_err == CUDA_SUCCESS) ? cudaGetLastError() : cudaErrorLaunchFailure;
    if (err != cudaSuccess) return err;
    
    const CUresult interp_err = interp_kernel_config_.launch(stream);
    err = (interp_err == CUDA_SUCCESS) ? cudaGetLastError() : cudaErrorLaunchFailure;
    
    return cudaGetLastError();
}

std::span<const CUgraphNode> ChannelEstimator::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    ls_node_ = graph->add_kernel_node(deps, ls_kernel_config_.get_kernel_params());
    interp_node_ = graph->add_kernel_node({&ls_node_, 1}, interp_kernel_config_.get_kernel_params());
    return {&interp_node_, 1};
}

void ChannelEstimator::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams& /*params*/) {
    if (!ls_node_ || !interp_node_) {
        throw std::runtime_error("Channel estimation graph nodes not initialized");
    }
    auto ls_params = ls_kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, ls_node_, &ls_params);
    auto interp_params = interp_kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, interp_node_, &interp_params);
}

framework::pipeline::ModuleMemoryRequirements ChannelEstimator::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements reqs{};
    
    size_t estimates_bytes = params_.num_resource_blocks * 12ULL * params_.num_ofdm_symbols * sizeof(cuComplex);
    reqs.device_tensor_bytes = estimates_bytes;
    reqs.dynamic_kernel_descriptor_bytes = sizeof(ChannelEstDescriptor);
    reqs.alignment = 256; // CUDA memory alignment
    
    return reqs;
}

framework::pipeline::OutputPortMemoryCharacteristics
ChannelEstimator::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics chars{};
    
    if (port_name == "channel_estimates") {
        chars.provides_fixed_address_for_zero_copy = true;
    }
    
    return chars;
}

std::vector<std::string> ChannelEstimator::get_input_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : input_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<std::string> ChannelEstimator::get_output_port_names() const {
    std::vector<std::string> names;
    for (const auto& port : output_ports_) {
        names.push_back(port.name);
    }
    return names;
}

std::vector<framework::tensor::TensorInfo> ChannelEstimator::get_input_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : input_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

std::vector<framework::tensor::TensorInfo> ChannelEstimator::get_output_tensor_info(std::string_view port_name) const {
    std::vector<framework::tensor::TensorInfo> tensor_infos;
    for (const auto& port : output_ports_) {
        if (port.name == port_name) {
            for (const auto& tensor : port.tensors) {
                tensor_infos.push_back(tensor.tensor_info);
            }
        }
    }
    return tensor_infos;
}

} // namespace channel_estimation