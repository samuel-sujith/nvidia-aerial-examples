/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CHANNEL_ESTIMATION_MODULE_HPP
#define CHANNEL_ESTIMATION_MODULE_HPP

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <memory>
#include <string>
#include <vector>
#include <span>
#include <algorithm>  // for std::min

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include "tensor/data_types.hpp"
#include <span>

// Abstract base interface for channel estimators
class IChannelEstimator {
public:
    virtual ~IChannelEstimator() = default;
    virtual void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) = 0;
    virtual void warmup(cudaStream_t stream) = 0;
    virtual void configure_io(const framework::pipeline::DynamicParams& params, cudaStream_t stream) = 0;
    virtual std::vector<framework::tensor::TensorInfo> get_input_tensor_info(std::string_view port_name) const = 0;
    virtual std::vector<framework::tensor::TensorInfo> get_output_tensor_info(std::string_view port_name) const = 0;
    virtual std::vector<std::string> get_input_port_names() const = 0;
    virtual std::vector<std::string> get_output_port_names() const = 0;
    virtual framework::pipeline::ModuleMemoryRequirements get_requirements() const = 0;
    virtual framework::pipeline::OutputPortMemoryCharacteristics get_output_memory_characteristics(std::string_view port_name) const = 0;
    virtual void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) = 0;
    virtual std::vector<framework::pipeline::PortInfo> get_outputs() const = 0;
    virtual void execute(cudaStream_t stream) = 0;
};

namespace channel_estimation {

/// Channel estimation algorithm types
enum class ChannelEstAlgorithm {
    LEAST_SQUARES,      ///< Least squares estimation
    MMSE,               ///< Minimum mean square error  
    LINEAR_INTERPOLATION, ///< Linear interpolation between pilots
    ML_TENSORRT         ///< Machine learning-based estimation using TensorRT
};

/// Channel estimation parameters
struct ChannelEstParams {
    ChannelEstAlgorithm algorithm{ChannelEstAlgorithm::LEAST_SQUARES};
    int num_rx_antennas{1};         ///< Number of receive antennas
    int num_tx_layers{1};           ///< Number of transmit layers
    int num_resource_blocks{273};   ///< Number of resource blocks
    int num_ofdm_symbols{14};       ///< Number of OFDM symbols per slot
    int pilot_spacing{4};           ///< Pilot symbol spacing in frequency
    float noise_variance{0.1f};     ///< Estimated noise variance for MMSE
    float beta_scaling{1.0f};       ///< Channel scaling factor

    // ML-based estimation options
    std::string model_path;         ///< Path to TensorRT engine/model file
    int max_batch_size{32};         ///< Maximum batch size for inference
    bool use_fp16{true};            ///< Use FP16 precision for inference
    int ml_input_size{0};           ///< ML model input size
    int ml_output_size{0};          ///< ML model output size
};

/// GPU kernel descriptor for channel estimation
struct ChannelEstDescriptor {
    const cuComplex* rx_pilots;         ///< Received pilot symbols [num_pilots]
    const cuComplex* tx_pilots;         ///< Known transmitted pilots [num_pilots]
    cuComplex* channel_estimates;       ///< Output channel estimates [num_subcarriers * num_symbols]
    cuComplex* pilot_estimates;         ///< Dedicated buffer for pilot estimates
    const ChannelEstParams* params;     ///< FIXED: const device params pointer
    int num_pilots;                     ///< Total number of pilot symbols
    int num_data_subcarriers;           ///< Number of data subcarriers to interpolate
};

/// Channel estimator module implementing the framework interface
class ChannelEstimator final : public IChannelEstimator, public framework::pipeline::IModule, 
                              public framework::pipeline::IAllocationInfoProvider,
                              public framework::pipeline::IGraphNodeProvider,
                              public framework::pipeline::IStreamExecutor {
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
    [[nodiscard]] std::string_view get_type_id() const override { return "channel_estimator"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }
    
    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    void warmup(cudaStream_t stream) override;
    
    void configure_io(
        const framework::pipeline::DynamicParams& params,
        cudaStream_t stream
    ) override;
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;
    
    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;
    
    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;
    
    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;
    
    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;
    
    // IModule casting methods
    framework::pipeline::IGraphNodeProvider* as_graph_node_provider() override { return this; }
    framework::pipeline::IStreamExecutor* as_stream_executor() override { return this; }
    
    // IStreamExecutor interface  
    void execute(cudaStream_t stream) override;

    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
        gsl_lite::not_null<framework::pipeline::IGraph*> graph,
        std::span<const CUgraphNode> deps) override;
    void update_graph_node_params(CUgraphExec exec, const framework::pipeline::DynamicParams& params) override;

private:
    std::string module_id_;
    ChannelEstParams params_;
    framework::pipeline::ModuleMemorySlice mem_slice_;
    
    // Port information
    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;
    
    // GPU resources
    ChannelEstDescriptor* d_descriptor_;
    ChannelEstDescriptor h_descriptor_;
    
    // FIXED: Device copy of params (CRITICAL)
    ChannelEstParams* d_params_{nullptr};
    
    // Device memory pointers
    cuComplex* d_pilot_symbols_{nullptr};
    cuComplex* d_channel_estimates_{nullptr};
    cuComplex* d_pilot_estimates_{nullptr};
    
    // Current tensor pointers (set during set_inputs)
    const cuComplex* current_rx_pilots_{nullptr};
    const cuComplex* current_tx_pilots_{nullptr};
    cuComplex* current_channel_estimates_{nullptr};

    std::unique_ptr<framework::pipeline::KernelDescriptorAccessor> kernel_desc_mgr_;
    ChannelEstDescriptor* dynamic_params_cpu_ptr_{nullptr};
    ChannelEstDescriptor* dynamic_params_gpu_ptr_{nullptr};
    framework::pipeline::KernelLaunchConfig<1> ls_kernel_config_;
    framework::pipeline::KernelLaunchConfig<1> interp_kernel_config_;
    CUgraphNode ls_node_{nullptr};
    CUgraphNode interp_node_{nullptr};
    
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void setup_port_info();
    cudaError_t launch_channel_estimation_kernel(cudaStream_t stream);
};

} // namespace channel_estimation

#endif // CHANNEL_ESTIMATION_MODULE_HPP
