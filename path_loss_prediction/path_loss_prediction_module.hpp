/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <span>

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/igraph_node_provider.hpp"
#include "pipeline/igraph.hpp"
#include "pipeline/kernel_descriptor_accessor.hpp"
#include "pipeline/kernel_launch_config.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <fstream>
#endif

namespace path_loss_prediction {

enum class ModelType {
    TensorRT_MLP,
    XGBoostStyle
};

struct PathLossModelParams {
    int num_features = 8;
    int hidden_size = 16;
    int batch_size = 32;
    int num_trees = 8;
    ModelType model_type = ModelType::TensorRT_MLP;
    std::string model_path;
};

struct PathLossDescriptor {
    const float* features;
    float* outputs;
    const float* w1;
    const float* b1;
    const float* w2;
    const float* b2;
    const int* tree_feature_idx;
    const float* tree_thresholds;
    const float* tree_left_values;
    const float* tree_right_values;
    int num_features;
    int hidden_size;
    int num_trees;
    int batch_size;
    int model_type;
};

class PathLossPredictionModule final : public framework::pipeline::IModule,
                                      public framework::pipeline::IAllocationInfoProvider,
                                      public framework::pipeline::IGraphNodeProvider,
                                      public framework::pipeline::IStreamExecutor {
public:
    explicit PathLossPredictionModule(
        const std::string& module_id,
        const PathLossModelParams& params);

    ~PathLossPredictionModule() override;

    [[nodiscard]] std::string_view get_type_id() const override { return "path_loss_predictor"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }

    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    void warmup(cudaStream_t stream) override;

    void configure_io(
        const framework::pipeline::DynamicParams& params,
        cudaStream_t stream) override;

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

    framework::pipeline::IGraphNodeProvider* as_graph_node_provider() override { return this; }
    framework::pipeline::IStreamExecutor* as_stream_executor() override { return this; }

    void execute(cudaStream_t stream) override;

    [[nodiscard]] std::span<const CUgraphNode> add_node_to_graph(
        gsl_lite::not_null<framework::pipeline::IGraph*> graph,
        std::span<const CUgraphNode> deps) override;
    void update_graph_node_params(CUgraphExec exec, const framework::pipeline::DynamicParams& params) override;

private:
    void setup_port_info();
    void allocate_model_buffers();
    void deallocate_model_buffers();
    bool load_model_file(
        const std::string& path,
        ModelType& model_type,
        int& num_features,
        int& hidden_size,
        int& num_trees,
        std::vector<float>& w1,
        std::vector<float>& b1,
        std::vector<float>& w2,
        std::vector<float>& b2,
        std::vector<int>& feature_idx,
        std::vector<float>& thresholds,
        std::vector<float>& left_values,
        std::vector<float>& right_values) const;
    void run_fallback_mlp(cudaStream_t stream);
    void run_xgboost_style(cudaStream_t stream);
    bool run_tensorrt(cudaStream_t stream);

    std::string module_id_;
    PathLossModelParams params_;
    framework::pipeline::ModuleMemorySlice mem_slice_{};

    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;

    const void* current_features_{nullptr};
    float* d_output_{nullptr};

    float* d_mlp_w1_{nullptr};
    float* d_mlp_b1_{nullptr};
    float* d_mlp_w2_{nullptr};
    float* d_mlp_b2_{nullptr};

    int* d_tree_feature_idx_{nullptr};
    float* d_tree_threshold_{nullptr};
    float* d_tree_left_value_{nullptr};
    float* d_tree_right_value_{nullptr};

#ifdef TENSORRT_AVAILABLE
    class TrtLogger;
    nvinfer1::IRuntime* trt_runtime_{nullptr};
    nvinfer1::ICudaEngine* trt_engine_{nullptr};
    nvinfer1::IExecutionContext* trt_context_{nullptr};
#else
    void* trt_runtime_{nullptr};
    void* trt_engine_{nullptr};
    void* trt_context_{nullptr};
#endif
    float* d_trt_input_{nullptr};
    bool trt_enabled_{false};
    bool model_loaded_{false};

    void cleanup_tensorrt_resources();
    bool initialize_tensorrt_engine();
    bool load_engine_from_file(const std::string& engine_path);
    bool run_trt_inference(cudaStream_t stream);

    std::unique_ptr<framework::pipeline::KernelDescriptorAccessor> kernel_desc_mgr_;
    PathLossDescriptor* dynamic_params_cpu_ptr_{nullptr};
    PathLossDescriptor* dynamic_params_gpu_ptr_{nullptr};
    framework::pipeline::KernelLaunchConfig<1> kernel_config_;
    CUgraphNode kernel_node_{nullptr};
};

} // namespace path_loss_prediction
