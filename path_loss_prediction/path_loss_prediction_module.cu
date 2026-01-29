/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "path_loss_prediction_module.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>

class path_loss_prediction::PathLossPredictionModule::TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static path_loss_prediction::PathLossPredictionModule::TrtLogger gTrtLogger;
#endif

namespace path_loss_prediction {
namespace {

__global__ void path_loss_predict_kernel(const PathLossDescriptor* desc) {
    if (!desc || !desc->features || !desc->outputs) {
        return;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= desc->batch_size) {
        return;
    }

    const float* x = desc->features + idx * desc->num_features;
    if (desc->model_type == static_cast<int>(ModelType::XGBoostStyle)) {
        float sum = 0.0f;
        for (int t = 0; t < desc->num_trees; ++t) {
            int f = desc->tree_feature_idx[t];
            float val = x[f] < desc->tree_thresholds[t]
                ? desc->tree_left_values[t]
                : desc->tree_right_values[t];
            sum += val;
        }
        desc->outputs[idx] = sum;
        return;
    }

    float out = desc->b2[0];
    for (int h = 0; h < desc->hidden_size; ++h) {
        float acc = desc->b1[h];
        const float* w = desc->w1 + h * desc->num_features;
        for (int f = 0; f < desc->num_features; ++f) {
            acc += w[f] * x[f];
        }
        float relu = acc > 0.0f ? acc : 0.0f;
        out += desc->w2[h] * relu;
    }
    desc->outputs[idx] = out;
}

} // namespace

PathLossPredictionModule::PathLossPredictionModule(
    const std::string& module_id,
    const PathLossModelParams& params)
    : module_id_(module_id), params_(params) {
    setup_port_info();
#ifdef TENSORRT_AVAILABLE
    auto dot = params_.model_path.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? "" : params_.model_path.substr(dot);
    if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
        initialize_tensorrt_engine();
    }
#endif
}

PathLossPredictionModule::~PathLossPredictionModule() {
    deallocate_model_buffers();
    cleanup_tensorrt_resources();
}

void PathLossPredictionModule::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    d_output_ = reinterpret_cast<float*>(mem_slice_.device_tensor_ptr);
    allocate_model_buffers();

    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<PathLossDescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<PathLossDescriptor>(0);
    if (!dynamic_params_gpu_ptr_) {
        throw std::runtime_error("Path loss dynamic descriptor device pointer not allocated");
    }

    dynamic_params_cpu_ptr_->outputs = d_output_;
    dynamic_params_cpu_ptr_->w1 = d_mlp_w1_;
    dynamic_params_cpu_ptr_->b1 = d_mlp_b1_;
    dynamic_params_cpu_ptr_->w2 = d_mlp_w2_;
    dynamic_params_cpu_ptr_->b2 = d_mlp_b2_;
    dynamic_params_cpu_ptr_->tree_feature_idx = d_tree_feature_idx_;
    dynamic_params_cpu_ptr_->tree_thresholds = d_tree_threshold_;
    dynamic_params_cpu_ptr_->tree_left_values = d_tree_left_value_;
    dynamic_params_cpu_ptr_->tree_right_values = d_tree_right_value_;
    dynamic_params_cpu_ptr_->num_features = params_.num_features;
    dynamic_params_cpu_ptr_->hidden_size = params_.hidden_size;
    dynamic_params_cpu_ptr_->num_trees = params_.num_trees;
    dynamic_params_cpu_ptr_->batch_size = params_.batch_size;
    dynamic_params_cpu_ptr_->model_type = static_cast<int>(params_.model_type);

    kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(path_loss_predict_kernel));
    int threads = 256;
    int blocks = (params_.batch_size + threads - 1) / threads;
    kernel_config_.setup_kernel_dimensions(dim3(blocks, 1, 1), dim3(threads, 1, 1));
    framework::pipeline::setup_kernel_arguments(kernel_config_, dynamic_params_gpu_ptr_);
}

void PathLossPredictionModule::warmup(cudaStream_t stream) {
    if (d_output_) {
        cudaMemsetAsync(d_output_, 0, params_.batch_size * sizeof(float), stream);
    }
}

void PathLossPredictionModule::configure_io(
    const framework::pipeline::DynamicParams&,
    cudaStream_t stream) {
    if (!dynamic_params_cpu_ptr_) {
        throw std::runtime_error("Path loss dynamic descriptor not initialized");
    }
    if (!kernel_desc_mgr_) {
        throw std::runtime_error("Path loss kernel descriptor manager not initialized");
    }
    if (!current_features_) {
        throw std::runtime_error("Path loss input features not set");
    }
    if (!d_output_ || !d_mlp_w1_ || !d_mlp_b1_ || !d_mlp_w2_ || !d_mlp_b2_) {
        throw std::runtime_error("Path loss device buffers not initialized");
    }
    if (!d_tree_feature_idx_ || !d_tree_threshold_ || !d_tree_left_value_ || !d_tree_right_value_) {
        throw std::runtime_error("Path loss tree buffers not initialized");
    }

    dynamic_params_cpu_ptr_->features = static_cast<const float*>(current_features_);
    dynamic_params_cpu_ptr_->outputs = d_output_;
    dynamic_params_cpu_ptr_->w1 = d_mlp_w1_;
    dynamic_params_cpu_ptr_->b1 = d_mlp_b1_;
    dynamic_params_cpu_ptr_->w2 = d_mlp_w2_;
    dynamic_params_cpu_ptr_->b2 = d_mlp_b2_;
    dynamic_params_cpu_ptr_->tree_feature_idx = d_tree_feature_idx_;
    dynamic_params_cpu_ptr_->tree_thresholds = d_tree_threshold_;
    dynamic_params_cpu_ptr_->tree_left_values = d_tree_left_value_;
    dynamic_params_cpu_ptr_->tree_right_values = d_tree_right_value_;
    dynamic_params_cpu_ptr_->num_features = params_.num_features;
    dynamic_params_cpu_ptr_->hidden_size = params_.hidden_size;
    dynamic_params_cpu_ptr_->num_trees = params_.num_trees;
    dynamic_params_cpu_ptr_->batch_size = params_.batch_size;
    dynamic_params_cpu_ptr_->model_type = static_cast<int>(params_.model_type);

    kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);
}

std::vector<framework::tensor::TensorInfo>
PathLossPredictionModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "features") {
        return {input_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<framework::tensor::TensorInfo>
PathLossPredictionModule::get_output_tensor_info(std::string_view port_name) const {
    if (port_name == "path_loss_db") {
        return {output_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<std::string> PathLossPredictionModule::get_input_port_names() const {
    return {"features"};
}

std::vector<std::string> PathLossPredictionModule::get_output_port_names() const {
    return {"path_loss_db"};
}

framework::pipeline::ModuleMemoryRequirements PathLossPredictionModule::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements requirements;
    requirements.dynamic_kernel_descriptor_bytes = sizeof(PathLossDescriptor);
    requirements.device_tensor_bytes = params_.batch_size * sizeof(float);
    requirements.alignment = 256;
    return requirements;
}

framework::pipeline::OutputPortMemoryCharacteristics
PathLossPredictionModule::get_output_memory_characteristics(std::string_view) const {
    framework::pipeline::OutputPortMemoryCharacteristics characteristics;
    characteristics.provides_fixed_address_for_zero_copy = true;
    return characteristics;
}

void PathLossPredictionModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    if (!inputs.empty() && !inputs[0].tensors.empty()) {
        current_features_ = inputs[0].tensors[0].device_ptr;
    }
}

std::vector<framework::pipeline::PortInfo> PathLossPredictionModule::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_output_;
    }
    return outputs;
}

void PathLossPredictionModule::execute(cudaStream_t stream) {
    if (!current_features_) {
        throw std::runtime_error("Input features not set for execution");
    }

    if (!d_output_) {
        throw std::runtime_error("Path loss output buffer not initialized");
    }
    if (!model_loaded_ && !trt_enabled_) {
        throw std::runtime_error("Path loss model not loaded");
    }

    if (trt_enabled_) {
        if (!run_trt_inference(stream)) {
            throw std::runtime_error("TensorRT inference failed");
        }
        return;
    }

    const CUresult launch_err = kernel_config_.launch(stream);
    if (launch_err != CUDA_SUCCESS) {
        throw std::runtime_error("Path loss kernel launch failed");
    }
}

std::span<const CUgraphNode> PathLossPredictionModule::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());
    return {&kernel_node_, 1};
}

void PathLossPredictionModule::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams&) {
    if (!kernel_node_) {
        throw std::runtime_error("Path loss graph node not initialized");
    }
    const auto& kernel_params = kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, kernel_node_, &kernel_params);
}

void PathLossPredictionModule::setup_port_info() {
    input_ports_.resize(1);
    input_ports_[0].name = "features";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<std::size_t>(params_.batch_size),
         static_cast<std::size_t>(params_.num_features)});

    output_ports_.resize(1);
    output_ports_[0].name = "path_loss_db";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<std::size_t>(params_.batch_size)});
}

void PathLossPredictionModule::allocate_model_buffers() {
    const int num_features = params_.num_features;
    const int hidden_size = params_.hidden_size;
    int tree_count = params_.num_trees;

    std::vector<float> h_w1(hidden_size * num_features, 0.01f);
    std::vector<float> h_b1(hidden_size, 0.0f);
    std::vector<float> h_w2(hidden_size, 0.02f);
    std::vector<float> h_b2(1, 80.0f);
    std::vector<int> h_feature_idx(tree_count);
    std::vector<float> h_thresholds(tree_count);
    std::vector<float> h_left(tree_count);
    std::vector<float> h_right(tree_count);

    for (int i = 0; i < tree_count; ++i) {
        h_feature_idx[i] = i % num_features;
        h_thresholds[i] = 1.0f;
        h_left[i] = 1.5f;
        h_right[i] = 2.5f;
    }

    auto has_binary_model = [](const std::string& path) {
        auto dot = path.find_last_of('.');
        std::string ext = (dot == std::string::npos) ? "" : path.substr(dot);
        return ext == ".bin";
    };

    if (!params_.model_path.empty() && has_binary_model(params_.model_path)) {
        ModelType model_type = params_.model_type;
        int file_features = num_features;
        int file_hidden = hidden_size;
        int file_trees = tree_count;
        std::vector<float> file_w1;
        std::vector<float> file_b1;
        std::vector<float> file_w2;
        std::vector<float> file_b2;
        std::vector<int> file_feat_idx;
        std::vector<float> file_thr;
        std::vector<float> file_left;
        std::vector<float> file_right;

        if (load_model_file(params_.model_path, model_type, file_features, file_hidden, file_trees,
                            file_w1, file_b1, file_w2, file_b2,
                            file_feat_idx, file_thr, file_left, file_right)) {
            params_.model_type = model_type;
            if (model_type == ModelType::TensorRT_MLP) {
                if (file_features == num_features && file_hidden == hidden_size) {
                    if (file_w1.size() == h_w1.size() && file_b1.size() == h_b1.size() &&
                        file_w2.size() == h_w2.size() && file_b2.size() == h_b2.size()) {
                        h_w1 = std::move(file_w1);
                        h_b1 = std::move(file_b1);
                        h_w2 = std::move(file_w2);
                        h_b2 = std::move(file_b2);
                    } else {
                        std::cerr << "MLP weight sizes do not match expected layout; using defaults\n";
                    }
                } else {
                    std::cerr << "MLP dimensions do not match config; using defaults\n";
                }
            } else {
                if (file_features == num_features && file_trees > 0) {
                    tree_count = file_trees;
                    if (file_feat_idx.size() == static_cast<size_t>(tree_count)) {
                        h_feature_idx = std::move(file_feat_idx);
                        h_thresholds = std::move(file_thr);
                        h_left = std::move(file_left);
                        h_right = std::move(file_right);
                    } else {
                        std::cerr << "XGBoost-style model size mismatch; using defaults\n";
                    }
                } else {
                    std::cerr << "XGBoost-style feature/tree size mismatch; using defaults\n";
                }
            }
        }
    }

    cudaMalloc(&d_mlp_w1_, h_w1.size() * sizeof(float));
    cudaMalloc(&d_mlp_b1_, h_b1.size() * sizeof(float));
    cudaMalloc(&d_mlp_w2_, h_w2.size() * sizeof(float));
    cudaMalloc(&d_mlp_b2_, h_b2.size() * sizeof(float));
    cudaMemcpy(d_mlp_w1_, h_w1.data(), h_w1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_b1_, h_b1.data(), h_b1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_w2_, h_w2.data(), h_w2.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mlp_b2_, h_b2.data(), h_b2.size() * sizeof(float), cudaMemcpyHostToDevice);

    params_.num_trees = static_cast<int>(h_feature_idx.size());
    cudaMalloc(&d_tree_feature_idx_, h_feature_idx.size() * sizeof(int));
    cudaMalloc(&d_tree_threshold_, h_thresholds.size() * sizeof(float));
    cudaMalloc(&d_tree_left_value_, h_left.size() * sizeof(float));
    cudaMalloc(&d_tree_right_value_, h_right.size() * sizeof(float));
    cudaMemcpy(d_tree_feature_idx_, h_feature_idx.data(), h_feature_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_threshold_, h_thresholds.data(), h_thresholds.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_left_value_, h_left.data(), h_left.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tree_right_value_, h_right.data(), h_right.size() * sizeof(float), cudaMemcpyHostToDevice);

    model_loaded_ = true;
}

void PathLossPredictionModule::deallocate_model_buffers() {
    if (d_mlp_w1_) cudaFree(d_mlp_w1_);
    if (d_mlp_b1_) cudaFree(d_mlp_b1_);
    if (d_mlp_w2_) cudaFree(d_mlp_w2_);
    if (d_mlp_b2_) cudaFree(d_mlp_b2_);
    d_mlp_w1_ = nullptr;
    d_mlp_b1_ = nullptr;
    d_mlp_w2_ = nullptr;
    d_mlp_b2_ = nullptr;

    if (d_tree_feature_idx_) cudaFree(d_tree_feature_idx_);
    if (d_tree_threshold_) cudaFree(d_tree_threshold_);
    if (d_tree_left_value_) cudaFree(d_tree_left_value_);
    if (d_tree_right_value_) cudaFree(d_tree_right_value_);
    d_tree_feature_idx_ = nullptr;
    d_tree_threshold_ = nullptr;
    d_tree_left_value_ = nullptr;
    d_tree_right_value_ = nullptr;
}

void PathLossPredictionModule::run_fallback_mlp(cudaStream_t stream) {
    int threads = 128;
    int blocks = (params_.batch_size + threads - 1) / threads;
    mlp_predict_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(current_features_),
        d_output_,
        d_mlp_w1_,
        d_mlp_b1_,
        d_mlp_w2_,
        d_mlp_b2_,
        params_.num_features,
        params_.hidden_size,
        params_.batch_size);
}

void PathLossPredictionModule::run_xgboost_style(cudaStream_t stream) {
    int threads = 128;
    int blocks = (params_.batch_size + threads - 1) / threads;
    xgboost_predict_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(current_features_),
        d_output_,
        d_tree_feature_idx_,
        d_tree_threshold_,
        d_tree_left_value_,
        d_tree_right_value_,
        params_.num_trees,
        params_.num_features,
        params_.batch_size);
}

bool PathLossPredictionModule::run_tensorrt(cudaStream_t stream) {
    return run_trt_inference(stream);
}

bool PathLossPredictionModule::load_engine_from_file(const std::string& engine_path) {
#ifdef TENSORRT_AVAILABLE
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cout << "Failed to open TensorRT engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), size);
    if (!trt_engine_) {
        std::cout << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }
    return true;
#else
    (void)engine_path;
    return false;
#endif
}

bool PathLossPredictionModule::initialize_tensorrt_engine() {
#if defined(TENSORRT_AVAILABLE) && !defined(TENSORRT_STUB)
    try {
        trt_runtime_ = nvinfer1::createInferRuntime(gTrtLogger);
        if (!trt_runtime_) {
            std::cout << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }

        if (!load_engine_from_file(params_.model_path)) {
            return false;
        }

        trt_context_ = trt_engine_->createExecutionContext();
        if (!trt_context_) {
            std::cout << "Failed to create TensorRT execution context" << std::endl;
            return false;
        }

        size_t input_size = static_cast<size_t>(params_.batch_size) *
                            static_cast<size_t>(params_.num_features) * sizeof(float);
        cudaError_t err = cudaMalloc(&d_trt_input_, input_size);
        if (err != cudaSuccess) {
            std::cout << "Failed to allocate TensorRT input memory: "
                      << cudaGetErrorString(err) << std::endl;
            return false;
        }

        trt_enabled_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cout << "TensorRT initialization error: " << e.what() << std::endl;
        return false;
    }
#else
    return false;
#endif
}

void PathLossPredictionModule::cleanup_tensorrt_resources() {
#ifdef TENSORRT_AVAILABLE
    if (trt_context_) {
        delete trt_context_;
        trt_context_ = nullptr;
    }
    if (trt_engine_) {
        delete trt_engine_;
        trt_engine_ = nullptr;
    }
    if (trt_runtime_) {
        delete trt_runtime_;
        trt_runtime_ = nullptr;
    }
#else
    trt_context_ = nullptr;
    trt_engine_ = nullptr;
    trt_runtime_ = nullptr;
#endif

    if (d_trt_input_) {
        cudaFree(d_trt_input_);
        d_trt_input_ = nullptr;
    }
    trt_enabled_ = false;
}

bool PathLossPredictionModule::run_trt_inference(cudaStream_t stream) {
#if defined(TENSORRT_AVAILABLE) && !defined(TENSORRT_STUB)
    if (!trt_context_ || !trt_engine_ || !trt_enabled_) {
        return false;
    }

    size_t input_size = static_cast<size_t>(params_.batch_size) *
                        static_cast<size_t>(params_.num_features) * sizeof(float);
    cudaError_t err = cudaMemcpyAsync(
        d_trt_input_,
        current_features_,
        input_size,
        cudaMemcpyDeviceToDevice,
        stream
    );
    if (err != cudaSuccess) {
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    int num_io = trt_engine_->getNbIOTensors();
    for (int i = 0; i < num_io; ++i) {
        const char* name = trt_engine_->getIOTensorName(i);
        auto mode = trt_engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            trt_context_->setTensorAddress(name, d_trt_input_);
        } else {
            trt_context_->setTensorAddress(name, d_output_);
        }
    }
    return trt_context_->enqueueV3(stream);
#else
    int input_index = -1;
    int output_index = -1;
    for (int i = 0; i < trt_engine_->getNbBindings(); ++i) {
        if (trt_engine_->bindingIsInput(i)) {
            input_index = i;
        } else {
            output_index = i;
        }
    }
    if (input_index < 0 || output_index < 0) {
        return false;
    }

    void* bindings[2];
    bindings[input_index] = d_trt_input_;
    bindings[output_index] = d_output_;

#if NV_TENSORRT_MAJOR >= 8
    return trt_context_->enqueueV2(bindings, stream, nullptr);
#else
    return trt_context_->enqueue(1, bindings, stream, nullptr);
#endif
#endif
#else
    (void)stream;
    return false;
#endif
}

bool PathLossPredictionModule::load_model_file(
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
    std::vector<float>& right_values) const {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open model file: " << path << "\n";
        return false;
    }

    char magic[4] = {};
    file.read(magic, sizeof(magic));
    if (std::memcmp(magic, "PLM1", 4) != 0) {
        std::cerr << "Invalid model header\n";
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t type = 0;
    std::uint32_t features = 0;
    std::uint32_t hidden = 0;
    std::uint32_t trees = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&type), sizeof(type));
    file.read(reinterpret_cast<char*>(&features), sizeof(features));
    file.read(reinterpret_cast<char*>(&hidden), sizeof(hidden));
    file.read(reinterpret_cast<char*>(&trees), sizeof(trees));

    if (!file || version != 1) {
        std::cerr << "Unsupported model version\n";
        return false;
    }

    model_type = (type == 1) ? ModelType::XGBoostStyle : ModelType::TensorRT_MLP;
    num_features = static_cast<int>(features);
    hidden_size = static_cast<int>(hidden);
    num_trees = static_cast<int>(trees);

    if (model_type == ModelType::TensorRT_MLP) {
        const size_t w1_count = static_cast<size_t>(hidden_size) * num_features;
        const size_t b1_count = static_cast<size_t>(hidden_size);
        const size_t w2_count = static_cast<size_t>(hidden_size);
        const size_t b2_count = 1;

        w1.resize(w1_count);
        b1.resize(b1_count);
        w2.resize(w2_count);
        b2.resize(b2_count);

        file.read(reinterpret_cast<char*>(w1.data()), w1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(w2.data()), w2.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));
    } else {
        feature_idx.resize(static_cast<size_t>(num_trees));
        thresholds.resize(static_cast<size_t>(num_trees));
        left_values.resize(static_cast<size_t>(num_trees));
        right_values.resize(static_cast<size_t>(num_trees));

        file.read(reinterpret_cast<char*>(feature_idx.data()), feature_idx.size() * sizeof(int));
        file.read(reinterpret_cast<char*>(thresholds.data()), thresholds.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(left_values.data()), left_values.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(right_values.data()), right_values.size() * sizeof(float));
    }

    if (!file) {
        std::cerr << "Model file truncated\n";
        return false;
    }

    return true;
}

} // namespace path_loss_prediction
