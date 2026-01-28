#include "user_scheduling_module.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef TENSORRT_AVAILABLE
#include <NvInferVersion.h>
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger gLogger;
#endif

namespace user_scheduling {

namespace {

__global__ void schedule_scores_kernel(const UserSchedulingDescriptor* desc) {
    if (!desc || !desc->features || !desc->scores || !desc->mean || !desc->std || !desc->weights) {
        return;
    }
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= desc->num_ues) {
        return;
    }

    float z = desc->bias;
    const float* ue_features = desc->features + idx * desc->num_features;
    for (int f = 0; f < desc->num_features; ++f) {
        float norm = (ue_features[f] - desc->mean[f]) / desc->std[f];
        z += norm * desc->weights[f];
    }
    desc->scores[idx] = 1.0f / (1.0f + expf(-z));
}

} // namespace

UserSchedulingModule::UserSchedulingModule(const std::string& module_id, const SchedulingParams& params)
    : module_id_(module_id), params_(params) {
    setup_port_info();
    if (!params_.model_path.empty()) {
        auto dot = params_.model_path.find_last_of('.');
        std::string ext = (dot == std::string::npos) ? "" : params_.model_path.substr(dot);
        if (ext == ".engine" || ext == ".trt") {
            initialize_tensorrt_engine();
        }
    }
}

UserSchedulingModule::~UserSchedulingModule() {
    cleanup_tensorrt_resources();
    deallocate_gpu_memory();
}

void UserSchedulingModule::setup_port_info() {
    input_ports_.resize(1);
    input_ports_[0].name = "ue_features";
    input_ports_[0].tensors.resize(1);
    input_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(params_.num_ues), static_cast<size_t>(params_.num_features)}
    );

    output_ports_.resize(1);
    output_ports_[0].name = "ue_scores";
    output_ports_[0].tensors.resize(1);
    output_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(params_.num_ues)}
    );
}

void UserSchedulingModule::allocate_gpu_memory() {
    cudaError_t err = cudaMalloc(&d_mean_, params_.num_features * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for mean");
    }
    err = cudaMalloc(&d_std_, params_.num_features * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for std");
    }
    err = cudaMalloc(&d_weights_, params_.num_features * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for weights");
    }
}

void UserSchedulingModule::deallocate_gpu_memory() {
    if (d_mean_) {
        cudaFree(d_mean_);
        d_mean_ = nullptr;
    }
    if (d_std_) {
        cudaFree(d_std_);
        d_std_ = nullptr;
    }
    if (d_weights_) {
        cudaFree(d_weights_);
        d_weights_ = nullptr;
    }
}

std::vector<framework::tensor::TensorInfo>
UserSchedulingModule::get_input_tensor_info(std::string_view port_name) const {
    if (port_name == "ue_features") {
        return {input_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<framework::tensor::TensorInfo>
UserSchedulingModule::get_output_tensor_info(std::string_view port_name) const {
    if (port_name == "ue_scores") {
        return {output_ports_[0].tensors[0].tensor_info};
    }
    return {};
}

std::vector<std::string> UserSchedulingModule::get_input_port_names() const {
    return {"ue_features"};
}

std::vector<std::string> UserSchedulingModule::get_output_port_names() const {
    return {"ue_scores"};
}

framework::pipeline::ModuleMemoryRequirements UserSchedulingModule::get_requirements() const {
    framework::pipeline::ModuleMemoryRequirements requirements;
    requirements.dynamic_kernel_descriptor_bytes = sizeof(UserSchedulingDescriptor);
    requirements.device_tensor_bytes = params_.num_ues * sizeof(float);
    requirements.alignment = 256;
    return requirements;
}

framework::pipeline::OutputPortMemoryCharacteristics
UserSchedulingModule::get_output_memory_characteristics(std::string_view port_name) const {
    framework::pipeline::OutputPortMemoryCharacteristics characteristics;
    characteristics.provides_fixed_address_for_zero_copy = true;
    return characteristics;
}

void UserSchedulingModule::setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) {
    mem_slice_ = memory_slice;
    if (!mem_slice_.device_tensor_ptr) {
        throw std::runtime_error("User scheduling output device buffer not allocated");
    }
    d_scores_ = reinterpret_cast<float*>(mem_slice_.device_tensor_ptr);
    allocate_gpu_memory();

    kernel_desc_mgr_ = std::make_unique<framework::pipeline::KernelDescriptorAccessor>(memory_slice);
    dynamic_params_cpu_ptr_ =
        &kernel_desc_mgr_->create_dynamic_param<UserSchedulingDescriptor>(0);
    dynamic_params_gpu_ptr_ = kernel_desc_mgr_->get_dynamic_device_ptr<UserSchedulingDescriptor>(0);
    if (!dynamic_params_gpu_ptr_) {
        throw std::runtime_error("User scheduling dynamic descriptor device pointer not allocated");
    }

    dynamic_params_cpu_ptr_->scores = d_scores_;
    dynamic_params_cpu_ptr_->num_ues = params_.num_ues;
    dynamic_params_cpu_ptr_->num_features = params_.num_features;

    kernel_config_.setup_kernel_function(reinterpret_cast<const void*>(schedule_scores_kernel));
    int threads = 256;
    int blocks = (params_.num_ues + threads - 1) / threads;
    kernel_config_.setup_kernel_dimensions(dim3(blocks, 1, 1), dim3(threads, 1, 1));
    framework::pipeline::setup_kernel_arguments(kernel_config_, *dynamic_params_gpu_ptr_);
}

void UserSchedulingModule::warmup(cudaStream_t stream) {
    if (!d_scores_) {
        throw std::runtime_error("User scheduling output buffer not initialized");
    }
    cudaMemsetAsync(d_scores_, 0, params_.num_ues * sizeof(float), stream);
}

void UserSchedulingModule::configure_io(
    const framework::pipeline::DynamicParams& params,
    cudaStream_t stream
) {
    (void)params;
    if (!dynamic_params_cpu_ptr_) {
        throw std::runtime_error("User scheduling dynamic descriptor not initialized");
    }
    if (!kernel_desc_mgr_) {
        throw std::runtime_error("User scheduling kernel descriptor manager not initialized");
    }
    if (!current_features_) {
        throw std::runtime_error("User scheduling input features not set");
    }
    if (!d_scores_ || !d_mean_ || !d_std_ || !d_weights_) {
        throw std::runtime_error("User scheduling device buffers not initialized");
    }
    dynamic_params_cpu_ptr_->features = static_cast<const float*>(current_features_);
    dynamic_params_cpu_ptr_->scores = d_scores_;
    dynamic_params_cpu_ptr_->num_ues = params_.num_ues;
    dynamic_params_cpu_ptr_->num_features = params_.num_features;
    dynamic_params_cpu_ptr_->mean = d_mean_;
    dynamic_params_cpu_ptr_->std = d_std_;
    dynamic_params_cpu_ptr_->weights = d_weights_;
    dynamic_params_cpu_ptr_->bias = model_bias_;
    kernel_desc_mgr_->copy_dynamic_descriptors_to_device(stream);
}

void UserSchedulingModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
    if (!inputs.empty() && !inputs[0].tensors.empty()) {
        current_features_ = inputs[0].tensors[0].device_ptr;
    }
}

std::vector<framework::pipeline::PortInfo> UserSchedulingModule::get_outputs() const {
    auto outputs = output_ports_;
    if (!outputs.empty() && !outputs[0].tensors.empty()) {
        outputs[0].tensors[0].device_ptr = d_scores_;
    }
    return outputs;
}

bool UserSchedulingModule::set_model(
    const std::vector<float>& mean,
    const std::vector<float>& std,
    const std::vector<float>& weights,
    float bias
) {
    if (mean.size() != static_cast<size_t>(params_.num_features) ||
        std.size() != static_cast<size_t>(params_.num_features) ||
        weights.size() != static_cast<size_t>(params_.num_features)) {
        return false;
    }

    if (!d_mean_ || !d_std_ || !d_weights_) {
        return false;
    }
    cudaError_t err = cudaMemcpy(d_mean_, mean.data(), mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemcpy(d_std_, std.data(), std.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemcpy(d_weights_, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return false;
    }

    model_bias_ = bias;
    model_loaded_ = true;
    if (dynamic_params_cpu_ptr_) {
        dynamic_params_cpu_ptr_->mean = d_mean_;
        dynamic_params_cpu_ptr_->std = d_std_;
        dynamic_params_cpu_ptr_->weights = d_weights_;
        dynamic_params_cpu_ptr_->bias = model_bias_;
    }
    return true;
}

void UserSchedulingModule::execute(cudaStream_t stream) {
    if (!current_features_) {
        throw std::runtime_error("Input features not set for execution");
    }
    if (!d_scores_) {
        throw std::runtime_error("User scheduling output buffer not initialized");
    }
    if (!model_loaded_ && !trt_enabled_) {
        throw std::runtime_error("Scheduling model not loaded");
    }

    if (trt_enabled_) {
        if (!run_trt_inference(stream)) {
            throw std::runtime_error("TensorRT inference failed");
        }
        return;
    }

    const CUresult launch_err = kernel_config_.launch(stream);
    if (launch_err != CUDA_SUCCESS) {
        throw std::runtime_error("Scheduling kernel launch failed");
    }
}

std::span<const CUgraphNode> UserSchedulingModule::add_node_to_graph(
    gsl_lite::not_null<framework::pipeline::IGraph*> graph,
    std::span<const CUgraphNode> deps) {
    kernel_node_ = graph->add_kernel_node(deps, kernel_config_.get_kernel_params());
    return {&kernel_node_, 1};
}

void UserSchedulingModule::update_graph_node_params(
    CUgraphExec exec,
    const framework::pipeline::DynamicParams& /*params*/) {
    if (!kernel_node_) {
        throw std::runtime_error("User scheduling graph node not initialized");
    }
    const auto& kernel_params = kernel_config_.get_kernel_params();
    cuGraphExecKernelNodeSetParams(exec, kernel_node_, &kernel_params);
}

bool UserSchedulingModule::load_engine_from_file(const std::string& engine_path) {
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

bool UserSchedulingModule::initialize_tensorrt_engine() {
#if defined(TENSORRT_AVAILABLE) && !defined(TENSORRT_STUB)
    try {
        trt_runtime_ = nvinfer1::createInferRuntime(gLogger);
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

        size_t input_size = static_cast<size_t>(params_.num_ues) *
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

void UserSchedulingModule::cleanup_tensorrt_resources() {
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

bool UserSchedulingModule::run_trt_inference(cudaStream_t stream) {
#if defined(TENSORRT_AVAILABLE) && !defined(TENSORRT_STUB)
    if (!trt_context_ || !trt_engine_) {
        return false;
    }

    size_t input_size = static_cast<size_t>(params_.num_ues) *
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
            trt_context_->setTensorAddress(name, d_scores_);
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
    bindings[output_index] = d_scores_;

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

} // namespace user_scheduling
