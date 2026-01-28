#pragma once

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"

#include <cuda_runtime.h>
#include <span>
#include <string>
#include <vector>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferVersion.h>
#include <fstream>
#endif

namespace user_scheduling {

struct SchedulingParams {
    int num_ues = 16;
    int num_features = 5;
    std::string model_path;
};

class UserSchedulingModule : public framework::pipeline::IModule,
                             public framework::pipeline::IAllocationInfoProvider,
                             public framework::pipeline::IStreamExecutor {
public:
    explicit UserSchedulingModule(const std::string& module_id, const SchedulingParams& params);
    ~UserSchedulingModule() override;

    [[nodiscard]] std::string_view get_type_id() const override { return "user_scheduling_module"; }
    [[nodiscard]] std::string_view get_instance_id() const override { return module_id_; }

    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_input_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<framework::tensor::TensorInfo>
    get_output_tensor_info(std::string_view port_name) const override;

    [[nodiscard]] std::vector<std::string> get_input_port_names() const override;
    [[nodiscard]] std::vector<std::string> get_output_port_names() const override;

    [[nodiscard]] framework::pipeline::ModuleMemoryRequirements get_requirements() const override;

    [[nodiscard]] framework::pipeline::OutputPortMemoryCharacteristics
    get_output_memory_characteristics(std::string_view port_name) const override;

    void setup_memory(const framework::pipeline::ModuleMemorySlice& memory_slice) override;
    void warmup(cudaStream_t stream) override;
    void configure_io(const framework::pipeline::DynamicParams& params, cudaStream_t stream) override;

    framework::pipeline::IGraphNodeProvider* as_graph_node_provider() override { return nullptr; }
    framework::pipeline::IStreamExecutor* as_stream_executor() override { return this; }

    void set_inputs(std::span<const framework::pipeline::PortInfo> inputs) override;
    [[nodiscard]] std::vector<framework::pipeline::PortInfo> get_outputs() const override;
    void execute(cudaStream_t stream) override;

    bool set_model(
        const std::vector<float>& mean,
        const std::vector<float>& std,
        const std::vector<float>& weights,
        float bias
    );

private:
    std::string module_id_;
    SchedulingParams params_;
    framework::pipeline::ModuleMemorySlice mem_slice_;

    std::vector<framework::pipeline::PortInfo> input_ports_;
    std::vector<framework::pipeline::PortInfo> output_ports_;

    const void* current_features_{nullptr};

    float* d_scores_{nullptr};
    float* d_mean_{nullptr};
    float* d_std_{nullptr};
    float* d_weights_{nullptr};
    float model_bias_{0.0f};
    bool model_loaded_{false};

#ifdef TENSORRT_AVAILABLE
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

    void setup_port_info();
    void allocate_gpu_memory();
    void deallocate_gpu_memory();
    void cleanup_tensorrt_resources();
    bool initialize_tensorrt_engine();
    bool load_engine_from_file(const std::string& engine_path);
    bool run_trt_inference(cudaStream_t stream);
};

} // namespace user_scheduling
