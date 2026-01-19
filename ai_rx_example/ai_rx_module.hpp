
#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
#endif

// ...existing code...

#pragma once

#include "pipeline/imodule.hpp"
#include "pipeline/istream_executor.hpp"
#include "pipeline/iallocation_info_provider.hpp"
#include "pipeline/types.hpp"
#include "tensor/tensor_info.hpp"
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <span>

namespace ai_rx_example {

struct AiRxParams {
	int num_symbols = 1024;
	std::string model_path;
};

class AiRxModule : public framework::pipeline::IModule,
				   public framework::pipeline::IAllocationInfoProvider,
				   public framework::pipeline::IStreamExecutor {
public:
	explicit AiRxModule(const std::string& module_id, const AiRxParams& params);
	~AiRxModule() override;

	// IModule interface
	[[nodiscard]] std::string_view get_type_id() const override { return "ai_rx_module"; }
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

	[[nodiscard]] const std::string& get_module_id() const { return module_id_; }

private:
	std::string module_id_;
	AiRxParams params_;

	// Port info
	std::vector<framework::pipeline::PortInfo> input_ports_;
	std::vector<framework::pipeline::PortInfo> output_ports_;

	// Device memory
	float* d_rx_symbols_ = nullptr;
	float* d_rx_bits_ = nullptr;
	void* current_rx_symbols_ = nullptr;

#ifdef TENSORRT_AVAILABLE
	nvinfer1::IRuntime* trt_runtime_ = nullptr;
	nvinfer1::ICudaEngine* trt_engine_ = nullptr;
	nvinfer1::IExecutionContext* trt_context_ = nullptr;
#else
	void* trt_runtime_ = nullptr;
	void* trt_engine_ = nullptr;
	void* trt_context_ = nullptr;
#endif
	float* d_trt_input_ = nullptr;
	float* d_trt_output_ = nullptr;

	// Internal helpers
	void setup_port_info();
	void allocate_gpu_memory();
	void deallocate_gpu_memory();
	void cleanup_tensorrt_resources();
	bool load_engine_from_file(const std::string& engine_path);
	bool initialize_tensorrt_engine();
	bool run_trt_inference(cudaStream_t stream);
};

} // namespace ai_rx_example
