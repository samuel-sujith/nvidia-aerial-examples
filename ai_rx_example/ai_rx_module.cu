#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <fstream>
#include <iostream>
class Logger : public nvinfer1::ILogger {
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << "TensorRT: " << msg << std::endl;
		}
	}
};
static Logger gLogger;
#endif
	// TensorRT engine initialization if model_path is provided
	if (!params_.model_path.empty()) {
		initialize_tensorrt_engine();
	}
void AiRxModule::cleanup_tensorrt_resources() {
#ifdef TENSORRT_AVAILABLE
	if (trt_context_) { trt_context_->destroy(); trt_context_ = nullptr; }
	if (trt_engine_) { trt_engine_->destroy(); trt_engine_ = nullptr; }
	if (trt_runtime_) { trt_runtime_->destroy(); trt_runtime_ = nullptr; }
	if (d_trt_input_) { cudaFree(d_trt_input_); d_trt_input_ = nullptr; }
	if (d_trt_output_) { cudaFree(d_trt_output_); d_trt_output_ = nullptr; }
#endif
}

bool AiRxModule::load_engine_from_file(const std::string& engine_path) {
#ifdef TENSORRT_AVAILABLE
	std::ifstream file(engine_path, std::ios::binary);
	if (!file.good()) return false;
	file.seekg(0, file.end);
	size_t size = file.tellg();
	file.seekg(0, file.beg);
	std::vector<char> engine_data(size);
	file.read(engine_data.data(), size);
	file.close();
	trt_runtime_ = nvinfer1::createInferRuntime(gLogger);
	trt_engine_ = trt_runtime_->deserializeCudaEngine(engine_data.data(), size, nullptr);
	if (!trt_engine_) return false;
	trt_context_ = trt_engine_->createExecutionContext();
	return trt_context_ != nullptr;
#else
	return false;
#endif
}

bool AiRxModule::initialize_tensorrt_engine() {
	cleanup_tensorrt_resources();
	if (!load_engine_from_file(params_.model_path)) return false;
#ifdef TENSORRT_AVAILABLE
	int input_idx = trt_engine_->getBindingIndex("input");
	int output_idx = trt_engine_->getBindingIndex("output");
	auto input_dims = trt_engine_->getBindingDimensions(input_idx);
	auto output_dims = trt_engine_->getBindingDimensions(output_idx);
	size_t input_size = 1;
	for (int i = 0; i < input_dims.nbDims; ++i) input_size *= input_dims.d[i];
	size_t output_size = 1;
	for (int i = 0; i < output_dims.nbDims; ++i) output_size *= output_dims.d[i];
	cudaMalloc(&d_trt_input_, input_size * sizeof(float));
	cudaMalloc(&d_trt_output_, output_size * sizeof(float));
#endif
	return true;
}

bool AiRxModule::run_trt_inference(cudaStream_t stream) {
#ifdef TENSORRT_AVAILABLE
	void* bindings[2];
	int input_idx = trt_engine_->getBindingIndex("input");
	int output_idx = trt_engine_->getBindingIndex("output");
	bindings[input_idx] = d_trt_input_;
	bindings[output_idx] = d_trt_output_;
	return trt_context_->enqueueV2(bindings, stream, nullptr);
#else
	return false;
#endif
}

#include "ai_rx_module.hpp"
#include <cuda_runtime.h>

#include "ai_rx_module.hpp"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

namespace ai_rx_example {

// CUDA kernel for AI Rx inference (dummy thresholding)
__global__ void ai_rx_infer_kernel_cuda(const float* rx_symbols, float* rx_bits, int num_symbols) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_symbols) {
		rx_bits[idx] = rx_symbols[2*idx] > 0 ? 1.0f : 0.0f;
	}
}

AiRxModule::AiRxModule(const std::string& module_id, const AiRxParams& params)
	: module_id_(module_id), params_(params), d_rx_symbols_(nullptr), d_rx_bits_(nullptr), current_rx_symbols_(nullptr) {
	setup_port_info();
	allocate_gpu_memory();
#ifdef TENSORRT_AVAILABLE
	if (!params_.model_path.empty()) {
		initialize_tensorrt_engine();
	}
#endif
}

AiRxModule::~AiRxModule() {
	deallocate_gpu_memory();
}

void AiRxModule::setup_port_info() {
	using namespace framework::tensor;
	input_ports_.resize(1);
	input_ports_[0].name = "rx_symbols";
	input_ports_[0].tensors.resize(1);
	input_ports_[0].tensors[0].device_ptr = nullptr;
	input_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
		NvDataType::TensorR32F, std::vector<size_t>{static_cast<size_t>(params_.num_symbols), 2});

	output_ports_.resize(1);
	output_ports_[0].name = "rx_bits";
	output_ports_[0].tensors.resize(1);
	output_ports_[0].tensors[0].device_ptr = nullptr;
	output_ports_[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
		NvDataType::TensorR32F, std::vector<size_t>{static_cast<size_t>(params_.num_symbols)});
}

std::vector<std::string> AiRxModule::get_input_port_names() const {
	std::vector<std::string> names;
	for (const auto& port : input_ports_) names.push_back(port.name);
	return names;
}

std::vector<std::string> AiRxModule::get_output_port_names() const {
	std::vector<std::string> names;
	for (const auto& port : output_ports_) names.push_back(port.name);
	return names;
}

std::vector<framework::tensor::TensorInfo>
AiRxModule::get_input_tensor_info(std::string_view port_name) const {
	using namespace framework::tensor;
	std::vector<TensorInfo> infos;
	if (port_name == "rx_symbols") {
		infos.emplace_back(NvDataType::TensorR32F, std::vector<size_t>{static_cast<size_t>(params_.num_symbols), 2});
	}
	return infos;
}

std::vector<framework::tensor::TensorInfo>
AiRxModule::get_output_tensor_info(std::string_view port_name) const {
	using namespace framework::tensor;
	std::vector<TensorInfo> infos;
	if (port_name == "rx_bits") {
		infos.emplace_back(NvDataType::TensorR32F, std::vector<size_t>{static_cast<size_t>(params_.num_symbols)});
	}
	return infos;
}

framework::pipeline::ModuleMemoryRequirements AiRxModule::get_requirements() const {
	framework::pipeline::ModuleMemoryRequirements req;
	req.device_tensor_bytes = params_.num_symbols * (2 * sizeof(float) + sizeof(float)); // rx_symbols + rx_bits
	req.alignment = 256;
	return req;
}

framework::pipeline::OutputPortMemoryCharacteristics
AiRxModule::get_output_memory_characteristics(std::string_view) const {
	framework::pipeline::OutputPortMemoryCharacteristics c{};
	c.provides_fixed_address_for_zero_copy = true;
	return c;
}

void AiRxModule::setup_memory(const framework::pipeline::ModuleMemorySlice&) {
	allocate_gpu_memory();
}

void AiRxModule::warmup(cudaStream_t stream) {
	cudaStreamSynchronize(stream);
}

void AiRxModule::configure_io(const framework::pipeline::DynamicParams&, cudaStream_t) {
	// No dynamic params for now
}

void AiRxModule::set_inputs(std::span<const framework::pipeline::PortInfo> inputs) {
	for (const auto& port : inputs) {
		if (port.name == "rx_symbols" && !port.tensors.empty()) {
			current_rx_symbols_ = port.tensors[0].device_ptr;
		}
	}
}

std::vector<framework::pipeline::PortInfo> AiRxModule::get_outputs() const {
	auto outputs = output_ports_;
	if (!outputs.empty() && !outputs[0].tensors.empty()) {
		outputs[0].tensors[0].device_ptr = d_rx_bits_;
	}
	return outputs;
}

void AiRxModule::execute(cudaStream_t stream) {
	if (!current_rx_symbols_) {
		throw std::runtime_error("Input rx_symbols not set for AI Rx");
	}
	cudaMemcpyAsync(d_rx_symbols_, current_rx_symbols_, params_.num_symbols * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
#ifdef TENSORRT_AVAILABLE
	if (!params_.model_path.empty() && trt_context_ && d_trt_input_ && d_trt_output_) {
		// Copy input to TensorRT input buffer
		cudaMemcpyAsync(d_trt_input_, d_rx_symbols_, params_.num_symbols * 2 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
		run_trt_inference(stream);
		// Copy output from TensorRT output buffer
		cudaMemcpyAsync(d_rx_bits_, d_trt_output_, params_.num_symbols * sizeof(float), cudaMemcpyDeviceToDevice, stream);
	} else
#endif
	{
		int threads = 256;
		int blocks = (params_.num_symbols + threads - 1) / threads;
		ai_rx_infer_kernel_cuda<<<blocks, threads, 0, stream>>>(d_rx_symbols_, d_rx_bits_, params_.num_symbols);
	}
}

void AiRxModule::allocate_gpu_memory() {
	cudaMalloc(&d_rx_symbols_, params_.num_symbols * 2 * sizeof(float));
	cudaMalloc(&d_rx_bits_, params_.num_symbols * sizeof(float));
}

void AiRxModule::deallocate_gpu_memory() {
	if (d_rx_symbols_) cudaFree(d_rx_symbols_);
	if (d_rx_bits_) cudaFree(d_rx_bits_);
	d_rx_symbols_ = nullptr;
	d_rx_bits_ = nullptr;
}

} // namespace ai_rx_example
