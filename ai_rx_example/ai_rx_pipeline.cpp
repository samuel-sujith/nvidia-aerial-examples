

#include "ai_rx_pipeline.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>

namespace ai_rx_example {

AiRxPipeline::AiRxPipeline(const PipelineConfig& config)
	: config_(config), h_rx_symbols_(nullptr), h_rx_bits_(nullptr), d_rx_symbols_(nullptr), d_rx_bits_(nullptr) {
}

AiRxPipeline::~AiRxPipeline() {
	deallocate_buffers();
}

bool AiRxPipeline::initialize() {
	try {
		ai_rx_module_ = std::make_shared<AiRxModule>(config_.module_id, config_.ai_rx_params);
		allocate_buffers();
		reset_metrics();
		return true;
	} catch (const std::exception& e) {
		std::cerr << "Failed to initialize AI Rx pipeline: " << e.what() << std::endl;
		return false;
	}
}

bool AiRxPipeline::process_rx(
	const std::vector<std::pair<float, float>>& rx_symbols,
	std::vector<float>& rx_bits,
	cudaStream_t stream
) {
	if (!ai_rx_module_) {
		std::cerr << "AI Rx module not initialized" << std::endl;
		return false;
	}

	auto start_time = std::chrono::high_resolution_clock::now();

	size_t num_symbols = rx_symbols.size();
	if (num_symbols == 0) return false;

	// Prepare host buffer
	for (size_t i = 0; i < num_symbols; ++i) {
		h_rx_symbols_[2*i] = rx_symbols[i].first;
		h_rx_symbols_[2*i+1] = rx_symbols[i].second;
	}

	// Copy input to device
	cudaMemcpyAsync(d_rx_symbols_, h_rx_symbols_, num_symbols * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);

	// Set up input port
	std::vector<framework::pipeline::PortInfo> inputs(1);
	inputs[0].name = "rx_symbols";
	inputs[0].tensors.resize(1);
	inputs[0].tensors[0].device_ptr = d_rx_symbols_;
	// tensor_info is not used in this minimal example
	ai_rx_module_->set_inputs(inputs);

	// Run inference
	ai_rx_module_->execute(stream);

	// Copy output from device
	cudaMemcpyAsync(h_rx_bits_, d_rx_bits_, num_symbols * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// Fill output vector
	rx_bits.resize(num_symbols);
	std::copy(h_rx_bits_, h_rx_bits_ + num_symbols, rx_bits.begin());

	// Update metrics
	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	update_metrics(duration.count() / 1000.0, num_symbols);

	return true;
}

void AiRxPipeline::allocate_buffers() {
	size_t num_symbols = config_.ai_rx_params.num_symbols;
	cudaMallocHost(&h_rx_symbols_, num_symbols * 2 * sizeof(float));
	cudaMallocHost(&h_rx_bits_, num_symbols * sizeof(float));
	cudaMalloc(&d_rx_symbols_, num_symbols * 2 * sizeof(float));
	cudaMalloc(&d_rx_bits_, num_symbols * sizeof(float));
}

void AiRxPipeline::deallocate_buffers() {
	if (h_rx_symbols_) cudaFreeHost(h_rx_symbols_);
	if (h_rx_bits_) cudaFreeHost(h_rx_bits_);
	if (d_rx_symbols_) cudaFree(d_rx_symbols_);
	if (d_rx_bits_) cudaFree(d_rx_bits_);
	h_rx_symbols_ = nullptr;
	h_rx_bits_ = nullptr;
	d_rx_symbols_ = nullptr;
	d_rx_bits_ = nullptr;
}

void AiRxPipeline::reset_metrics() {
	std::lock_guard<std::mutex> lock(metrics_mutex_);
	metrics_ = PerformanceMetrics{};
}

AiRxPipeline::PerformanceMetrics AiRxPipeline::get_performance_metrics() const {
	std::lock_guard<std::mutex> lock(metrics_mutex_);
	return metrics_;
}

void AiRxPipeline::update_metrics(double processing_time_ms, size_t num_symbols) {
	std::lock_guard<std::mutex> lock(metrics_mutex_);
	metrics_.total_processed_frames++;
	metrics_.avg_processing_time_ms = (metrics_.avg_processing_time_ms * (metrics_.total_processed_frames - 1) + processing_time_ms) / metrics_.total_processed_frames;
	metrics_.peak_processing_time_ms = std::max(metrics_.peak_processing_time_ms, processing_time_ms);
	metrics_.throughput_mbps = (num_symbols * sizeof(float) * 8) / (processing_time_ms * 1000.0); // Mbps
}

size_t AiRxPipeline::get_rx_symbols_size() const {
	return config_.ai_rx_params.num_symbols * 2 * sizeof(float);
}

size_t AiRxPipeline::get_rx_bits_size() const {
	return config_.ai_rx_params.num_symbols * sizeof(float);
}

} // namespace ai_rx_example
