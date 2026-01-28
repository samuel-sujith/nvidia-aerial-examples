

#pragma once

#include "ai_rx_module.hpp"
#include "pipeline/ipipeline.hpp"
#include "pipeline/types.hpp"
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <mutex>
#include <chrono>

namespace ai_rx_example {

class AiRxPipeline {
public:
	struct PipelineConfig {
		AiRxParams ai_rx_params;
		std::string module_id;
		bool enable_profiling = false;
		size_t stream_buffer_size = 1024;
		PipelineConfig() { module_id = "ai_rx_pipeline"; }
	};

	struct PerformanceMetrics {
		size_t total_processed_frames = 0;
		double avg_processing_time_ms = 0.0;
		double peak_processing_time_ms = 0.0;
		double throughput_mbps = 0.0;
	};

	explicit AiRxPipeline(const PipelineConfig& config);
	~AiRxPipeline();

	bool initialize();
	bool process_rx(
		const std::vector<std::pair<float, float>>& rx_symbols,
		std::vector<float>& rx_bits,
		cudaStream_t stream = 0
	);

	void reset_metrics();
	PerformanceMetrics get_performance_metrics() const;

private:
	PipelineConfig config_;
	std::shared_ptr<AiRxModule> ai_rx_module_;

	// Performance tracking
	mutable std::mutex metrics_mutex_;
	PerformanceMetrics metrics_;
	std::chrono::high_resolution_clock::time_point start_time_;

	// Memory management
	float* h_rx_symbols_ = nullptr;
	float* h_rx_bits_ = nullptr;
	float* d_rx_symbols_ = nullptr;
	float* d_rx_bits_ = nullptr;
	void* d_module_tensor_ = nullptr;
	size_t module_tensor_bytes_ = 0;

	void allocate_buffers();
	void deallocate_buffers();
	void update_metrics(double processing_time_ms, size_t num_symbols);
	size_t get_rx_symbols_size() const;
	size_t get_rx_bits_size() const;
};

} // namespace ai_rx_example
