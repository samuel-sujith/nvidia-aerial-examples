#include "user_scheduling_pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace user_scheduling {

UserSchedulingPipeline::UserSchedulingPipeline(const PipelineConfig& config)
    : config_(config) {}

UserSchedulingPipeline::~UserSchedulingPipeline() {
    deallocate_buffers();
}

bool UserSchedulingPipeline::initialize() {
    try {
        if (!config_.model_path.empty()) {
            config_.scheduling_params.model_path = config_.model_path;
        }
        scheduling_module_ = std::make_shared<UserSchedulingModule>(
            config_.module_id,
            config_.scheduling_params
        );

        auto requirements = scheduling_module_->get_requirements();
        module_tensor_bytes_ = requirements.device_tensor_bytes;
        static_desc_bytes_ = requirements.static_kernel_descriptor_bytes;
        dynamic_desc_bytes_ = requirements.dynamic_kernel_descriptor_bytes;
        if (module_tensor_bytes_ > 0) {
            cudaMalloc(&d_module_tensor_, module_tensor_bytes_);
            framework::pipeline::ModuleMemorySlice slice{};
            slice.device_tensor_ptr = reinterpret_cast<std::byte*>(d_module_tensor_);
            slice.device_tensor_bytes = module_tensor_bytes_;
            if (static_desc_bytes_ > 0) {
                cudaMallocHost(&static_desc_cpu_, static_desc_bytes_);
                cudaMalloc(&static_desc_gpu_, static_desc_bytes_);
                slice.static_kernel_descriptor_cpu_ptr = static_desc_cpu_;
                slice.static_kernel_descriptor_gpu_ptr = static_desc_gpu_;
                slice.static_kernel_descriptor_bytes = static_desc_bytes_;
            }
            if (dynamic_desc_bytes_ > 0) {
                cudaMallocHost(&dynamic_desc_cpu_, dynamic_desc_bytes_);
                cudaMalloc(&dynamic_desc_gpu_, dynamic_desc_bytes_);
                slice.dynamic_kernel_descriptor_cpu_ptr = dynamic_desc_cpu_;
                slice.dynamic_kernel_descriptor_gpu_ptr = dynamic_desc_gpu_;
                slice.dynamic_kernel_descriptor_bytes = dynamic_desc_bytes_;
            }
            scheduling_module_->setup_memory(slice);
        }

        allocate_buffers();

        std::vector<float> mean;
        std::vector<float> std;
        std::vector<float> weights;
        float bias = 0.0f;
        bool loaded = false;

        if (!config_.model_path.empty()) {
            auto dot = config_.model_path.find_last_of('.');
            std::string ext = (dot == std::string::npos) ? "" : config_.model_path.substr(dot);
            if (ext == ".txt") {
                loaded = load_model_file(config_.model_path, mean, std, weights, bias);
            }
        }

        if (!loaded) {
            load_default_model(mean, std, weights, bias);
        }

        if (!scheduling_module_->set_model(mean, std, weights, bias)) {
            throw std::runtime_error("Failed to configure scheduling model");
        }

        return true;
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to initialize user scheduling pipeline: %s\n", e.what());
        return false;
    }
}

bool UserSchedulingPipeline::process_scheduling(
    const std::vector<UeFeatures>& ues,
    std::vector<int>& scheduled_ue_ids,
    std::vector<float>* scores_out,
    cudaStream_t stream
) {
    if (!scheduling_module_) {
        fprintf(stderr, "Scheduling module not initialized\n");
        return false;
    }

    if (ues.size() != static_cast<size_t>(config_.scheduling_params.num_ues)) {
        fprintf(stderr, "Expected %d UEs, received %zu\n",
                config_.scheduling_params.num_ues, ues.size());
        return false;
    }

    if (config_.num_scheduled <= 0 ||
        config_.num_scheduled > config_.scheduling_params.num_ues) {
        fprintf(stderr, "Invalid num_scheduled configuration\n");
        return false;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const int num_features = config_.scheduling_params.num_features;
    for (size_t i = 0; i < ues.size(); ++i) {
        size_t base = i * static_cast<size_t>(num_features);
        h_features_[base + 0] = ues[i].sinr_db;
        h_features_[base + 1] = ues[i].buffer_bytes;
        h_features_[base + 2] = ues[i].avg_rate_mbps;
        h_features_[base + 3] = ues[i].qos_priority;
        h_features_[base + 4] = ues[i].harq_pending;
        for (int f = 5; f < num_features; ++f) {
            h_features_[base + f] = 0.0f;
        }
    }

    size_t features_bytes = ues.size() * static_cast<size_t>(num_features) * sizeof(float);
    cudaMemcpyAsync(d_features_, h_features_, features_bytes, cudaMemcpyHostToDevice, stream);

    std::vector<framework::pipeline::PortInfo> inputs(1);
    inputs[0].name = "ue_features";
    inputs[0].tensors.resize(1);
    inputs[0].tensors[0].device_ptr = d_features_;
    inputs[0].tensors[0].tensor_info = framework::tensor::TensorInfo(
        framework::tensor::NvDataType::TensorR32F,
        {static_cast<size_t>(config_.scheduling_params.num_ues),
         static_cast<size_t>(config_.scheduling_params.num_features)}
    );

    scheduling_module_->set_inputs(inputs);
    scheduling_module_->configure_io({}, stream);
    scheduling_module_->execute(stream);

    auto outputs = scheduling_module_->get_outputs();
    if (outputs.empty() || outputs[0].tensors.empty()) {
        fprintf(stderr, "Scheduling module produced no output\n");
        return false;
    }

    cudaMemcpyAsync(
        h_scores_,
        outputs[0].tensors[0].device_ptr,
        ues.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA stream synchronization failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    std::vector<std::pair<float, int>> scored;
    scored.reserve(ues.size());
    for (size_t i = 0; i < ues.size(); ++i) {
        scored.emplace_back(h_scores_[i], ues[i].ue_id);
    }

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    scheduled_ue_ids.clear();
    scheduled_ue_ids.reserve(config_.num_scheduled);
    for (int i = 0; i < config_.num_scheduled; ++i) {
        scheduled_ue_ids.push_back(scored[i].second);
    }

    if (scores_out) {
        scores_out->assign(h_scores_, h_scores_ + ues.size());
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    update_metrics(duration.count() / 1000.0, ues.size());

    return true;
}

UserSchedulingPipeline::PerformanceMetrics UserSchedulingPipeline::get_performance_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void UserSchedulingPipeline::allocate_buffers() {
    size_t features_bytes =
        static_cast<size_t>(config_.scheduling_params.num_ues) *
        static_cast<size_t>(config_.scheduling_params.num_features) *
        sizeof(float);
    size_t scores_bytes = static_cast<size_t>(config_.scheduling_params.num_ues) * sizeof(float);

    cudaMallocHost(&h_features_, features_bytes);
    cudaMallocHost(&h_scores_, scores_bytes);
    cudaMalloc(&d_features_, features_bytes);
}

void UserSchedulingPipeline::deallocate_buffers() {
    if (h_features_) {
        cudaFreeHost(h_features_);
        h_features_ = nullptr;
    }
    if (h_scores_) {
        cudaFreeHost(h_scores_);
        h_scores_ = nullptr;
    }
    if (d_features_) {
        cudaFree(d_features_);
        d_features_ = nullptr;
    }
    if (d_module_tensor_) {
        cudaFree(d_module_tensor_);
        d_module_tensor_ = nullptr;
    }
    if (static_desc_cpu_) {
        cudaFreeHost(static_desc_cpu_);
        static_desc_cpu_ = nullptr;
    }
    if (static_desc_gpu_) {
        cudaFree(static_desc_gpu_);
        static_desc_gpu_ = nullptr;
    }
    if (dynamic_desc_cpu_) {
        cudaFreeHost(dynamic_desc_cpu_);
        dynamic_desc_cpu_ = nullptr;
    }
    if (dynamic_desc_gpu_) {
        cudaFree(dynamic_desc_gpu_);
        dynamic_desc_gpu_ = nullptr;
    }
    module_tensor_bytes_ = 0;
    static_desc_bytes_ = 0;
    dynamic_desc_bytes_ = 0;
}

void UserSchedulingPipeline::update_metrics(double processing_time_ms, size_t num_ues) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.total_processed_frames++;
    metrics_.avg_processing_time_ms =
        (metrics_.avg_processing_time_ms * (metrics_.total_processed_frames - 1) + processing_time_ms) /
        metrics_.total_processed_frames;
    metrics_.peak_processing_time_ms = std::max(metrics_.peak_processing_time_ms, processing_time_ms);
    metrics_.throughput_ues_per_ms = num_ues / std::max(processing_time_ms, 1e-6);
}

bool UserSchedulingPipeline::load_model_file(
    const std::string& path,
    std::vector<float>& mean,
    std::vector<float>& std,
    std::vector<float>& weights,
    float& bias
) const {
    std::ifstream file(path);
    if (!file) {
        fprintf(stderr, "Failed to open model file: %s\n", path.c_str());
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        auto parse_list = [](const std::string& text) {
            std::vector<float> out;
            std::stringstream ss(text);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    out.push_back(std::stof(token));
                }
            }
            return out;
        };

        if (key == "mean") {
            mean = parse_list(value);
        } else if (key == "std") {
            std = parse_list(value);
        } else if (key == "weights") {
            weights = parse_list(value);
        } else if (key == "bias") {
            bias = std::stof(value);
        }
    }

    bool valid = mean.size() == std.size() &&
                 mean.size() == weights.size() &&
                 mean.size() == static_cast<size_t>(config_.scheduling_params.num_features);
    if (!valid) {
        fprintf(stderr, "Model file has invalid dimensions\n");
    }
    return valid;
}

void UserSchedulingPipeline::load_default_model(
    std::vector<float>& mean,
    std::vector<float>& std,
    std::vector<float>& weights,
    float& bias
) const {
    mean.assign(config_.scheduling_params.num_features, 0.0f);
    std.assign(config_.scheduling_params.num_features, 1.0f);
    weights = {0.8f, 0.6f, -0.5f, 0.4f, 0.2f};
    if (weights.size() < static_cast<size_t>(config_.scheduling_params.num_features)) {
        weights.resize(config_.scheduling_params.num_features, 0.0f);
    }
    bias = 0.0f;
}

} // namespace user_scheduling
