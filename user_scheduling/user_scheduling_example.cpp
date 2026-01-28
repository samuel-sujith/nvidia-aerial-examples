// User Scheduling Example - NVIDIA Aerial Framework
#include "user_scheduling_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<user_scheduling::UeFeatures> generate_ues(int num_ues, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> sinr_dist(10.0f, 6.0f);
    std::uniform_real_distribution<float> rate_dist(1.0f, 200.0f);
    std::uniform_real_distribution<float> qos_dist(0.5f, 1.0f);
    std::uniform_int_distribution<int> buffer_dist(10'000, 5'000'000);
    std::bernoulli_distribution harq_dist(0.2);

    std::vector<user_scheduling::UeFeatures> ues;
    ues.reserve(static_cast<size_t>(num_ues));
    for (int i = 0; i < num_ues; ++i) {
        user_scheduling::UeFeatures ue;
        ue.ue_id = i;
        ue.sinr_db = std::max(-5.0f, std::min(30.0f, sinr_dist(rng)));
        ue.buffer_bytes = static_cast<float>(buffer_dist(rng));
        ue.avg_rate_mbps = rate_dist(rng);
        ue.qos_priority = qos_dist(rng);
        ue.harq_pending = harq_dist(rng) ? 1.0f : 0.0f;
        ues.push_back(ue);
    }
    return ues;
}

} // namespace

int main(int argc, char** argv) {
    int num_ues = 16;
    int num_scheduled = 4;
    int seed = 42;
    std::string model_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--num-ues" && i + 1 < argc) {
            num_ues = std::stoi(argv[++i]);
        } else if (arg == "--num-scheduled" && i + 1 < argc) {
            num_scheduled = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (num_scheduled <= 0 || num_ues <= 0 || num_scheduled > num_ues) {
        std::cerr << "Invalid UE or scheduling counts.\n";
        return 1;
    }

    user_scheduling::UserSchedulingPipeline::PipelineConfig config;
    config.scheduling_params.num_ues = num_ues;
    config.scheduling_params.num_features = 5;
    config.num_scheduled = num_scheduled;
    config.model_path = model_path;
    config.enable_profiling = true;

    user_scheduling::UserSchedulingPipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize user scheduling pipeline\n";
        return 1;
    }

    auto ues = generate_ues(num_ues, seed);
    std::vector<int> scheduled_ids;
    std::vector<float> scores;

    if (!pipeline.process_scheduling(ues, scheduled_ids, &scores)) {
        std::cerr << "Scheduling processing failed\n";
        return 1;
    }

    std::vector<std::pair<float, user_scheduling::UeFeatures>> scored;
    scored.reserve(ues.size());
    for (size_t i = 0; i < ues.size(); ++i) {
        scored.emplace_back(scores[i], ues[i]);
    }
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "\n=== Scheduling Decision ===\n";
    std::cout << "Selected " << num_scheduled << " UEs out of " << num_ues << "\n\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < num_scheduled; ++i) {
        const auto& ue = scored[i].second;
        std::cout << "UE " << ue.ue_id
                  << " | score=" << scored[i].first
                  << " | sinr_db=" << ue.sinr_db
                  << " | buffer_bytes=" << ue.buffer_bytes
                  << " | avg_rate_mbps=" << ue.avg_rate_mbps
                  << " | qos=" << ue.qos_priority
                  << " | harq=" << ue.harq_pending
                  << "\n";
    }

    auto metrics = pipeline.get_performance_metrics();
    std::cout << "\nPerformance:\n";
    std::cout << "  Avg processing time: " << metrics.avg_processing_time_ms << " ms\n";
    std::cout << "  Peak processing time: " << metrics.peak_processing_time_ms << " ms\n";
    std::cout << "  Throughput: " << metrics.throughput_ues_per_ms << " UEs/ms\n";

    return 0;
}
