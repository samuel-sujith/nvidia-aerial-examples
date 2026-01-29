/*
 * Path Loss Prediction Example (Pipeline API)
 */

#include "path_loss_prediction_pipeline.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::vector<float> generate_synthetic_batch(
    int batch_size,
    int num_features,
    float carrier_freq_ghz) {
    std::vector<float> features(static_cast<size_t>(batch_size) * num_features, 0.0f);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_km(0.05f, 1.5f);
    std::uniform_real_distribution<float> speed(0.0f, 30.0f);
    std::uniform_real_distribution<float> clutter(0.0f, 1.0f);

    for (int i = 0; i < batch_size; ++i) {
        float d_km = dist_km(rng);
        float logd = std::log10(d_km);
        float freq = carrier_freq_ghz;
        float sinr = 10.0f - 15.0f * d_km + clutter(rng) * 2.0f;
        float rsrp = -70.0f - 20.0f * logd + clutter(rng) * 1.5f;
        float shadow = clutter(rng) * 6.0f;
        float speed_mps = speed(rng);
        float load = clutter(rng);
        float buffer = clutter(rng);

        size_t base = static_cast<size_t>(i) * num_features;
        features[base + 0] = rsrp;
        features[base + 1] = sinr;
        features[base + 2] = d_km * 1000.0f;
        features[base + 3] = freq;
        features[base + 4] = shadow;
        features[base + 5] = speed_mps;
        features[base + 6] = load;
        features[base + 7] = buffer;
    }

    return features;
}

} // namespace

int main(int argc, char** argv) {
    path_loss_prediction::PathLossPredictionPipeline::PipelineConfig config;
    config.model_params.batch_size = 32;
    config.model_params.num_features = 8;
    config.model_params.hidden_size = 16;
    config.model_params.model_type = path_loss_prediction::ModelType::TensorRT_MLP;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--xgboost") {
            config.model_params.model_type = path_loss_prediction::ModelType::XGBoostStyle;
        } else if (arg == "--mlp") {
            config.model_params.model_type = path_loss_prediction::ModelType::TensorRT_MLP;
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_params.model_path = argv[++i];
        }
    }

    path_loss_prediction::PathLossPredictionPipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize path loss prediction pipeline\n";
        return 1;
    }

    auto features = generate_synthetic_batch(
        config.model_params.batch_size,
        config.model_params.num_features,
        3.5f);

    std::vector<float> path_loss;
    if (!pipeline.predict(features, path_loss)) {
        std::cerr << "Path loss prediction failed\n";
        return 1;
    }

    std::cout << "Predicted path loss (dB) for first 5 UEs:\n";
    for (size_t i = 0; i < std::min<size_t>(5, path_loss.size()); ++i) {
        std::cout << "  UE " << i << ": " << path_loss[i] << " dB\n";
    }

    return 0;
}
