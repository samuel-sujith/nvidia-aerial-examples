

#include "ai_rx_pipeline.hpp"
#include <iostream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    using namespace ai_rx_example;

    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
        std::cout << "Using model path: " << model_path << std::endl;
    } else {
        std::cout << "No model path provided. Using default CUDA kernel for inference." << std::endl;
    }

    // Example: generate dummy Rx symbols (real, imag pairs)
    std::vector<std::pair<float, float>> rx_symbols = {
        {0.7f, -0.2f}, {1.1f, 0.9f}, {-0.8f, 0.3f}, {0.0f, -1.0f}
    };

    // Configure and initialize pipeline
    AiRxPipeline::PipelineConfig config;
    config.module_id = "ai_rx_module";
    config.ai_rx_params.num_symbols = rx_symbols.size();
    if (!model_path.empty()) {
        config.ai_rx_params.model_path = model_path;
    }
    AiRxPipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize AI Rx pipeline" << std::endl;
        return 1;
    }

    // Run AI Rx processing
    std::vector<float> rx_bits;
    if (!pipeline.process_rx(rx_symbols, rx_bits)) {
        std::cerr << "AI Rx processing failed" << std::endl;
        return 1;
    }

    // Print results
    std::cout << "AI Rx Example: Inference results\n";
    for (size_t i = 0; i < rx_bits.size(); ++i) {
        std::cout << "Symbol " << i << ": " << rx_bits[i] << std::endl;
    }
    return 0;
}
