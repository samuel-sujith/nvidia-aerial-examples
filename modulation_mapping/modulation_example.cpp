#include "modulation_pipeline.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace framework::examples;

/// Demonstrate basic modulation pipeline usage
void demonstrate_basic_modulation() {
    std::cout << "=== Basic Modulation Pipeline Demo ===\n";
    
    // Create configuration
    ModulationPipelineConfig config;
    config.modulation_order = ModulationScheme::QAM_16;
    config.max_batch_size = 1000;
    
    // Create pipeline
    auto pipeline = ModulationPipelineFactory::create(config);
    
    if (pipeline) {
        std::cout << "Pipeline created successfully\n";
        
        // Generate test data
        std::vector<uint8_t> input_bits = {
            0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
            1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0
        };
        
        std::cout << "Input bits: " << input_bits.size() << std::endl;
        std::cout << "Modulation order: QAM_16" << std::endl;
        
        // In a real implementation, this would call pipeline methods
        // For now, just demonstrate the pipeline creation and setup
        std::cout << "Pipeline ready for processing" << std::endl;
    } else {
        std::cerr << "Failed to create modulation pipeline\n";
    }
    
    std::cout << "\n";
}

/// Demonstrate batch processing performance
void demonstrate_batch_processing() {
    std::cout << "=== Batch Processing Demo ===\n";
    
    // High performance configuration  
    ModulationPipelineConfig config;
    config.modulation_order = ModulationScheme::QAM_64;
    config.max_batch_size = 2000;
    
    auto pipeline = ModulationPipelineFactory::create(config);
    
    if (pipeline) {
        std::cout << "Batch pipeline created successfully\n";
        
        // Generate large batch of test data
        const size_t num_batches = 10;
        const size_t bits_per_batch = 1000;
        
        std::cout << "Processing " << num_batches << " batches of " << bits_per_batch << " bits each\n";
        
        // Simulate batch processing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // In a real implementation, this would process actual data
        for (size_t batch = 0; batch < num_batches; ++batch) {
            std::vector<uint8_t> batch_data(bits_per_batch);
            // Fill with test pattern
            for (size_t i = 0; i < bits_per_batch; ++i) {
                batch_data[i] = (i * 17 + batch * 23) % 2;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "Batch processing completed\n";
        std::cout << "Processing time: " << duration.count() << " μs\n";
        
    } else {
        std::cerr << "Failed to setup batch pipeline\n";
    }
    
    std::cout << "\n";
}

/// Demonstrate different modulation orders
void demonstrate_modulation_orders() {
    std::cout << "=== Modulation Orders Demo ===\n";
    
    std::vector<ModulationScheme> orders = {
        ModulationScheme::QPSK,
        ModulationScheme::QAM_16,
        ModulationScheme::QAM_64
    };
    
    std::vector<std::string> order_names = {
        "QPSK", "16-QAM", "64-QAM"
    };
    
    // Test data
    std::vector<uint8_t> input_bits(64);  
    for (size_t i = 0; i < input_bits.size(); ++i) {
        input_bits[i] = i % 2;
    }
    
    for (size_t i = 0; i < orders.size(); ++i) {
        std::cout << "\nTesting " << order_names[i] << ":\n";
        
        ModulationPipelineConfig config;
        config.modulation_order = orders[i];
        config.max_batch_size = 1000;
        
        auto pipeline = ModulationPipelineFactory::create(config);
        
        if (pipeline) {
            std::cout << "  Pipeline created successfully\n";
            std::cout << "  Input bits: " << input_bits.size() << "\n";
            std::cout << "  Modulation order: " << order_names[i] << "\n";
        } else {
            std::cerr << "  Failed to setup pipeline\n";
        }
    }
    
    std::cout << "\n";
}

int main() {
    try {
        std::cout << "NVIDIA Aerial Framework - Modulation Pipeline Examples\n";
        std::cout << "======================================================\n\n";
        
        demonstrate_basic_modulation();
        demonstrate_batch_processing();
        demonstrate_modulation_orders();
        
        std::cout << "All demos completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}