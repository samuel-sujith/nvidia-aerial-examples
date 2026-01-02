#include "modulation_pipeline.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>

using namespace modulation;

/// Demonstrate basic modulation pipeline usage
void demonstrate_basic_modulation() {
    std::cout << "=== Basic Modulation Pipeline Demo ===\n";
    
    // Create configuration
    ModulationPipelineConfig config;
    config.modulation_order = ModulationScheme::QAM_16;
    config.max_batch_size = 1000;
    config.enable_cuda_graphs = true;
    
    // Create pipeline
    auto pipeline = ModulationPipelineFactory::create_pipeline(config);
    
    // Setup pipeline
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup modulation pipeline\n";
        return;
    }
    
    // Generate test data
    std::vector<uint8_t> input_bits = {
        0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
        1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0
    };
    
    std::vector<std::complex<float>> output_symbols;
    
    // Execute modulation
    auto result = pipeline->modulate_bits(input_bits, output_symbols);
    
    if (result.is_success()) {
        std::cout << "Modulation successful!\n";
        std::cout << "Input bits: " << input_bits.size() << "\n";
        std::cout << "Output symbols: " << output_symbols.size() << "\n";
        
        // Display first few symbols
        std::cout << "First 4 symbols:\n";
        for (size_t i = 0; i < std::min(size_t(4), output_symbols.size()); ++i) {
            std::cout << "  Symbol " << i << ": (" 
                     << output_symbols[i].real() << ", " 
                     << output_symbols[i].imag() << ")\n";
        }
        
        // Show performance stats
        auto stats = pipeline->get_modulation_stats();
        std::cout << "Throughput: " << stats.average_throughput_msps() << " Msps\n";
        std::cout << "Latency: " << stats.average_latency_us() << " μs\n";
        
    } else {
        std::cerr << "Modulation failed: " << result.message << "\n";
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate batch processing performance
void demonstrate_batch_processing() {
    std::cout << "=== Batch Processing Demo ===\n";
    
    // High performance configuration
    auto config = ModulationPipelineFactory::get_high_performance_config(ModulationScheme::QAM_64);
    auto pipeline = ModulationPipelineFactory::create_pipeline(config);
    
    ::framework::pipeline::PipelineSpec spec;
    if (!pipeline->setup(spec)) {
        std::cerr << "Failed to setup pipeline\n";
        return;
    }
    
    // Generate large batch of data
    const size_t num_batches = 10;
    const size_t bits_per_batch = 10000;
    
    std::vector<std::vector<uint8_t>> input_batches(num_batches);
    std::vector<std::vector<std::complex<float>>> output_batches(num_batches);
    
    // Fill with random-like data
    for (size_t batch = 0; batch < num_batches; ++batch) {
        input_batches[batch].resize(bits_per_batch);
        for (size_t i = 0; i < bits_per_batch; ++i) {
            input_batches[batch][i] = (i * 17 + batch * 23) % 2;
        }
    }
    
    // Process batch
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = pipeline->modulate_batch(input_batches, output_batches);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (result.is_success()) {
        size_t total_symbols = 0;
        for (const auto& batch : output_batches) {
            total_symbols += batch.size();
        }
        
        double throughput_msps = static_cast<double>(total_symbols) / duration.count();
        
        std::cout << "Batch processing successful!\n";
        std::cout << "Processed " << num_batches << " batches\n";
        std::cout << "Total symbols: " << total_symbols << "\n";
        std::cout << "Processing time: " << duration.count() << " μs\n";
        std::cout << "Throughput: " << throughput_msps << " Msps\n";
        
        // Show detailed stats
        auto stats = pipeline->get_modulation_stats();
        std::cout << "Pipeline stats:\n";
        std::cout << "  Total symbols processed: " << stats.total_symbols_processed << "\n";
        std::cout << "  Total batches processed: " << stats.total_batches_processed << "\n";
        std::cout << "  Average latency: " << stats.average_latency_us() << " μs\n";
        
    } else {
        std::cerr << "Batch processing failed: " << result.message << "\n";
    }
    
    pipeline->teardown();
    std::cout << "\n";
}

/// Demonstrate different modulation orders
void demonstrate_modulation_orders() {
    std::cout << "=== Modulation Orders Demo ===\n";
    
    std::vector<ModulationScheme> orders = {
        ModulationScheme::QPSK,
        ModulationScheme::QAM_16,
        ModulationScheme::QAM_64,
        ModulationScheme::QAM_256
    };
    
    std::vector<std::string> order_names = {
        "QPSK", "16-QAM", "64-QAM", "256-QAM"
    };
    
    // Test data - ensure it's compatible with all modulation orders
    std::vector<uint8_t> input_bits(1000);
    for (size_t i = 0; i < input_bits.size(); ++i) {
        input_bits[i] = i % 2;
    }
    
    for (size_t i = 0; i < orders.size(); ++i) {
        std::cout << "\nTesting " << order_names[i] << ":\n";
        
        auto config = ModulationPipelineFactory::get_default_config(orders[i]);
        auto pipeline = ModulationPipelineFactory::create_pipeline(config);
        
        ::framework::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "  Failed to setup pipeline\n";
            continue;
        }
        
        std::vector<std::complex<float>> output_symbols;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto result = pipeline->modulate_bits(input_bits, output_symbols);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        if (result.is_success()) {
            double throughput = static_cast<double>(output_symbols.size()) / duration.count();
            
            std::cout << "  Input bits: " << input_bits.size() << "\n";
            std::cout << "  Output symbols: " << output_symbols.size() << "\n";
            std::cout << "  Processing time: " << duration.count() << " μs\n";
            std::cout << "  Throughput: " << throughput << " Msps\n";
            
            // Show constellation properties
            if (output_symbols.size() > 0) {
                float avg_power = 0.0f;
                for (const auto& symbol : output_symbols) {
                    avg_power += std::norm(symbol);
                }
                avg_power /= output_symbols.size();
                std::cout << "  Average symbol power: " << avg_power << "\n";
            }
            
        } else {
            std::cerr << "  Modulation failed: " << result.message << "\n";
        }
        
        pipeline->teardown();
    }
    
    std::cout << "\n";
}

/// Performance comparison between configurations
void demonstrate_performance_comparison() {
    std::cout << "=== Performance Comparison Demo ===\n";
    
    std::vector<std::pair<std::string, ModulationPipelineConfig>> configs = {
        {"Default", ModulationPipelineFactory::get_default_config(ModulationScheme::QAM_16)},
        {"High Performance", ModulationPipelineFactory::get_high_performance_config(ModulationScheme::QAM_16)},
        {"Low Latency", ModulationPipelineFactory::get_low_latency_config(ModulationScheme::QAM_16)}
    };
    
    // Test data
    std::vector<uint8_t> input_bits(5000);
    for (size_t i = 0; i < input_bits.size(); ++i) {
        input_bits[i] = (i * 7) % 2;
    }
    
    const int num_iterations = 10;
    
    for (const auto& [config_name, config] : configs) {
        std::cout << "\nTesting " << config_name << " configuration:\n";
        
        auto pipeline = ModulationPipelineFactory::create_pipeline(config);
        
        ::framework::pipeline::PipelineSpec spec;
        if (!pipeline->setup(spec)) {
            std::cerr << "  Failed to setup pipeline\n";
            continue;
        }
        
        // Warmup
        std::vector<std::complex<float>> dummy_output;
        pipeline->modulate_bits(input_bits, dummy_output);
        
        // Benchmark
        std::vector<double> execution_times;
        
        for (int iter = 0; iter < num_iterations; ++iter) {
            std::vector<std::complex<float>> output_symbols;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = pipeline->modulate_bits(input_bits, output_symbols);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            if (result.is_success()) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                execution_times.push_back(duration.count());
            }
        }
        
        if (!execution_times.empty()) {
            double avg_time = 0.0;
            double min_time = execution_times[0];
            double max_time = execution_times[0];
            
            for (double time : execution_times) {
                avg_time += time;
                min_time = std::min(min_time, time);
                max_time = std::max(max_time, time);
            }
            avg_time /= execution_times.size();
            
            double throughput = static_cast<double>(dummy_output.size()) / avg_time;
            
            std::cout << "  Average time: " << avg_time << " μs\n";
            std::cout << "  Min time: " << min_time << " μs\n";
            std::cout << "  Max time: " << max_time << " μs\n";
            std::cout << "  Throughput: " << throughput << " Msps\n";
            
            auto stats = pipeline->get_modulation_stats();
            std::cout << "  Total processed: " << stats.total_symbols_processed << " symbols\n";
        }
        
        pipeline->teardown();
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
        demonstrate_performance_comparison();
        
        std::cout << "All demos completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}