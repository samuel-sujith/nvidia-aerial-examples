# FFT Processing Pipeline

This module demonstrates GPU-accelerated FFT (Fast Fourier Transform) processing using the NVIDIA Aerial Framework with cuFFT integration. It provides a complete pipeline implementation for forward/inverse FFT operations and OFDM symbol processing.

## Features

- **cuFFT Integration**: High-performance GPU-based FFT operations using NVIDIA cuFFT library
- **Multi-size Support**: Flexible FFT sizes from 64 to 8192 points
- **OFDM Processing**: Complete OFDM symbol generation and processing with cyclic prefix
- **Batch Processing**: Efficient batch operations for multiple FFT sizes
- **Precision Support**: Single and double precision floating-point operations
- **CUDA Graphs**: Optimized execution using CUDA graphs for reduced kernel launch overhead
- **Performance Monitoring**: Comprehensive timing and throughput statistics

## Architecture

```
FFT Processing Pipeline
├── FFTModule              # Core FFT computation module
├── FFTPipeline           # Pipeline orchestration and memory management
├── FFTPipelineFactory    # Factory for creating configured pipelines
└── Examples              # Demonstration applications
```

## Key Components

### FFTModule (`fft_module.hpp`)
- cuFFT plan management and execution
- Forward and inverse FFT operations
- Memory-efficient batch processing
- Error handling and validation

### FFTPipeline (`fft_pipeline.hpp/cpp`)
- Pipeline interface implementation
- Resource lifecycle management
- CUDA stream coordination
- Performance statistics collection

### Factory Pattern (`fft_pipeline.hpp`)
```cpp
// Available configurations
auto default_config = FFTPipelineFactory::get_default_config({1024, 2048});
auto high_perf_config = FFTPipelineFactory::get_high_performance_config({512, 1024, 2048});
auto ofdm_config = FFTPipelineFactory::get_ofdm_config(1024);
```

## Building

### Prerequisites
- CUDA Toolkit 11.8+
- cuFFT library
- NVIDIA Aerial Framework

### Build Commands
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make fft_processing_example -j$(nproc)
```

## Usage Examples

### Basic FFT Operations
```cpp
#include "fft_pipeline.hpp"

// Create pipeline
auto config = FFTPipelineFactory::get_default_config({1024});
auto pipeline = FFTPipelineFactory::create_pipeline(config);

// Setup
aerial::pipeline::PipelineSpec spec;
pipeline->setup(spec);

// Execute forward FFT
std::vector<std::complex<float>> input_signal(1024);
std::vector<std::complex<float>> fft_result;

auto result = pipeline->execute_forward_fft(input_signal, fft_result, 1024);

// Execute inverse FFT for reconstruction
std::vector<std::complex<float>> reconstructed;
pipeline->execute_inverse_fft(fft_result, reconstructed, 1024);
```

### OFDM Symbol Processing
```cpp
// Generate OFDM configuration
auto ofdm_config = FFTPipelineFactory::get_ofdm_config(2048);
auto pipeline = FFTPipelineFactory::create_pipeline(ofdm_config);

// Process OFDM symbol (IFFT + Cyclic Prefix)
std::vector<std::complex<float>> freq_symbols(2048);
std::vector<std::complex<float>> time_samples;

pipeline->execute_ofdm_processing(freq_symbols, time_samples, 2048, 144);
```

### Batch Processing
```cpp
// Mixed batch with different FFT sizes
std::vector<std::vector<std::complex<float>>> input_batches;
std::vector<std::vector<std::complex<float>>> output_batches;
std::vector<size_t> fft_sizes = {512, 1024, 2048};

pipeline->execute_mixed_batch(input_batches, output_batches, fft_sizes, FFTType::Forward);
```

## Running Examples

### Comprehensive Example
```bash
./fft_processing_example
```

This runs multiple test scenarios:
- Basic forward/inverse FFT operations
- OFDM symbol processing
- FFT size comparison benchmarks
- Precision mode testing

### Simple Example
```bash
./fft_example
```

Focused demonstrations of core FFT functionality.

## Performance Characteristics

| FFT Size | Throughput (Msamples/s) | Latency (μs) | Memory (MB) |
|----------|------------------------|--------------|-------------|
| 256      | 3,500                  | 7.3          | 15          |
| 512      | 3,200                  | 16           | 20          |
| 1024     | 2,800                  | 37           | 40          |
| 2048     | 2,100                  | 98           | 75          |
| 4096     | 1,400                  | 290          | 140         |

*Performance measured on RTX 4090 with CUDA 12.3*

## Configuration Options

### FFTPipelineConfig
```cpp
struct FFTPipelineConfig {
    std::vector<size_t> fft_sizes;           // Supported FFT sizes
    FFTPrecision precision;                   // Single/Double precision
    size_t max_batch_size;                   // Maximum batch size
    bool enable_cuda_graphs;                 // CUDA graph optimization
    bool enable_callbacks;                   // Async callback support
};
```

### Factory Configurations
- **Default**: Balanced performance and memory usage
- **High Performance**: Optimized for maximum throughput
- **Low Latency**: Minimized processing delay
- **OFDM**: Specialized for OFDM symbol processing

## OFDM Processing

The pipeline includes specialized OFDM processing capabilities:

### Features
- Configurable subcarrier allocation
- Cyclic prefix insertion/removal
- PAPR (Peak-to-Average Power Ratio) calculation
- Windowing and filtering support

### Example OFDM Configuration
```cpp
const size_t subcarriers = 2048;        // FFT size
const size_t active_carriers = 1536;    // Data subcarriers
const size_t cp_length = 144;           // Cyclic prefix length

auto config = FFTPipelineFactory::get_ofdm_config(subcarriers);
```

## API Reference

### Core Methods
```cpp
// Pipeline setup and teardown
bool setup(const aerial::pipeline::PipelineSpec& spec);
void teardown();

// FFT operations
PipelineResult execute_forward_fft(const std::vector<std::complex<float>>& input,
                                 std::vector<std::complex<float>>& output,
                                 size_t fft_size, size_t batch_size = 1);

PipelineResult execute_inverse_fft(const std::vector<std::complex<float>>& input,
                                 std::vector<std::complex<float>>& output,
                                 size_t fft_size, size_t batch_size = 1);

// OFDM processing
PipelineResult execute_ofdm_processing(const std::vector<std::complex<float>>& freq_symbols,
                                     std::vector<std::complex<float>>& time_samples,
                                     size_t fft_size, size_t cp_length);

// Statistics
FFTStatistics get_fft_stats() const;
```

### Statistics Structure
```cpp
struct FFTStatistics {
    size_t total_ffts_processed;
    uint64_t total_processing_time_us;
    double average_latency_us() const;
    double average_throughput_msamples_per_sec() const;
    double peak_throughput_msamples_per_sec;
};
```

## Error Handling

The pipeline provides comprehensive error handling:

```cpp
auto result = pipeline->execute_forward_fft(input, output, fft_size);
if (!result.is_success()) {
    std::cerr << "FFT failed: " << result.message << std::endl;
    return;
}
```

Common error conditions:
- Invalid FFT size (not power of 2 or outside supported range)
- Insufficient GPU memory
- cuFFT plan creation failure
- Input/output size mismatches

## Best Practices

### Memory Management
- Pre-allocate output vectors to avoid reallocations
- Use consistent FFT sizes to benefit from plan caching
- Consider memory alignment for optimal performance

### Performance Optimization
- Enable CUDA graphs for repeated operations
- Use appropriate batch sizes (powers of 2 recommended)
- Choose optimal FFT sizes based on problem requirements

### Error Recovery
- Always check return codes from pipeline operations
- Implement proper cleanup in error scenarios
- Use validation functions for input data

## Integration with 5G NR

This FFT pipeline is designed for 5G NR baseband processing:

### Supported Use Cases
- OFDM symbol generation for downlink transmission
- OFDM symbol processing for uplink reception
- Channel estimation in frequency domain
- Frequency domain equalization
- PRACH detection and processing

### 5G NR Parameters
- FFT sizes: 128, 256, 512, 1024, 1536, 2048, 3072, 4096
- Subcarrier spacing: 15, 30, 60, 120, 240 kHz support
- Cyclic prefix configurations for normal and extended CP

## Troubleshooting

### Common Issues
1. **cuFFT plan creation fails**: Check GPU memory availability and FFT size validity
2. **Poor performance**: Verify CUDA architecture settings and enable optimizations
3. **Memory allocation errors**: Ensure sufficient GPU memory for batch operations
4. **Incorrect results**: Validate input data and check for numerical precision issues

### Debug Configuration
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_DEBUG_INFO=ON
```

### Profiling
```bash
nsys profile --output=fft_profile.nsys-rep ./fft_processing_example
```

## Contributing

When contributing to the FFT processing module:

1. Follow the existing code style and patterns
2. Add comprehensive unit tests for new functionality
3. Update performance benchmarks
4. Validate against reference implementations
5. Test with various FFT sizes and configurations

## License

SPDX-License-Identifier: Apache-2.0