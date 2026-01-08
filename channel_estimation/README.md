# Channel Estimation Example


This example demonstrates wireless channel estimation using the NVIDIA Aerial Framework. The implementation showcases real-world channel estimation algorithms including Least Squares (LS), Linear Interpolation, and Machine Learning (ML) models (via TensorRT) with CUDA acceleration for 5G NR and LTE systems.

## Overview

Channel estimation is a fundamental process in wireless communication systems that determines the channel response between transmitter and receiver. Accurate channel knowledge is essential for:
- **Coherent Detection**: Enabling optimal symbol detection
- **Equalization**: Compensating for channel impairments
- **Beamforming**: Optimizing antenna array performance  
- **Link Adaptation**: Adjusting transmission parameters based on channel quality
- **MIMO Processing**: Enabling spatial multiplexing and diversity


This example implements:
- **Least Squares (LS) Estimation**: Direct channel estimation from pilot symbols
- **Linear Interpolation**: Time and frequency domain channel interpolation
- **Machine Learning (ML) Estimation**: Channel estimation using a neural network model deployed with TensorRT (optional)
- **CUDA-accelerated Processing**: GPU kernels for high-performance computing
- **Real Framework Integration**: Uses actual Aerial Framework interfaces
## ML Model Support (TensorRT)

The channel estimation pipeline supports an optional ML-based estimator using a neural network model deployed with NVIDIA TensorRT. This enables data-driven channel estimation for advanced use cases.

### Requirements
- TensorRT 8.x+ (with development headers)
- A compatible ONNX or TensorRT engine file for channel estimation
- Build with `-DTENSORRT_AVAILABLE=ON` (or ensure the macro is defined)

### Usage
- Pass the ML model path and select the ML algorithm via command-line or configuration:
  ```bash
  ./channel_estimation_example --algorithm ml_tensorrt --model_path /path/to/model.engine
  ```
- The pipeline will use the ML estimator if TensorRT is available and a valid model is provided. Otherwise, it falls back to classic algorithms.

### Configuration Parameters
Additional parameters for ML estimation:
```
struct ChannelEstimationParams {
    // ...existing fields...
    std::string model_path;      // Path to TensorRT engine file
    int ml_input_size;           // Input size for ML model
    int ml_output_size;          // Output size for ML model
    bool use_fp16;               // Use FP16 precision for inference
    int max_batch_size;          // Max batch size for inference
};
```

### Example Output (ML Estimator)
```
Testing ML (TensorRT) Channel Estimation...
  Processing Time: 0.35 ms
  ML Estimated Channel - Samples: 14336, Avg Power: 1.02, Peak: 2.3
  Channel MSE: 0.061
```


## Architecture

### Channel Model
In OFDM systems, the received signal can be modeled as:
```
Y[k] = H[k] * X[k] + N[k]
```
Where:
- `Y[k]`: Received signal at subcarrier k
- `H[k]`: Channel frequency response at subcarrier k
- `X[k]`: Transmitted signal at subcarrier k  
- `N[k]`: Additive noise at subcarrier k

### Estimation Process
1. **Pilot Extraction**: Extract known reference signals from received data
2. **LS Estimation**: Calculate channel estimates at pilot positions
3. **Interpolation**: Estimate channel at data subcarriers
4. **Smoothing**: Apply filtering to reduce noise effects

## Files Structure

```
channel_estimation/
├── CMakeLists.txt                           # Build configuration
├── README.md                                # This documentation
├── channel_estimation_module.hpp            # Core channel estimation module interface
├── channel_estimation_module.cu             # Channel estimation CUDA implementation
├── channel_estimation_pipeline.hpp          # Pipeline wrapper interface
├── channel_estimation_pipeline.cu           # Pipeline wrapper implementation
└── channel_estimation_example.cpp           # Example application and tests
```

## Key Components


### ChannelEstimator Class
- **Framework Integration**: Inherits from `IModule`, `IAllocationInfoProvider`, `IStreamExecutor`
- **Multi-Algorithm Support**: LS, Interpolation, and ML (TensorRT) estimation
- **Flexible Configuration**: Supports various pilot patterns, interpolation, and ML model selection
- **Memory Management**: Efficient GPU memory allocation and management

### CUDA Kernels
- **ls_channel_estimation_kernel**: Implements LS estimation at pilot positions
- **linear_interpolation_kernel**: Performs linear interpolation between pilot estimates
- **channel_smoothing_kernel**: Applies noise reduction filtering
- **Optimized Memory Access**: Coalesced global memory access patterns


### Pipeline Wrapper  
- **High-level Interface**: Simplified API for channel estimation processing
- **Performance Monitoring**: Built-in profiling and metrics collection
- **Flexible I/O**: Support for both host and device memory operations
- **Parameter Management**: Dynamic parameter updates without reinitialization
- **ML Model Integration**: Seamless selection and execution of ML-based estimator if configured

## Algorithm Details

### Least Squares (LS) Estimation
LS estimation provides the channel estimate by directly computing:
```
Ĥ[k] = Y[k] / X[k]
```
for pilot subcarriers k.

**Advantages:**
- Simple and computationally efficient
- No prior channel knowledge required
- Low latency implementation

**Disadvantages:**
- Susceptible to noise amplification
- No exploitation of channel correlation
- Requires sufficient pilot density

### Linear Interpolation
For data subcarriers between pilots, linear interpolation estimates:
```
Ĥ[k] = Ĥ[k₁] + (k - k₁)/(k₂ - k₁) * (Ĥ[k₂] - Ĥ[k₁])
```
where k₁ and k₂ are adjacent pilot subcarriers.

**Frequency Domain Interpolation:**
- Interpolates between frequency-adjacent pilots
- Suitable for slowly varying channels
- Handles frequency-selective fading

**Time Domain Interpolation:**
- Interpolates between time-adjacent pilots  
- Tracks channel evolution over time
- Handles time-varying channels

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0+
- CMake 3.18+
- NVIDIA Aerial Framework installed
- C++20 compatible compiler
- GPU with Compute Capability 7.5+

### Build Steps
```bash
cd channel_estimation
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run Example
```bash
./channel_estimation_example
```

### Expected Output
```
========================================
   NVIDIA Aerial Channel Estimation    
========================================

Channel Configuration:
  Subcarriers: 1024
  OFDM Symbols: 14
  Pilot Density: 8
  SNR: 20.0 dB

Initializing channel estimation pipeline...
Pipeline initialized successfully!

Generating test data...
Transmitted Pilots - Samples: 1792, Avg Power: 1.0, Peak: 1.0
Channel Response - Samples: 1024, Avg Power: 1.02, Peak: 2.1
Received Pilots - Samples: 1792, Avg Power: 1.05, Peak: 2.3

Testing LS Channel Estimation...
  Processing Time: 0.43 ms
  LS Estimated Channel - Samples: 14336, Avg Power: 1.01, Peak: 2.2
  Channel MSE: 0.089

Testing with Interpolation...
  Processing Time: 0.48 ms  
  Interpolated Channel - Samples: 14336, Avg Power: 1.00, Peak: 2.1
  Interpolated Channel MSE: 0.067

Performance Metrics:
  Average Processing Time: 0.46 ms
  Peak Processing Time: 0.48 ms
  Total Processed Frames: 2
  Throughput: 62.3 MB/s
```

## Performance Characteristics

### Computational Complexity
- **LS Estimation**: O(P) where P = number of pilots
- **Linear Interpolation**: O(N) where N = total subcarriers
- **GPU Parallelization**: Concurrent processing across subcarriers and OFDM symbols

### Memory Requirements
- **Received Pilots**: `num_pilots × num_symbols × sizeof(complex<float>)`
- **Transmitted Pilots**: `num_pilots × num_symbols × sizeof(complex<float>)`
- **Channel Estimates**: `num_subcarriers × num_symbols × sizeof(complex<float>)`
- **Temporary Buffers**: Additional working memory for interpolation

### Typical Performance
- **1024 subcarriers, 14 symbols**: ~0.4-0.5 ms processing time
- **Memory Bandwidth**: ~60-70 GB/s effective throughput  
- **GPU Utilization**: >75% on modern NVIDIA GPUs

## Integration with Aerial Framework

### Framework Interfaces
- **IModule**: Core module lifecycle and identification
- **IAllocationInfoProvider**: Memory requirement specification
- **IStreamExecutor**: CUDA stream-based execution  
- **IMemoryPoolAllocator**: Efficient memory management

### Tensor Operations
- **TensorInfo**: Metadata for multi-dimensional arrays
- **PortInfo**: Input/output data flow specification
- **ModuleMemorySlice**: Framework memory allocation

### Pipeline Integration
- Compatible with Aerial Framework pipeline architecture
- Supports dynamic parameter updates
- Zero-copy operations where possible
- Proper error handling and logging

## Advanced Features

### Pilot Pattern Support
- **Scattered Pilots**: Distributed across time and frequency
- **Block Pilots**: Concentrated pilot blocks
- **Comb Pilots**: Regular frequency spacing
- **Edge Pilots**: Boundary pilot placement

### Noise Reduction
- **Wiener Filtering**: MMSE-based channel smoothing
- **Moving Average**: Simple temporal smoothing
- **Frequency Domain Filtering**: Spectral noise reduction

### Quality Metrics
- **Mean Square Error (MSE)**: Channel estimation accuracy
- **Signal-to-Noise Ratio (SNR)**: Channel quality assessment
- **Correlation Coefficients**: Channel tracking performance

## Use Cases

### 5G NR Systems
- **DMRS-based Estimation**: Demodulation reference signals
- **CSI-RS Processing**: Channel state information reference signals
- **Beam Management**: Channel estimation for beamforming
- **Massive MIMO**: Multi-antenna channel estimation

### LTE Systems  
- **Cell-specific Reference Signals**: CRS-based channel estimation
- **UE-specific Reference Signals**: DMRS processing
- **Positioning Reference Signals**: PRS-based measurements

### Wi-Fi Systems
- **Long Training Fields**: 802.11 preamble processing
- **Pilot Subcarriers**: OFDM pilot-based estimation
- **Channel State Information**: CSI feedback generation

## Future Enhancements


### Advanced Algorithms
- **MMSE Estimation**: Minimum mean square error estimation
- **DFT-based Methods**: Transform domain processing  
- **Kalman Filtering**: Optimal tracking for time-varying channels
- **Machine Learning**: AI-based channel estimation (now supported via TensorRT)

### Optimization Opportunities
- **Batch Processing**: Multiple subframes in parallel
- **Memory Optimization**: Reduce memory footprint
- **Kernel Fusion**: Combine operations for better efficiency
- **Mixed Precision**: Use FP16 for increased throughput

### Extended Functionality
- **Multi-path Estimation**: Detailed delay profile estimation
- **Doppler Estimation**: Velocity and mobility tracking
- **Interference Estimation**: Co-channel interference assessment
- **Channel Prediction**: Exploit temporal correlation for prediction

## Configuration Parameters

### ChannelEstimationParams Structure
```cpp
struct ChannelEstimationParams {
    int num_subcarriers;           // Total OFDM subcarriers
    int num_ofdm_symbols;          // Number of OFDM symbols
    int pilot_density;             // Pilot spacing (every N subcarriers)
    float noise_variance;          // Noise power estimate
    InterpolationMethod interp_method;  // Interpolation algorithm
    bool enable_smoothing;         // Enable noise reduction
    float smoothing_factor;        // Temporal smoothing parameter
};
```

### Interpolation Methods
- **LINEAR**: Linear interpolation between pilots
- **CUBIC**: Cubic spline interpolation (future)
- **DFT**: Transform domain interpolation (future)
- **WIENER**: MMSE-based interpolation (future)

## Troubleshooting

### Common Issues
- **Poor Channel Estimates**: Check pilot SNR and density
- **High MSE**: Increase pilot density or enable smoothing
- **Performance Issues**: Verify GPU utilization and memory bandwidth
- **Compilation Errors**: Check Aerial Framework installation and paths

### Debug Mode
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./channel_estimation_example
```

### Parameter Tuning
- **Pilot Density**: Balance between accuracy and overhead
- **Smoothing Factor**: Adjust based on channel coherence time  
- **Noise Variance**: Accurate estimation improves MMSE performance
- **Interpolation Method**: Choose based on channel characteristics

### Performance Optimization
- Profile memory access patterns with NSight Compute
- Adjust CUDA block sizes for target GPU architecture
- Enable fast math optimizations for production builds
- Monitor thermal throttling and power limits

## CUDA Troubleshooting and Robustness Notes

### Robust Device Pointer and Buffer Management
- The pipeline now includes robust device pointer propagation and defensive checks for all device memory (input/output/buffer pointers).
- CUDA kernels include bounds checks and device-side debug prints to help diagnose pointer and memory issues.
- If you encounter `cudaStreamSynchronize failed after kernel: an illegal memory access was encountered`, check:
  - That all device pointers (rx_pilots, tx_pilots, channel_estimates, pilot_estimates) are valid and allocated for the correct number of elements.
  - That the kernel launch configuration matches the buffer sizes (blockSize, gridSize).
  - That the input data is copied to device memory correctly and not freed before kernel execution.
- Device-side debug prints for `tid == 0` will show pointer values and the first element of each input buffer to help diagnose issues.

### Example of Successful Run
```
# channel_estimation_example
Creating channel estimation pipeline...
Pipeline ID: test_channel_estimation
Setting up pipeline...
[DEBUG] Allocated: d_params_=0x7f755c800200 d_pilot_estimates_=0x7f755c800800
Warming up pipeline...
Generating test data...
Number of pilots: 75
Number of subcarriers: 300
[DEBUG] Example device ptrs: d_rx_pilots=0x7f755c809000, d_tx_pilots=0x7f755c809400, d_channel_estimates=0x7f755c809800
[DEBUG] Pipeline: set_inputs called with all_ports. Now calling configure_io...
Channel estimation pipeline executed successfully!
First 5 channel estimates:
  [0]: (0.706574, 0.372196)
  [1]: (0.757888, 0.338698)
  [2]: (0.809201, 0.305201)
  [3]: (0.860515, 0.271703)
  [4]: (0.911829, 0.238206)
Test completed successfully!
```

## Machine Learning (ML) Channel Estimation: Training and Usage

This example supports ML-based channel estimation using a neural network model deployed with NVIDIA TensorRT.

- **How to train a model:**
  1. Prepare (received pilots, true channel) data.
  2. Train a neural network (e.g., in PyTorch or TensorFlow) to map pilots to channel estimates.
  3. Export the model to ONNX, then convert to TensorRT engine (`.engine` file) using `trtexec` or TensorRT Python API.
- **How to use in this example:**
  1. Build with TensorRT support.
  2. Run with `--algorithm ml_tensorrt --model_path /path/to/model.engine` and set other ML parameters as needed.

See [README-ML.md](./README-ML.md) for a step-by-step guide on training and deploying your own ML channel estimator.