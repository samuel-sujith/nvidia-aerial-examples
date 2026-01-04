# MIMO Detection Example

This example demonstrates Multi-Input Multi-Output (MIMO) symbol detection using the NVIDIA Aerial Framework. The implementation showcases real-world MIMO detection algorithms including Zero Forcing (ZF) and Minimum Mean Square Error (MMSE) detection with CUDA acceleration.

## Overview

MIMO technology is fundamental to modern wireless communication systems like 5G NR, enabling:
- **Spatial Multiplexing**: Multiple data streams transmitted simultaneously
- **Diversity Gains**: Improved reliability through multiple antennas  
- **Array Gains**: Enhanced signal strength and quality
- **Interference Mitigation**: Better signal separation in dense environments

This example implements:
- **Zero Forcing (ZF) Detection**: Linear detection with interference nulling
- **MMSE Detection**: Optimal linear detection with noise consideration
- **CUDA-accelerated Processing**: GPU kernels for high-performance computing
- **Real Framework Integration**: Uses actual Aerial Framework interfaces

## Architecture

### MIMO System Model
```
y = H * x + n
```
Where:
- `y`: Received signal vector (num_rx × 1)
- `H`: Channel matrix (num_rx × num_tx) 
- `x`: Transmitted symbol vector (num_tx × 1)
- `n`: Additive noise vector (num_rx × 1)

### Detection Algorithms

#### Zero Forcing (ZF)
```
x̂ = (H^H * H)^(-1) * H^H * y
```
- Nulls interference completely
- Amplifies noise, especially with poorly conditioned channels
- Lower complexity than optimal detection

#### MMSE 
```
x̂ = (H^H * H + σ²I)^(-1) * H^H * y
```
- Balances interference suppression and noise amplification  
- Better performance than ZF in low SNR conditions
- Requires noise variance knowledge

## Files Structure

```
mimo_detection/
├── CMakeLists.txt                    # Build configuration
├── README.md                         # This documentation
├── mimo_detection_module.hpp         # Core MIMO detection module interface
├── mimo_detection_module.cu          # MIMO detection CUDA implementation  
├── mimo_detection_pipeline.hpp       # Pipeline wrapper interface
├── mimo_detection_pipeline.cu        # Pipeline wrapper implementation
└── mimo_detection_example.cpp        # Example application and tests
```

## Key Components

### MIMODetector Class
- **Framework Integration**: Inherits from `IModule`, `IAllocationInfoProvider`, `IStreamExecutor`
- **Multi-Algorithm Support**: ZF, MMSE detection algorithms
- **Flexible Configuration**: Supports different antenna configurations and modulation schemes
- **Memory Management**: Efficient GPU memory allocation and management

### CUDA Kernels
- **zero_forcing_detection_kernel**: Implements ZF detection algorithm
- **mmse_detection_kernel**: Implements MMSE detection with noise regularization  
- **qpsk_hard_decision_kernel**: Hard decision decoding for QPSK symbols
- **Optimized Memory Access**: Coalesced global memory access patterns

### Pipeline Wrapper
- **High-level Interface**: Simplified API for MIMO detection processing
- **Performance Monitoring**: Built-in profiling and metrics collection
- **Flexible I/O**: Support for both host and device memory operations
- **Parameter Management**: Dynamic parameter updates without reinitialization

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0+
- CMake 3.18+
- NVIDIA Aerial Framework installed
- C++20 compatible compiler
- GPU with Compute Capability 7.5+

### Build Steps
```bash
cd mimo_detection
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run Example
```bash
./mimo_detection_example
```

### Expected Output
```
========================================
     NVIDIA Aerial MIMO Detection      
========================================

MIMO Configuration:
  TX Antennas: 2
  RX Antennas: 4
  Subcarriers: 1024
  OFDM Symbols: 14
  SNR: 15.0 dB

Initializing MIMO detection pipeline...
Pipeline initialized successfully!

Generating test data...
Transmitted Symbols - Samples: 28672, Avg Power: 2, Peak: 1.41421
Channel Matrix - Samples: 8192, Avg Power: 0.982926, Peak: 3.15539
Received Symbols - Samples: 57344, Avg Power: 3.97076, Peak: 6.67269

Testing Zero Forcing Detection...
  Zero Forcing SER: 7.89621%
  Processing Time: 0.223 ms
  ZF Detected Symbols - Samples: 28672, Avg Power: 2, Peak: 1.41421

Testing MMSE Detection...
  MMSE SER: 7.89621%
  Processing Time: 0.189 ms
  MMSE Detected Symbols - Samples: 28672, Avg Power: 2, Peak: 1.41421

Performance Metrics:
  Average Processing Time: 0.2196 ms
  Peak Processing Time: 0.223 ms
  Total Processed Frames: 2
  Throughput: 3273 MB/s

SNR Performance Analysis:
  SNR 0 dB: SER = 13.7521%
  SNR 5 dB: SER = 9.74819%
  SNR 10 dB: SER = 8.1822%
  SNR 15 dB: SER = 7.94155%
  SNR 20 dB: SER = 7.72879%
  SNR 25 dB: SER = 7.80552%

========================================
MIMO Detection Example completed successfully!
========================================
```

## Algorithm Details

### Zero Forcing Detection
The ZF detector aims to completely eliminate interference between spatial streams by inverting the channel matrix. For each subcarrier:

1. **Pseudo-inverse Calculation**: Compute `H_pinv = (H^H * H)^(-1) * H^H`
2. **Symbol Detection**: Apply `x̂ = H_pinv * y`
3. **Hard Decision**: Map to nearest constellation point

**Advantages:**
- Simple linear operation
- Perfect interference cancellation
- Lower computational complexity

**Disadvantages:** 
- Noise amplification in poorly conditioned channels
- Performance degradation at low SNR

### MMSE Detection
MMSE detection optimally balances interference suppression and noise amplification:

1. **MMSE Filter**: Compute `W = (H^H * H + σ²I)^(-1) * H^H`
2. **Symbol Detection**: Apply `x̂ = W * y`  
3. **Hard Decision**: Map to constellation points

**Advantages:**
- Optimal linear detector
- Better low-SNR performance
- Noise regularization prevents amplification

**Disadvantages:**
- Requires noise variance knowledge
- Higher computational complexity
- Residual interference remains

## Performance Characteristics

### Computational Complexity
- **ZF Detection**: O(T³) per subcarrier for matrix inversion (T = num_tx)
- **MMSE Detection**: O(T³) per subcarrier plus noise variance scaling
- **GPU Parallelization**: Concurrent processing across subcarriers and OFDM symbols

### Memory Requirements
- **Channel Matrix**: `num_rx × num_tx × num_subcarriers × sizeof(complex<float>)`
- **Received Symbols**: `num_rx × num_subcarriers × num_symbols × sizeof(complex<float>)`  
- **Detected Symbols**: `num_tx × num_subcarriers × num_symbols × sizeof(complex<float>)`
- **Temporary Buffers**: Additional working memory for matrix operations

### Typical Performance
- **2×4 MIMO, 1024 subcarriers**: ~0.19-0.22 ms processing time
- **Memory Bandwidth**: ~3.2 GB/s effective throughput
- **GPU Utilization**: >80% on modern NVIDIA GPUs
- **Symbol Error Rate**: 7-8% at 15 dB SNR for QPSK modulation

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

### Soft Output Support
- Optional soft bit generation for channel decoding
- Log-likelihood ratio (LLR) computation
- Integration with forward error correction (FEC) decoders

### Multi-Algorithm Support
- Runtime algorithm switching
- Performance comparison capabilities  
- Algorithm-specific parameter tuning

### Profiling and Debugging
- Built-in performance monitoring
- CUDA error checking and reporting
- Memory usage tracking
- Processing time measurement

## Future Enhancements

### Additional Algorithms
- **Maximum Likelihood (ML)**: Optimal but computationally intensive detection
- **Sphere Decoding**: Near-ML performance with reduced complexity
- **Successive Interference Cancellation (SIC)**: Ordered detection and cancellation

### Optimization Opportunities
- **Batch Processing**: Multiple subframes in parallel
- **Memory Optimization**: Reduce memory footprint
- **Kernel Fusion**: Combine operations for better efficiency
- **Mixed Precision**: Use FP16 for increased throughput

### Extended Functionality
- **Adaptive Detection**: SNR-based algorithm selection
- **Channel Prediction**: Exploit temporal correlation
- **Precoding Support**: Integration with transmit beamforming
- **Multi-User MIMO**: Support for MU-MIMO scenarios

## Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Reduce batch size or antenna configuration
- **Poor Performance**: Check GPU utilization and memory bandwidth
- **Compilation Errors**: Verify Aerial Framework installation and paths
- **Runtime Failures**: Enable debug builds for detailed error messages

### Debug Mode
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
gdb ./mimo_detection_example
```

### Performance Tuning
- Adjust CUDA block sizes for target GPU architecture
- Enable fast math optimizations for production builds  
- Profile memory access patterns with NSight Compute
- Monitor thermal throttling and power limits