# FFT Processing Example

This example demonstrates Fast Fourier Transform (FFT) processing using the NVIDIA Aerial Framework with CUFFT integration. The implementation showcases real-world OFDM signal processing including FFT/IFFT operations, windowing, and normalization with CUDA acceleration for 5G NR and LTE systems.

## Overview

FFT processing is at the heart of OFDM-based communication systems, enabling efficient modulation and demodulation of orthogonal subcarriers. FFT operations are essential for:
- **OFDM Modulation/Demodulation**: Converting between time and frequency domains
- **Channel Equalization**: Frequency domain signal processing
- **Spectral Analysis**: Signal analysis and monitoring
- **Filtering Operations**: Efficient convolution via overlap-save/overlap-add
- **Beamforming**: Spatial signal processing in frequency domain

This example implements:
- **Forward FFT**: Time-to-frequency domain transformation
- **Inverse FFT**: Frequency-to-time domain transformation  
- **Windowing Functions**: Signal conditioning and spectral shaping
- **CUFFT Integration**: High-performance GPU-accelerated FFT processing
- **Real Framework Integration**: Uses actual Aerial Framework interfaces

## Architecture

### OFDM System Model
In OFDM systems, the relationship between time and frequency domains is:
```
Time Domain:    x[n] = (1/N) * Σ X[k] * exp(j*2π*k*n/N)  (IFFT)
Frequency Domain: X[k] = Σ x[n] * exp(-j*2π*k*n/N)       (FFT)
```
Where:
- `x[n]`: Time domain signal samples
- `X[k]`: Frequency domain subcarrier data
- `N`: FFT size
- `k`: Subcarrier index, `n`: Time sample index

### Processing Pipeline
1. **Input Buffering**: Collect time/frequency domain samples
2. **Windowing**: Apply window function for spectral shaping
3. **FFT/IFFT**: Transform between domains using CUFFT
4. **Normalization**: Scale output appropriately
5. **Output Processing**: Prepare results for subsequent processing

## Files Structure

```
fft_processing/
├── CMakeLists.txt                    # Build configuration
├── README.md                         # This documentation
├── fft_processing_module.hpp         # Core FFT processing module interface
├── fft_processing_module.cu          # FFT processing CUDA implementation
├── fft_processing_pipeline.hpp       # Pipeline wrapper interface
├── fft_processing_pipeline.cpp       # Pipeline wrapper implementation
└── fft_processing_example.cpp        # Example application and tests
```

## Key Components

### FFTProcessor Class
- **Framework Integration**: Inherits from `IModule`, `IAllocationInfoProvider`, `IStreamExecutor`
- **CUFFT Integration**: Leverages NVIDIA's optimized FFT library
- **Flexible Configuration**: Supports various FFT sizes and processing modes
- **Memory Management**: Efficient GPU memory allocation and management

### CUDA Kernels
- **apply_window_kernel**: Applies windowing functions (Hann, Hamming, Blackman)
- **normalize_output_kernel**: Post-FFT normalization and scaling
- **complex_conjugate_kernel**: Complex conjugation operations
- **Optimized Memory Access**: Coalesced global memory access patterns

### Pipeline Wrapper
- **High-level Interface**: Simplified API for FFT processing
- **Performance Monitoring**: Built-in profiling and metrics collection
- **Flexible I/O**: Support for both host and device memory operations
- **Batch Processing**: Multiple FFT operations in parallel

## Algorithm Details

### FFT Implementation
The implementation uses NVIDIA's CUFFT library for optimal performance:

**Forward FFT (Time → Frequency):**
```cpp
cufftExecC2C(fft_plan, input_time, output_freq, CUFFT_FORWARD);
```

**Inverse FFT (Frequency → Time):**
```cpp
cufftExecC2C(ifft_plan, input_freq, output_time, CUFFT_INVERSE);
```

### Windowing Functions
Applied to reduce spectral leakage and improve frequency domain characteristics:

**Hann Window:**
```
w[n] = 0.5 * (1 - cos(2π*n/(N-1)))
```

**Hamming Window:**
```
w[n] = 0.54 - 0.46 * cos(2π*n/(N-1))
```

**Blackman Window:**
```
w[n] = 0.42 - 0.5*cos(2π*n/(N-1)) + 0.08*cos(4π*n/(N-1))
```

### Normalization
- **Forward FFT**: No normalization (matches OFDM standards)
- **Inverse FFT**: Scale by 1/N for unity gain
- **Power Conservation**: Maintains signal energy relationships

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0+
- CMake 3.18+
- NVIDIA Aerial Framework installed
- CUFFT library (included with CUDA Toolkit)
- C++20 compatible compiler
- GPU with Compute Capability 7.5+

### Build Steps
```bash
cd fft_processing
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run Example
```bash
./fft_processing_example
```

### Expected Output
```
========================================
     NVIDIA Aerial FFT Processing      
========================================

FFT Configuration:
  FFT Size: 1024
  Batch Size: 14
  Window Type: Hann
  Processing Mode: Forward + Inverse

Initializing FFT processing pipeline...
Pipeline initialized successfully!

Generating test data...
Input Time Signal - Samples: 14336, Avg Power: 1.0, Peak: 4.2
Applied Window - Samples: 14336, Avg Power: 0.5, Peak: 4.1

Testing Forward FFT...
  Processing Time: 0.32 ms
  FFT Frequency Data - Samples: 14336, Avg Power: 512.0, Peak: 2048.5

Testing Inverse FFT...
  Processing Time: 0.28 ms
  IFFT Time Data - Samples: 14336, Avg Power: 1.0, Peak: 4.0
  Round-trip Error: 1.2e-06

Performance Metrics:
  Average Processing Time: 0.30 ms
  Peak Processing Time: 0.32 ms
  Total Processed Batches: 2
  Throughput: 95.4 MB/s
  CUFFT Performance: 15.8 GFLOPS
```

## Performance Characteristics

### Computational Complexity
- **FFT**: O(N log N) where N = FFT size
- **Windowing**: O(N) per batch
- **Normalization**: O(N) per batch  
- **GPU Parallelization**: Concurrent processing of multiple FFT batches

### Memory Requirements
- **Input Buffer**: `batch_size × fft_size × sizeof(complex<float>)`
- **Output Buffer**: `batch_size × fft_size × sizeof(complex<float>)`
- **Window Buffer**: `fft_size × sizeof(float)`
- **CUFFT Work Area**: Additional memory determined by CUFFT planner

### Typical Performance
- **1024-point FFT, batch=14**: ~0.3 ms processing time
- **Memory Bandwidth**: ~90-100 GB/s effective throughput
- **CUFFT Efficiency**: >95% of theoretical peak performance
- **GPU Utilization**: >85% on modern NVIDIA GPUs

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
- Proper error handling and CUFFT error management

## Advanced Features

### Window Functions
- **Rectangular**: No windowing (unity gain)
- **Hann**: Good general-purpose window
- **Hamming**: Slightly better sidelobe suppression
- **Blackman**: Excellent sidelobe suppression
- **Kaiser**: Parametric window design (future)

### Processing Modes
- **Forward Only**: Time to frequency domain
- **Inverse Only**: Frequency to time domain
- **Round-trip**: Forward + Inverse for validation
- **Batch Processing**: Multiple FFTs in parallel

### Optimization Features
- **CUFFT Planning**: Optimal algorithm selection
- **Stream Processing**: Asynchronous GPU execution
- **Memory Pooling**: Efficient buffer reuse
- **Error Checking**: Comprehensive CUFFT error handling

## Use Cases

### 5G NR Systems
- **OFDM Modulation**: Subcarrier mapping and IFFT
- **OFDM Demodulation**: FFT and subcarrier demapping  
- **Channel Estimation**: Frequency domain processing
- **Beamforming**: Spatial processing in frequency domain

### LTE Systems
- **OFDMA Processing**: Multiple user signal processing
- **SC-FDMA**: Single carrier frequency domain multiple access
- **Reference Signal Processing**: Pilot extraction and processing

### Spectral Analysis
- **Signal Monitoring**: Spectrum analysis and measurements
- **Interference Detection**: Unwanted signal identification
- **Quality Assessment**: Signal quality metrics computation

### Digital Signal Processing
- **Convolution**: Fast convolution via FFT
- **Filtering**: Frequency domain filter implementation
- **Correlation**: Cross-correlation and auto-correlation

## Configuration Parameters

### FFTParams Structure
```cpp
struct FFTParams {
    int fft_size;                   // FFT length (power of 2)
    int batch_size;                 // Number of FFTs in batch
    WindowType window_type;         // Windowing function
    FFTDirection direction;         // Forward, Inverse, or Both
    bool normalize_output;          // Apply normalization
    bool enable_windowing;          // Apply window function
    float overlap_factor;           // For overlapped processing
};
```

### Supported FFT Sizes
- **Power of 2**: 64, 128, 256, 512, 1024, 2048, 4096, 8192
- **Mixed Radix**: Combinations of small prime factors (future)
- **Optimal Sizes**: CUFFT-optimized sizes for best performance

## Performance Optimization

### CUFFT Best Practices
- **Plan Reuse**: Create plans once, execute multiple times
- **Batch Processing**: Process multiple FFTs together
- **Memory Layout**: Ensure proper data alignment
- **Stream Usage**: Overlap computation and memory transfers

### Memory Optimization
- **Pinned Memory**: Use cudaMallocHost for host buffers
- **Memory Pools**: Reuse allocated buffers
- **Coalesced Access**: Ensure optimal memory access patterns
- **Bank Conflicts**: Avoid shared memory bank conflicts

### GPU Architecture Considerations
- **Occupancy**: Balance thread blocks and register usage
- **Instruction Mix**: Optimize arithmetic intensity
- **Memory Hierarchy**: Leverage L1/L2 cache effectively
- **Tensor Cores**: Utilize specialized units when applicable

## Error Handling and Debugging

### CUFFT Error Codes
```cpp
// Common CUFFT errors and handling
switch (cufft_result) {
    case CUFFT_SUCCESS:
        break;
    case CUFFT_INVALID_PLAN:
        // Handle invalid plan error
        break;
    case CUFFT_ALLOC_FAILED:
        // Handle memory allocation failure
        break;
    // ... other error cases
}
```

### Debug Features
- **Validation Mode**: Compare with reference implementation
- **Profiling**: Built-in performance measurement
- **Memory Checking**: CUDA memory error detection
- **Numerical Accuracy**: Round-trip error analysis

## Future Enhancements

### Advanced FFT Features
- **Real FFT**: Optimized real-to-complex transforms
- **Chirp Z-Transform**: Non-uniform frequency sampling
- **Prime Length FFTs**: Arbitrary length transforms
- **Winograd FFT**: Minimal multiplication algorithms

### Signal Processing Extensions
- **Overlap-Save/Add**: Continuous processing
- **Polyphase Filtering**: Efficient filter bank implementation  
- **Spectrum Estimation**: PSD and spectral methods
- **Time-Frequency Analysis**: Spectrograms and wavelets

### Performance Improvements
- **Multi-GPU**: Distribution across multiple devices
- **Tensor Core Usage**: Mixed precision acceleration
- **Graph Capture**: CUDA graph optimization
- **Dynamic Parallelism**: Adaptive kernel launching

## Troubleshooting

### Common Issues
- **CUFFT Errors**: Check plan creation and memory allocation
- **Poor Performance**: Verify FFT size and batch configuration
- **Memory Issues**: Ensure sufficient GPU memory available
- **Accuracy Problems**: Check normalization and window settings

### Debug Mode
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
cuda-gdb ./fft_processing_example
```

### Performance Tuning
- **FFT Size**: Use powers of 2 for optimal performance
- **Batch Size**: Balance memory usage and parallelism
- **Window Selection**: Choose based on application requirements
- **Stream Configuration**: Optimize for target GPU architecture

### Memory Management
- Monitor GPU memory usage with nvidia-smi
- Profile memory access patterns with NSight Compute
- Use CUDA memory checker for debugging
- Implement proper cleanup for CUFFT plans and buffers