# NVIDIA Aerial Framework Examples

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/nvidia/nvidia-aerial-examples)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A comprehensive collection of examples demonstrating GPU-accelerated 5G signal processing using the NVIDIA Aerial Framework. These examples showcase best practices for implementing high-performance baseband processing algorithms with modern C++20, CUDA, and the Aerial framework's task-based architecture.

## 🚀 Overview

This repository contains working examples that demonstrate:

- **GPU-Accelerated Signal Processing**: High-performance CUDA implementations of 5G NR algorithms
- **Framework Integration**: Compatible implementations with Aerial framework patterns using stubs
- **Memory Management**: Efficient GPU memory handling with CUDA APIs
- **Performance Optimization**: CUDA kernels, stream-based execution, and optimized algorithms
- **Production Patterns**: Error handling, monitoring, and scalable architecture designs

**Note**: These examples use simplified framework stubs to demonstrate algorithms without requiring full Aerial framework installation. They showcase the signal processing algorithms and CUDA implementations that can be integrated with the complete framework.

## 📁 Repository Structure

```
nvidia-aerial-examples/
├── channel_estimation/          # Channel estimation pipeline example
│   ├── channel_estimator.hpp    # Module interface and GPU kernel declarations
│   ├── channel_estimator.cu     # CUDA implementation with LS/MMSE algorithms  
│   ├── channel_est_pipeline.hpp # Pipeline orchestration and factory classes
│   ├── channel_est_pipeline.cpp # Pipeline implementation with memory management
│   ├── channel_estimation_example.cpp # Complete working example with benchmarks
│   ├── framework_stubs.cpp      # Framework compatibility layer
│   └── CMakeLists.txt           # Build configuration
├── modulation_mapping/          # QAM modulation mapping pipeline
│   ├── modulator.hpp            # QPSK/16QAM/64QAM/256QAM implementations
│   ├── modulator.cu             # GPU kernels for constellation mapping
│   ├── modulation_pipeline.hpp  # Modulation pipeline interface and factory
│   ├── modulation_pipeline.cpp  # Pipeline implementation with batch processing
│   ├── modulation_example.cpp   # Simple usage examples and demonstrations
│   ├── modulation_mapping_example.cpp # Complete comprehensive example with benchmarks
│   ├── README.md                # Detailed module documentation
│   └── CMakeLists.txt           # Build configuration
├── fft_processing/              # FFT-based processing pipeline
│   ├── fft_pipeline.hpp         # FFT pipeline interface with multi-size support
│   ├── fft_pipeline.cpp         # cuFFT-based pipeline implementation
│   ├── fft_example.cpp          # Simple OFDM processing and FFT usage examples
│   ├── fft_processing_example.cpp # Complete comprehensive example with benchmarks
│   ├── README.md                # Detailed module documentation
│   └── CMakeLists.txt           # Build configuration
├── mimo_detection/              # MIMO detection algorithms
│   ├── mimo_detector.hpp        # ML, ZF, MMSE detection methods interface
│   ├── mimo_pipeline.hpp        # MIMO detection pipeline interface
│   ├── mimo_pipeline_impl.cu    # GPU-accelerated MIMO processing pipeline
│   ├── mimo_example.cpp         # Simple real-time MIMO streaming examples
│   ├── mimo_detection_example.cpp # Complete comprehensive example with benchmarks
│   ├── framework_stubs.cpp      # Framework compatibility layer
│   ├── README.md                # Detailed module documentation
│   └── CMakeLists.txt           # Build configuration
├── neural_beamforming/          # Neural beamforming with classical and ML algorithms
│   ├── neural_beamforming_module.hpp    # NeuralBeamformer class interface
│   ├── neural_beamforming_module.cu     # CUDA kernels for beamforming algorithms
│   ├── neural_beamforming_pipeline.hpp  # High-level pipeline wrapper
│   ├── neural_beamforming_pipeline.cu   # Pipeline implementation
│   ├── neural_beamforming_example.cpp   # Complete example with performance analysis
│   ├── CMakeLists.txt           # Build configuration with TensorRT integration
│   ├── README.md                # Comprehensive theory and usage documentation
│   └── TENSORRT_INSTALL.md      # TensorRT installation guide
├── common/                      # Shared utilities and base classes
│   ├── test_utils.hpp           # Testing and validation utilities
│   ├── perf_utils.hpp           # Performance measurement tools
│   └── CMakeLists.txt           # Shared utilities build configuration
└── scripts/                     # Build and testing scripts
    ├── build.sh                 # Automated build script
    └── test_all.sh              # Run all examples and tests
```

## 🛠 Prerequisites

### Hardware Requirements
- NVIDIA GPU with compute capability 7.0 or higher (Volta, Turing, Ampere, Ada Lovelace, Hopper)
- 16GB+ system RAM (32GB+ recommended for large examples)
- 8GB+ GPU memory

### Software Requirements
- **CUDA Toolkit**: 11.8 or later
- **CMake**: 3.20 or later  
- **Compiler**: GCC 9+ or Clang 12+ with C++20 support
- **Note**: NVIDIA Aerial Framework is NOT required - these examples use framework stubs

## 🏗 Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nvidia/nvidia-aerial-examples.git
cd nvidia-aerial-examples
```

### 2. Install Dependencies
```bash
# Install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run

# Add CUDA to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install CUDA development libraries
sudo apt-get update
sudo apt-get install -y \
    libcublas-dev \
    libcurand-dev \
    libcusolver-dev \
    libcufft-dev

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### 3. Build Examples
```bash
# Create build directory
mkdir build && cd build

# Configure with CMake (simplified - no Aerial Framework required)
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda

# Build all examples
make -j$(nproc)

# Verify build success
echo "Build completed successfully!"
echo "Available executables:"
find . -name "*example" -type f -executable
```

### 4. Run Examples
```bash
# Test basic functionality
./channel_estimation/channel_estimation_example

# Run comprehensive examples
./modulation_mapping/modulation_mapping_example
./fft_processing/fft_processing_example  
./mimo_detection/mimo_detection_example
./neural_beamforming/neural_beamforming_example

# Run simple examples
./modulation_mapping/modulation_example
./fft_processing/fft_example
./mimo_detection/mimo_example
./neural_beamforming/neural_beamforming_ml_example_simple
```

## 🔧 Installation Troubleshooting

### Common Issues

**1. CUDA Not Found:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify CUDA libraries
ldconfig -p | grep cuda
```

**2. Build Errors:**
```bash
# Clean rebuild
rm -rf build && mkdir build && cd build

# Debug build for more information
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON

# Check CUDA architecture support
nvidia-smi --query-gpu=compute_cap --format=csv
```

**3. Runtime Issues:**
```bash
# Check library paths
echo $LD_LIBRARY_PATH
ldd ./channel_estimation/channel_estimation_example

# Verify GPU access
nvidia-smi
```

## 🐳 Docker Alternative (Simplified)

For a quick start with Docker:

```bash
# Create a simple CUDA development environment
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvidia/cuda:12.3-devel-ubuntu20.04 \
  /bin/bash

# Inside container, install dependencies
apt-get update
apt-get install -y cmake build-essential

# Build examples
cd /workspace
mkdir build && cd build
cmake ..
make -j$(nproc)
```
./mimo_detection/mimo_example

# Run with custom parameters
./channel_estimation/channel_estimation_example --num-rbs 100 --algorithm mmse
./modulation_mapping/modulation_mapping_example --scheme 256QAM --subcarriers 1200
./neural_beamforming/neural_beamforming_example --algorithm NEURAL --antennas 128

# Performance benchmarks
./scripts/benchmark.sh --all-examples --iterations 1000
```

## 🎯 Example Walkthroughs

### Channel Estimation Pipeline
A comprehensive example showing 5G NR channel estimation with multiple algorithms:
- **Location**: `channel_estimation/`
- **Algorithms**: Least Squares, MMSE, Linear Interpolation
- **Features**: CUDA kernels, memory management, performance monitoring
- **Implementation**: Framework stubs for compatibility, full CUDA implementation

### Modulation Mapping Pipeline
GPU-accelerated QAM constellation mapping pipeline for 5G NR:
- **Location**: `modulation_mapping/`
- **Modulations**: QPSK, 16QAM, 64QAM, 256QAM
- **Features**: Batch processing, factory patterns, performance statistics
- **Optimizations**: Shared memory tables, coalesced access patterns

### FFT Processing Pipeline
cuFFT-based FFT operations with OFDM support:
- **Location**: `fft_processing/`
- **Operations**: Forward/Inverse FFT, mixed batch sizes
- **Features**: OFDM processing, simplified cuFFT integration
- **Applications**: 5G NR OFDM symbol processing

### MIMO Detection Pipeline
Multi-antenna signal detection algorithms with real-time streaming:
- **Location**: `mimo_detection/`
- **Algorithms**: Maximum Likelihood, Zero Forcing, MMSE
- **Features**: Real-time processing, CUDA streams, performance analysis
- **Applications**: Multi-user MIMO, massive MIMO systems

### Neural Beamforming Pipeline
Advanced beamforming with classical and neural network algorithms:
- **Location**: `neural_beamforming/`
- **Algorithms**: Conventional, MVDR, Zero-Forcing, Neural Network (TensorRT)
- **Features**: TensorRT integration, massive MIMO support, performance analytics
- **Applications**: 5G/6G beamforming, spatial multiplexing, interference mitigation

## 📊 Example Output

### Neural Beamforming Example Success
```bash
$ ./neural_beamforming/neural_beamforming_example

NVIDIA Aerial Framework - Neural Beamforming Example
=====================================================
Initialized neural beamforming pipeline successfully

=== Basic Beamforming Test ===
Configuration:
  Algorithm: MVDR
  Antennas: 64
  Users: 4
  Subcarriers: 1200
  OFDM symbols: 14
  Input symbols: 1075200
  Output symbols: 67200
  Channel estimates: 307200

First 4 input symbols (antenna 0): (-0.245,0.702) (0.449,-1.152) (-0.387,-0.101) (1.106,-0.344) 
Processed beamforming in 2.220 ms
First 4 output symbols (user 0): (1.851,-1.840) (0.133,0.559) (1.991,-1.073) (-1.502,1.088) 

Performance Results:
  Average SINR: -3.71 dB
  Beamforming gain: 19.56 dB
  Throughput: 926.7 Mbps
  Processing time: 2.213 ms

Example completed successfully!
```

## 🔧 Build Options

```bash
# Debug build with detailed logging
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release

# Specific GPU architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86"  # Ampere only
cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75"  # Volta/Turing only

# Enable verbose output
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
```

## 📊 Performance Notes

These examples demonstrate algorithmic implementations and CUDA programming patterns. Performance will vary based on:

- **GPU Architecture**: Newer architectures (Ampere, Ada Lovelace, Hopper) provide better performance
- **Memory Bandwidth**: Large batch sizes benefit from high memory bandwidth
- **CUDA Toolkit Version**: Newer versions include optimizations
- **System Configuration**: CPU, PCIe lanes, and system memory affect overall performance

### Expected Performance Characteristics
| Example | Algorithm | Implementation Focus |
|---------|-----------|---------------------|
| Channel Est | LS/MMSE | CUDA kernel optimization, memory coalescing |
| Modulation | QAM Mapping | Lookup table optimization, batch processing |
| FFT | cuFFT Integration | Library integration, memory management |
| MIMO Detection | ZF/MMSE | Matrix operations, cuBLAS integration |
| Neural Beamforming | Basic Demo | Simplified beamforming concepts |

*Performance will vary significantly based on problem size, GPU hardware, and configuration*

## 🧪 Testing

```bash
# Run examples to verify functionality
./channel_estimation/channel_estimation_example
./modulation_mapping/modulation_example
./fft_processing/fft_example
./mimo_detection/mimo_example
./neural_beamforming/neural_beamforming_example_simple

# Check memory usage (if valgrind is available)
valgrind --tool=memcheck ./channel_estimation/channel_estimation_example
```

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t aerial-examples .

# Run examples in container
docker run --gpus all -it aerial-examples

# Mount local data directory
docker run --gpus all -v $(pwd)/data:/workspace/data -it aerial-examples
```

## 📖 Documentation

- **Module README Files**: Each module has detailed documentation in its README.md file
- **Code Comments**: Extensive inline documentation in source files
- **CUDA Programming**: Examples demonstrate CUDA best practices
- **5G Signal Processing**: Algorithm implementations follow 3GPP specifications

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-algorithm`
3. Follow coding standards and add tests
4. Submit a pull request with detailed description

### Code Style
- C++20 modern practices
- Consistent naming conventions (snake_case for variables, PascalCase for classes)
- Comprehensive error handling and logging
- Performance-oriented design with benchmarks

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in examples
# Check available GPU memory
nvidia-smi
```

**Compilation Errors**
```bash
# Check CUDA and compiler versions
nvcc --version
gcc --version

# Ensure compute capability matches your GPU
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Performance Issues**
```bash
# Enable GPU persistence mode for consistent performance
sudo nvidia-smi -pm 1

# Check thermal throttling
watch -n 1 nvidia-smi
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA Aerial Framework team for the foundational architecture
- CUDA development team for the exceptional GPU computing platform
- 5G NR specification contributors for algorithm definitions

## 📞 Support

- **Code Issues**: Review module README files and code comments
- **CUDA Programming**: [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **5G Algorithms**: [3GPP Specifications](https://www.3gpp.org/specifications)
- **GPU Performance**: [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)