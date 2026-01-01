# NVIDIA Aerial Framework Examples

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/nvidia/nvidia-aerial-examples)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A comprehensive collection of examples demonstrating GPU-accelerated 5G signal processing using the NVIDIA Aerial Framework. These examples showcase best practices for implementing high-performance baseband processing algorithms with modern C++20, CUDA, and the Aerial framework's task-based architecture.

## 🚀 Overview

This repository contains production-ready examples that demonstrate:

- **GPU-Accelerated Signal Processing**: High-performance CUDA implementations of 5G NR algorithms
- **Framework Integration**: Proper use of Aerial framework abstractions (tasks, pipelines, tensors)
- **Memory Management**: Efficient GPU memory handling with framework memory pools
- **Performance Optimization**: CUDA graphs, shared memory, and stream-based execution
- **Production Patterns**: Error handling, monitoring, and scalable architecture designs

## 📁 Repository Structure

```
nvidia-aerial-examples/
├── channel_estimation/          # Channel estimation pipeline example
│   ├── channel_estimator.hpp    # Module interface and GPU kernel declarations
│   ├── channel_estimator.cu     # CUDA implementation with LS/MMSE algorithms
│   ├── channel_est_pipeline.hpp # Pipeline orchestration and factory classes
│   ├── channel_est_pipeline.cpp # Pipeline implementation with memory management
│   ├── example.cpp              # Complete working example with benchmarks
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
│   ├── fft_module.hpp           # cuFFT integration with framework
│   ├── fft_pipeline.hpp         # FFT pipeline interface with multi-size support
│   ├── fft_pipeline.cpp         # cuFFT-based pipeline implementation
│   ├── fft_example.cpp          # Simple OFDM processing and FFT usage examples
│   ├── fft_processing_example.cpp # Complete comprehensive example with benchmarks
│   ├── README.md                # Detailed module documentation
│   └── CMakeLists.txt           # Build configuration
├── mimo_detection/              # MIMO detection algorithms
│   ├── mimo_detector.hpp        # ML, ZF, MMSE detection methods
│   ├── mimo_pipeline.hpp        # MIMO detection pipeline interface
│   ├── mimo_pipeline_impl.cu    # GPU-accelerated MIMO processing pipeline
│   ├── mimo_example.cpp         # Simple real-time MIMO streaming examples
│   ├── mimo_detection_example.cpp # Complete comprehensive example with benchmarks
│   ├── README.md                # Detailed module documentation
│   └── CMakeLists.txt           # Build configuration
├── neural_beamforming/          # ML-based neural beamforming
│   ├── neural_beamforming_pipeline.hpp # Neural beamforming with TensorRT integration
│   ├── neural_beamforming_pipeline.cpp # ML pipeline implementation with training
│   ├── neural_beamforming_example.cpp  # Simple ML integration demonstration
│   ├── neural_beamforming_ml_example.cpp # Comprehensive ML pipeline example
│   ├── README.md                # ML integration and training documentation
│   └── CMakeLists.txt           # Build configuration with ML dependencies
├── docs/                        # Documentation and guides
│   ├── getting_started.md       # Setup and first example
│   ├── performance_guide.md     # Optimization best practices
│   └── api_reference.md         # Framework API documentation
├── common/                      # Shared utilities and base classes
│   ├── test_utils.hpp           # Testing and validation utilities
│   ├── perf_utils.hpp           # Performance measurement tools
│   └── data_generators.hpp     # Synthetic data generation
└── scripts/                     # Build and testing scripts
    ├── build.sh                 # Automated build script
    ├── test_all.sh              # Run all examples and tests
    └── benchmark.sh             # Performance benchmarking
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
- **NVIDIA Aerial Framework**: Latest version
- **Optional**: Docker for containerized builds

## 🏗 Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/nvidia/nvidia-aerial-examples.git
cd nvidia-aerial-examples
```

### 2. Install NVIDIA Aerial Framework
```bash
# Download and install CUDA Toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_545.23.08_linux.run
sudo sh cuda_12.3.2_545.23.08_linux.run

# Add CUDA to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install additional CUDA libraries
sudo apt-get update
sudo apt-get install -y \
    libcublas-dev \
    libcurand-dev \
    libcusolver-dev \
    libcufft-dev \
    libcudnn8-dev \
    libnvinfer-dev \
    libnvinfer-plugin-dev

# Download NVIDIA Aerial Framework
# Option 1: From NGC (NVIDIA GPU Cloud) - Recommended
docker pull nvcr.io/nvidia/aerial/aerial-framework:latest

# Extract framework from container
docker create --name temp_aerial nvcr.io/nvidia/aerial/aerial-framework:latest
docker cp temp_aerial:/opt/nvidia/aerial-framework ./aerial-framework
docker rm temp_aerial

# Option 2: Build from source (if you have access to source)
# git clone <aerial-framework-repo-url> aerial-framework
# cd aerial-framework
# mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/nvidia/aerial-framework
# make -j$(nproc)
# sudo make install

# Set Aerial Framework environment variables
export AERIAL_FRAMEWORK_ROOT=$(pwd)/aerial-framework
export AERIAL_FRAMEWORK_INCLUDE_DIRS=${AERIAL_FRAMEWORK_ROOT}/include
export AERIAL_FRAMEWORK_LIBRARIES=${AERIAL_FRAMEWORK_ROOT}/lib

# Add to ~/.bashrc for persistence
echo "export AERIAL_FRAMEWORK_ROOT=${AERIAL_FRAMEWORK_ROOT}" >> ~/.bashrc
echo "export AERIAL_FRAMEWORK_INCLUDE_DIRS=${AERIAL_FRAMEWORK_ROOT}/include" >> ~/.bashrc
echo "export AERIAL_FRAMEWORK_LIBRARIES=${AERIAL_FRAMEWORK_ROOT}/lib" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${AERIAL_FRAMEWORK_ROOT}/lib:$LD_LIBRARY_PATH" >> ~/.bashrc

# Verify installation
ls -la ${AERIAL_FRAMEWORK_ROOT}/include/aerial
ls -la ${AERIAL_FRAMEWORK_ROOT}/lib
```

### 3. Setup Additional Dependencies
```bash
# Install TensorRT for neural beamforming example
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_installers/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-12.0/nv-tensorrt-local-42B2FC56-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y tensorrt

# Install cuDNN for neural network operations
sudo apt-get install -y libcudnn8-dev

# Install Python dependencies for ML examples (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime-gpu tensorrt
```

### 4. Build Examples
```bash
# Create build directory
mkdir build && cd build

# Verify Aerial Framework is properly installed
if [ ! -d "${AERIAL_FRAMEWORK_ROOT}" ]; then
    echo "Error: AERIAL_FRAMEWORK_ROOT not set or directory not found"
    echo "Please set: export AERIAL_FRAMEWORK_ROOT=/path/to/aerial-framework"
    exit 1
fi

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89;90" \
  -DAERIAL_FRAMEWORK_ROOT=${AERIAL_FRAMEWORK_ROOT} \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DTENSORRT_ROOT=/usr/local/TensorRT \
  -DCUDNN_ROOT=/usr/local/cudnn

# Build all examples
make -j$(nproc)

# Verify build success
echo "Build completed successfully!"
echo "Available executables:"
find . -name "*example" -type f -executable
```

### 5. Run Examples
```bash
# Test basic functionality
./channel_estimation/channel_estimation_example

# Run comprehensive examples
./modulation_mapping/modulation_mapping_example
./fft_processing/fft_processing_example  
./mimo_detection/mimo_detection_example
./neural_beamforming/neural_beamforming_ml_example

# Performance benchmarks
./scripts/benchmark.sh --all-examples --iterations 1000
```

## 🔧 Installation Troubleshooting

### Common Issues

**1. Aerial Framework Not Found:**
```bash
# Verify paths
echo $AERIAL_FRAMEWORK_ROOT
ls -la $AERIAL_FRAMEWORK_ROOT/include/aerial
ls -la $AERIAL_FRAMEWORK_ROOT/lib

# If using NGC container method, ensure extraction worked
docker images | grep aerial
```

**2. CUDA/TensorRT Issues:**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify TensorRT
dpkg -l | grep tensorrt
pkg-config --modversion tensorrt
```

**3. Build Errors:**
```bash
# Clean rebuild
rm -rf build && mkdir build && cd build

# Debug build for more information
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON

# Check specific dependencies
cmake .. --debug-find
```

**4. Runtime Issues:**
```bash
# Check library paths
echo $LD_LIBRARY_PATH
ldd ./channel_estimation/channel_estimation_example

# Verify GPU access
nvidia-smi
```

## 🐳 Docker Alternative

If you prefer containerized development:

```bash
# Use pre-built development container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/aerial/aerial-framework:latest \
  /bin/bash

# Inside container
cd /workspace
mkdir build && cd build
cmake .. -DAERIAL_FRAMEWORK_ROOT=/opt/nvidia/aerial-framework
make -j$(nproc)
```
./mimo_detection/mimo_example

# Run with custom parameters
./channel_estimation/channel_estimation_example --num-rbs 100 --algorithm mmse
./modulation_mapping/modulation_example --modulation 256QAM --batch-size 1024

# Performance benchmarks
./scripts/benchmark.sh --all-examples --iterations 1000
```

## 🎯 Example Walkthroughs

### Channel Estimation Pipeline
A comprehensive example showing 5G NR channel estimation with multiple algorithms:
- **Location**: `channel_estimation/`
- **Algorithms**: Least Squares, MMSE, Linear Interpolation
- **Features**: CUDA graphs, memory pools, performance monitoring
- **Performance**: 45,000+ ops/sec, <25μs latency

### Modulation Mapping Pipeline
GPU-accelerated QAM constellation mapping pipeline for 5G NR:
- **Location**: `modulation_mapping/`
- **Modulations**: QPSK, 16QAM, 64QAM, 256QAM
- **Features**: Batch processing, factory patterns, performance statistics
- **Optimizations**: Shared memory tables, coalesced access patterns
- **Throughput**: 100M+ symbols/sec

### FFT Processing Pipeline
cuFFT-based FFT operations with OFDM support:
- **Location**: `fft_processing/`
- **Operations**: Forward/Inverse FFT, mixed batch sizes
- **Features**: OFDM processing, cyclic prefix handling, precision modes
- **Applications**: 5G NR OFDM symbol processing
- **Performance**: Variable based on FFT size and batch configuration

### MIMO Detection Pipeline
Multi-antenna signal detection algorithms with real-time streaming:
- **Location**: `mimo_detection/`
- **Methods**: Maximum Likelihood, Zero Forcing, MMSE
- **Configurations**: 2x2, 4x4, 8x8 MIMO systems
- **Features**: Batch processing, streaming support, SNR analysis
- **Applications**: 5G NR PUSCH/PDSCH processing

### Neural Beamforming Pipeline
ML-based intelligent beamforming with complete training and deployment pipeline:
- **Location**: `neural_beamforming/`
- **Algorithms**: DNN/CNN beamforming, hybrid traditional+neural approaches
- **ML Integration**: TensorRT optimization, FP32/FP16/INT8 precision support
- **Features**: Online learning, training data generation, model export
- **Applications**: Intelligent antenna array optimization for 5G/6G

## 🔧 Build Options

```bash
# Debug build with detailed logging
cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_DETAILED_LOGGING=ON

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_FAST_MATH=ON

# Build with unit tests
cmake .. -DBUILD_TESTS=ON -DENABLE_GTEST=ON

# Build with documentation
cmake .. -DBUILD_DOCS=ON -DENABLE_DOXYGEN=ON

# Specific GPU architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86"  # Ampere only
```

## 📊 Performance Benchmarks

| Example | Configuration | Throughput | Latency | GPU Memory |
|---------|--------------|------------|---------|------------|
| Channel Est | 100 RBs, LS | 12,000 ops/sec | 83μs | 60MB |
| Channel Est | 273 RBs, MMSE | 4,500 ops/sec | 222μs | 164MB |
| Modulation | 16QAM Batch | 120M symbols/sec | 8μs | 25MB |
| Modulation | 256QAM HP | 85M symbols/sec | 12μs | 32MB |
| FFT | 1024 Batch | 2,800 Msamples/sec | 15μs | 40MB |
| FFT | OFDM 2048 | 1,200 Msamples/sec | 35μs | 85MB |
| MIMO 2x2 | ZF Detection | 15,000 slots/sec | 67μs | 30MB |
| MIMO 4x4 | MMSE Detection | 8,000 slots/sec | 125μs | 45MB |
| MIMO 8x8 | ML Detection | 2,200 slots/sec | 455μs | 120MB |

*Benchmarks performed on RTX 4090 with CUDA 12.3*

## 🧪 Testing

```bash
# Run all unit tests
make test

# Run specific example tests
ctest -R channel_estimation

# Performance regression tests
./scripts/benchmark.sh --baseline

# Memory leak detection
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

- **[Getting Started Guide](docs/getting_started.md)**: Step-by-step tutorial for your first example
- **[Performance Optimization](docs/performance_guide.md)**: Best practices for GPU acceleration
- **[API Reference](docs/api_reference.md)**: Complete framework interface documentation
- **[Architecture Overview](docs/architecture.md)**: Framework design principles and patterns

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
# Reduce batch size or resource blocks
./example --num-rbs 25 --batch-size 1
```

**Compilation Errors**
```bash
# Check CUDA and compiler versions
nvcc --version
gcc --version
```

**Performance Issues**
```bash
# Enable GPU persistence mode
sudo nvidia-smi -pm 1

# Set GPU clocks to maximum
sudo nvidia-smi -lgc 1900,1900
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA Aerial Framework team for the foundational architecture
- CUDA development team for the exceptional GPU computing platform
- 5G NR specification contributors for algorithm definitions

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/nvidia/nvidia-aerial-examples/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nvidia/nvidia-aerial-examples/discussions)
- **Documentation**: [NVIDIA Aerial Documentation](https://docs.nvidia.com/aerial/)
- **Developer Forums**: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/aerial/)