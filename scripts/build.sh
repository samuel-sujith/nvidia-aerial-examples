#!/bin/bash

# NVIDIA Aerial Framework Examples - Build Script
# This script builds all examples with optimizations for different GPU architectures

set -e  # Exit on any error

# Configuration
BUILD_TYPE=${BUILD_TYPE:-Release}
BUILD_DIR=${BUILD_DIR:-build}
INSTALL_DIR=${INSTALL_DIR:-install}
CMAKE_ARGS=${CMAKE_ARGS:-}
JOBS=${JOBS:-$(nproc)}

# GPU Architecture detection and configuration
detect_gpu_arch() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Detecting GPU architecture..."
        
        # Get GPU compute capability
        GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d '.')
        
        case $GPU_ARCH in
            75) CUDA_ARCH="75" ;;  # T4, RTX 2080
            80) CUDA_ARCH="80" ;;  # A100, A10
            86) CUDA_ARCH="86" ;;  # RTX 3080/3090
            89) CUDA_ARCH="89" ;;  # RTX 4090, L40
            90) CUDA_ARCH="90" ;;  # H100
            *) 
                echo "Warning: Unknown GPU architecture $GPU_ARCH, using default"
                CUDA_ARCH="75;80;86;89;90"
                ;;
        esac
        
        echo "Detected GPU compute capability: $GPU_ARCH"
    else
        echo "Warning: nvidia-smi not found, using default CUDA architectures"
        CUDA_ARCH="75;80;86;89;90"
    fi
}

# Print usage information
print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build NVIDIA Aerial Framework Examples

OPTIONS:
    -h, --help              Show this help message
    -c, --clean             Clean build directory before building
    -d, --debug             Build in Debug mode (default: Release)
    -j, --jobs NUM          Number of parallel jobs (default: $(nproc))
    --install               Install after building
    --tests                 Build and run tests
    --examples-only         Build only examples (skip tests)
    --cuda-arch ARCH        Specify CUDA architecture (e.g., 75;80;86)

ENVIRONMENT VARIABLES:
    BUILD_TYPE              Build type (Release/Debug/RelWithDebInfo)
    BUILD_DIR               Build directory (default: build)
    INSTALL_DIR             Install directory (default: install)
    CMAKE_ARGS              Additional CMake arguments
    JOBS                    Number of parallel jobs
    CUDA_TOOLKIT_ROOT_DIR   CUDA installation path

EXAMPLES:
    $0                      # Basic build
    $0 --clean              # Clean build
    $0 --debug --tests      # Debug build with tests
    $0 --cuda-arch "80;86"  # Build for A100 and RTX 3080
EOF
}

# Parse command line arguments
CLEAN_BUILD=false
RUN_TESTS=false
INSTALL_AFTER_BUILD=false
EXAMPLES_ONLY=false
CUSTOM_CUDA_ARCH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -d|--debug)
            BUILD_TYPE=Debug
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --install)
            INSTALL_AFTER_BUILD=true
            shift
            ;;
        --tests)
            RUN_TESTS=true
            shift
            ;;
        --examples-only)
            EXAMPLES_ONLY=true
            shift
            ;;
        --cuda-arch)
            CUSTOM_CUDA_ARCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate environment
echo "=== NVIDIA Aerial Examples Build Script ==="
echo "Build type: $BUILD_TYPE"
echo "Build directory: $BUILD_DIR"
echo "Parallel jobs: $JOBS"

# Check for required tools
echo "Checking dependencies..."
for tool in cmake nvcc; do
    if ! command -v $tool &> /dev/null; then
        echo "Error: $tool is required but not found in PATH"
        exit 1
    fi
done

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo "CUDA version: $CUDA_VERSION"
    
    # Minimum CUDA version check
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [[ $CUDA_MAJOR -lt 11 || ($CUDA_MAJOR -eq 11 && $CUDA_MINOR -lt 8) ]]; then
        echo "Warning: CUDA 11.8+ recommended, found $CUDA_VERSION"
    fi
fi

# Detect or use custom GPU architecture
if [[ -n "$CUSTOM_CUDA_ARCH" ]]; then
    CUDA_ARCH="$CUSTOM_CUDA_ARCH"
    echo "Using custom CUDA architecture: $CUDA_ARCH"
else
    detect_gpu_arch
fi

# Clean build directory if requested
if [[ "$CLEAN_BUILD" == true ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
CMAKE_CONFIGURE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="../$INSTALL_DIR"
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
    -DCUDA_SEPARABLE_COMPILATION=ON
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# Add conditional flags
if [[ "$BUILD_TYPE" == "Debug" ]]; then
    CMAKE_CONFIGURE_ARGS+=(-DENABLE_DEBUG_INFO=ON)
    CMAKE_CONFIGURE_ARGS+=(-DCMAKE_CUDA_FLAGS_DEBUG="-g -G")
fi

if [[ "$EXAMPLES_ONLY" == true ]]; then
    CMAKE_CONFIGURE_ARGS+=(-DBUILD_TESTS=OFF)
else
    CMAKE_CONFIGURE_ARGS+=(-DBUILD_TESTS=ON)
fi

# Performance optimizations for Release builds
if [[ "$BUILD_TYPE" == "Release" ]]; then
    CMAKE_CONFIGURE_ARGS+=(-DCMAKE_CUDA_FLAGS_RELEASE="-O3 --use_fast_math")
    CMAKE_CONFIGURE_ARGS+=(-DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native")
fi

# Add any additional CMake arguments
if [[ -n "$CMAKE_ARGS" ]]; then
    CMAKE_CONFIGURE_ARGS+=($CMAKE_ARGS)
fi

echo "CMake configuration:"
printf '%s\n' "${CMAKE_CONFIGURE_ARGS[@]}"

cmake .. "${CMAKE_CONFIGURE_ARGS[@]}"

# Build
echo "Building with $JOBS parallel jobs..."
cmake --build . --config "$BUILD_TYPE" -j "$JOBS"

# Install if requested
if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
    echo "Installing..."
    cmake --install . --config "$BUILD_TYPE"
fi

# Run tests if requested
if [[ "$RUN_TESTS" == true && "$EXAMPLES_ONLY" == false ]]; then
    echo "Running tests..."
    ctest --build-config "$BUILD_TYPE" --output-on-failure -j "$JOBS"
fi

echo "Build completed successfully!"
echo "Build artifacts are in: $BUILD_DIR"

if [[ "$INSTALL_AFTER_BUILD" == true ]]; then
    echo "Installation completed in: $INSTALL_DIR"
fi

# Show example binaries
echo ""
echo "Available examples:"
find . -name "*example" -type f -executable | sort

echo ""
echo "Example usage:"
echo "  ./channel_estimation/channel_estimation_example"
echo "  ./modulation_mapping/modulation_example"
echo "  ./modulation_mapping/modulation_mapping_example"
echo "  ./fft_processing/fft_example"
echo "  ./fft_processing/fft_processing_example"
echo "  ./mimo_detection/mimo_example"
echo "  ./mimo_detection/mimo_detection_example"
echo "  ./neural_beamforming/neural_beamforming_example"
echo "  ./neural_beamforming/neural_beamforming_ml_example"