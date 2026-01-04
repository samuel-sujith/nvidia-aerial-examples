cmake_minimum_required(VERSION 3.18)
project(fft_processing LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set include directories
set(AERIAL_FRAMEWORK_INCLUDE_DIR "${AERIAL_FRAMEWORK_ROOT}/include")

# Check if aerial framework is available
if(NOT EXISTS ${AERIAL_FRAMEWORK_INCLUDE_DIR})
    message(FATAL_ERROR "Aerial framework headers not found at ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
endif()

# Include directories
include_directories(${AERIAL_FRAMEWORK_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# CUDA compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Create the FFT processing module library
add_library(fft_processor 
    fft_processing_module.cu
    fft_processing_pipeline.cpp
)

target_include_directories(fft_processor PUBLIC
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(fft_processor 
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cufft
    framework::pipeline
    framework::tensor
    framework::task
    framework::log
    NamedType
    quill::quill
)

# Set properties for CUDA compilation
set_target_properties(fft_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Create the example executable
add_executable(fft_processing_example
    fft_processing_example.cpp
)

target_link_libraries(fft_processing_example
    fft_processor
    CUDA::cudart
    CUDA::cufft
    framework::log
    quill::quill
)

target_include_directories(fft_processing_example PRIVATE
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(fft_processor PRIVATE -Wall -Wextra -fPIC)
    target_compile_options(fft_processing_example PRIVATE -Wall -Wextra)
endif()

# Install targets
install(TARGETS fft_processor fft_processing_example
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)