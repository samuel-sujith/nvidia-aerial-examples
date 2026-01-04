cmake_minimum_required(VERSION 3.18)
project(mimo_detection LANGUAGES CXX CUDA)

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

# Create the MIMO detection module library
add_library(mimo_detector 
    mimo_detection_module.cu
    mimo_detection_pipeline.cu
)

target_include_directories(mimo_detector PUBLIC
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(mimo_detector 
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cufft
    CUDA::cublas
    framework::pipeline
    framework::tensor
    framework::task
    framework::log
    NamedType
)

# Create the example executable
add_executable(mimo_detection_example mimo_detection_example.cpp)

target_link_libraries(mimo_detection_example 
    mimo_detector
)

# Installation
install(TARGETS mimo_detection_example 
    DESTINATION bin
)

# Print configuration
message(STATUS "MIMO Detection module configured successfully")