#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.20)

project(ChannelEstimationExample 
    VERSION 1.0.0
    DESCRIPTION "Channel Estimation Pipeline Example for Aerial Framework"
    LANGUAGES CXX CUDA
)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Include framework headers (adjust paths as needed)
set(FRAMEWORK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
include_directories(
    ${FRAMEWORK_ROOT}/task/lib/include
    ${FRAMEWORK_ROOT}/tensor/lib/include  
    ${FRAMEWORK_ROOT}/pipeline/lib/include
    ${FRAMEWORK_ROOT}/memory/lib/include
    ${FRAMEWORK_ROOT}/utils/lib/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
endif()

# CUDA compiler options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90") # Support multiple GPU architectures

# Define CUDA separable compilation for device code
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Channel Estimator Library
add_library(channel_estimator SHARED
    channel_estimator.cu
    channel_estimator.hpp
)

target_link_libraries(channel_estimator
    CUDA::cudart
    CUDA::cublas  # If using cuBLAS for MMSE
    CUDA::cufft   # If using FFT for frequency domain operations
)

target_include_directories(channel_estimator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA properties
set_property(TARGET channel_estimator PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Channel Estimation Pipeline Library  
add_library(channel_est_pipeline SHARED
    channel_est_pipeline.cpp
    channel_est_pipeline.hpp
)

target_link_libraries(channel_est_pipeline
    channel_estimator
    # Link to framework libraries (adjust as needed)
    # framework_task
    # framework_tensor
    # framework_pipeline
    # framework_memory
)

target_include_directories(channel_est_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Example executable
add_executable(channel_estimation_example
    channel_estimation_example.cpp
)

target_link_libraries(channel_estimation_example
    channel_est_pipeline
    channel_estimator
    CUDA::cudart
)

# Performance test executable
add_executable(channel_estimation_benchmark
    channel_estimation_benchmark.cpp  # Additional benchmark file
)

target_link_libraries(channel_estimation_benchmark
    channel_est_pipeline
    channel_estimator
    CUDA::cudart
)

# Unit tests (if testing framework is available)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS AND TARGET GTest::gtest_main)
    add_executable(channel_estimation_tests
        tests/channel_estimator_test.cpp
        tests/channel_pipeline_test.cpp
    )
    
    target_link_libraries(channel_estimation_tests
        channel_est_pipeline
        channel_estimator
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(channel_estimation_tests)
endif()

# Installation
install(TARGETS channel_estimator channel_est_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    channel_estimator.hpp
    channel_est_pipeline.hpp
    DESTINATION include/aerial/examples/channel_estimation
)

install(TARGETS channel_estimation_example channel_estimation_benchmark
    RUNTIME DESTINATION bin
)

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            @ONLY
        )
        
        add_custom_target(channel_estimation_docs ALL
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
            DESTINATION share/doc/aerial-channel-estimation
        )
    endif()
endif()

# Packaging
set(CPACK_PACKAGE_NAME "aerial-channel-estimation-example")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "NVIDIA Corporation")

include(CPack)