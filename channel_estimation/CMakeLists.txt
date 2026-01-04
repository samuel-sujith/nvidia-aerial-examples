cmake_minimum_required(VERSION 3.18)
project(channel_estimation LANGUAGES CXX CUDA)

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

# CUDA compile options
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70 -std=c++20")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Create the channel estimation module library
add_library(channel_estimator 
    channel_estimation_module.cu
    channel_estimation_pipeline.cpp
)

target_include_directories(channel_estimator PUBLIC
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(channel_estimator 
    CUDA::cudart
    CUDA::cuda_driver
    framework::pipeline
    framework::tensor
    framework::task
    framework::log
    NamedType
    quill::quill
)

# Set properties for CUDA compilation
set_target_properties(channel_estimator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Create the example executable
add_executable(channel_estimation_example
    channel_estimation_example.cpp
)

target_link_libraries(channel_estimation_example
    channel_estimator
    CUDA::cudart
    framework::log
    quill::quill
)

target_include_directories(channel_estimation_example PRIVATE
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(channel_estimator PRIVATE -Wall -Wextra -fPIC)
    target_compile_options(channel_estimation_example PRIVATE -Wall -Wextra)
endif()

# Install targets
install(TARGETS channel_estimator channel_estimation_example
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)