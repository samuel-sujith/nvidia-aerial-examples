
cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(channel_estimation LANGUAGES CXX CUDA)


# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# Add custom module path for TensorRT
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT)


# Set include directories
set(AERIAL_FRAMEWORK_INCLUDE_DIR "${AERIAL_FRAMEWORK_ROOT}/include")

# Check if aerial framework is available
if(NOT EXISTS ${AERIAL_FRAMEWORK_INCLUDE_DIR})
    message(FATAL_ERROR "Aerial framework headers not found at ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
endif()

include_directories(${AERIAL_FRAMEWORK_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")


# Core channel estimation module
add_library(channel_estimator 
    channel_estimation_module.cu
    channel_estimation_pipeline.cpp
    ml_channel_estimator_tensorrt.cu
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

set_target_properties(channel_estimator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_HOST_COMPILATION_CPP ON
)


# Add TensorRT support if available
if(TensorRT_FOUND)
    if("${TensorRT_VERSION}" MATCHES "stub")
        message(STATUS "TensorRT stub found - enabling ML channel estimation with fallback")
        target_compile_definitions(channel_estimator
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
    else()
        message(STATUS "TensorRT found - enabling full ML channel estimation")
        target_compile_definitions(channel_estimator
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
    endif()
    target_link_libraries(channel_estimator PUBLIC TensorRT::TensorRT)
    target_include_directories(channel_estimator PUBLIC ${TensorRT_INCLUDE_DIRS})
else()
    message(STATUS "TensorRT not found - ML channel estimation will use fallback")
    target_compile_definitions(channel_estimator
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
endif()

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

# Print configuration summary
message(STATUS "Channel Estimation Example Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "  AERIAL_FRAMEWORK_INCLUDE_DIR: ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
if(TensorRT_FOUND)
    message(STATUS "  TensorRT Version: ${TensorRT_VERSION}")
    message(STATUS "  TensorRT Include: ${TensorRT_INCLUDE_DIRS}")
    if("${TensorRT_VERSION}" MATCHES "stub")
        message(STATUS "  TensorRT Type: Stub implementation (for development/testing)")
    else()
        message(STATUS "  TensorRT Type: Official NVIDIA implementation")
        message(STATUS "  TensorRT Libraries: ${TensorRT_LIBRARIES}")
    endif()
else()
    message(STATUS "  TensorRT: Not available - using fallback implementation")
endif()