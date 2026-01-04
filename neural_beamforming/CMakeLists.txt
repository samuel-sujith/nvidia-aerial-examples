cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(neural_beamforming_example
    VERSION 1.0.0
    DESCRIPTION "NVIDIA Aerial Framework - Neural Beamforming Example"
    LANGUAGES CUDA CXX)

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

# Include directories
include_directories(${AERIAL_FRAMEWORK_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# CUDA architecture settings
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

# Core neural beamforming module
add_library(neural_beamforming_module
    neural_beamforming_module.cu
)

target_include_directories(neural_beamforming_module
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(neural_beamforming_module
    PUBLIC
        CUDA::cudart
        CUDA::cufft
        CUDA::cublas
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

target_compile_features(neural_beamforming_module
    PUBLIC cxx_std_20
)

set_target_properties(neural_beamforming_module PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

# Neural beamforming pipeline
add_library(neural_beamforming_pipeline
    neural_beamforming_pipeline.cu
)

target_include_directories(neural_beamforming_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(neural_beamforming_pipeline
    PUBLIC
        neural_beamforming_module
        CUDA::cudart
        CUDA::cufft
        CUDA::cublas
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

target_compile_features(neural_beamforming_pipeline
    PUBLIC cxx_std_20
)

set_target_properties(neural_beamforming_pipeline PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

# Add TensorRT support if available
if(TensorRT_FOUND)
    if("${TensorRT_VERSION}" MATCHES "stub")
        message(STATUS "TensorRT stub found - enabling neural network beamforming with fallback")
        target_compile_definitions(neural_beamforming_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
        target_compile_definitions(neural_beamforming_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
    else()
        message(STATUS "TensorRT found - enabling full neural network beamforming")
        target_compile_definitions(neural_beamforming_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
        target_compile_definitions(neural_beamforming_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
    endif()
    
    target_link_libraries(neural_beamforming_module
        PUBLIC TensorRT::TensorRT
    )
    target_link_libraries(neural_beamforming_pipeline
        PUBLIC TensorRT::TensorRT
    )
    target_include_directories(neural_beamforming_module
        PUBLIC ${TensorRT_INCLUDE_DIRS}
    )
    target_include_directories(neural_beamforming_pipeline
        PUBLIC ${TensorRT_INCLUDE_DIRS}
    )
else()
    message(STATUS "TensorRT not found - neural network beamforming will use fallback")
    target_compile_definitions(neural_beamforming_module
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
    target_compile_definitions(neural_beamforming_pipeline
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
endif()

# Neural beamforming example executable
add_executable(neural_beamforming_example
    neural_beamforming_example.cpp
)

target_include_directories(neural_beamforming_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(neural_beamforming_example
    PRIVATE
        neural_beamforming_pipeline
        neural_beamforming_module
        CUDA::cudart
        CUDA::cufft
        CUDA::cublas
)

target_compile_features(neural_beamforming_example
    PRIVATE cxx_std_20
)

set_target_properties(neural_beamforming_example PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_HOST_COMPILATION_CPP ON
)

# Install targets
install(TARGETS neural_beamforming_example
    RUNTIME DESTINATION bin
)

install(TARGETS neural_beamforming_module neural_beamforming_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    neural_beamforming_module.hpp 
    neural_beamforming_pipeline.hpp
    DESTINATION include
)

# Custom targets for convenience
add_custom_target(run_neural_beamforming
    COMMAND $<TARGET_FILE:neural_beamforming_example> --algorithm MVDR --antennas 64 --users 4
    DEPENDS neural_beamforming_example
    COMMENT "Running neural beamforming example with MVDR algorithm"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run_neural_ml
    COMMAND $<TARGET_FILE:neural_beamforming_example> --algorithm NEURAL --antennas 64 --users 4
    DEPENDS neural_beamforming_example
    COMMENT "Running neural beamforming example with neural network algorithm"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(test_beamforming_gain
    COMMAND $<TARGET_FILE:neural_beamforming_example> --test-gain
    DEPENDS neural_beamforming_example
    COMMENT "Running beamforming gain comparison test"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(test_sinr_performance
    COMMAND $<TARGET_FILE:neural_beamforming_example> --test-sinr --antennas 64 --users 4
    DEPENDS neural_beamforming_example
    COMMENT "Running SINR performance test"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Print configuration summary
message(STATUS "Neural Beamforming Example Configuration:")
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