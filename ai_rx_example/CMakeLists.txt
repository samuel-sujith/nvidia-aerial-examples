

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(ai_rx_example
    VERSION 1.0.0
    DESCRIPTION "NVIDIA Aerial Framework - AI Rx Example"
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
if(NOT EXISTS ${AERIAL_FRAMEWORK_INCLUDE_DIR})
    message(FATAL_ERROR "Aerial framework headers not found at ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
endif()
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

# Core AI Rx module
add_library(ai_rx_module
    ai_rx_module.cu
    ai_rx_module.hpp
)

target_include_directories(ai_rx_module
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(ai_rx_module
    PUBLIC
        CUDA::cudart
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

target_compile_features(ai_rx_module
    PUBLIC cxx_std_20
)

set_target_properties(ai_rx_module PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

# AI Rx pipeline
add_library(ai_rx_pipeline
    ai_rx_pipeline.cpp
    ai_rx_pipeline.hpp
)

target_include_directories(ai_rx_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(ai_rx_pipeline
    PUBLIC
        ai_rx_module
        CUDA::cudart
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

target_compile_features(ai_rx_pipeline
    PUBLIC cxx_std_20
)

set_target_properties(ai_rx_pipeline PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

# Add TensorRT support if available
if(TensorRT_FOUND)
    if("${TensorRT_VERSION}" MATCHES "stub")
        message(STATUS "TensorRT stub found - enabling AI Rx with fallback")
        target_compile_definitions(ai_rx_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
        target_compile_definitions(ai_rx_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
    else()
        message(STATUS "TensorRT found - enabling full AI Rx inference")
        target_compile_definitions(ai_rx_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
        target_compile_definitions(ai_rx_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
    endif()
    target_link_libraries(ai_rx_module
        PUBLIC TensorRT::TensorRT
    )
    target_link_libraries(ai_rx_pipeline
        PUBLIC TensorRT::TensorRT
    )
    target_include_directories(ai_rx_module
        PUBLIC ${TensorRT_INCLUDE_DIRS}
    )
    target_include_directories(ai_rx_pipeline
        PUBLIC ${TensorRT_INCLUDE_DIRS}
    )
else()
    message(STATUS "TensorRT not found - AI Rx will use fallback")
    target_compile_definitions(ai_rx_module
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
    target_compile_definitions(ai_rx_pipeline
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
endif()

# AI Rx example executable
add_executable(ai_rx_example
    ai_rx_example.cpp
)

target_include_directories(ai_rx_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(ai_rx_example
    PRIVATE
        ai_rx_pipeline
        ai_rx_module
        CUDA::cudart
)

target_compile_features(ai_rx_example
    PRIVATE cxx_std_20
)

set_target_properties(ai_rx_example PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_HOST_COMPILATION_CPP ON
)

# Install targets
install(TARGETS ai_rx_example
    RUNTIME DESTINATION bin
)

install(TARGETS ai_rx_module ai_rx_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES 
    ai_rx_module.hpp 
    ai_rx_pipeline.hpp
    DESTINATION include
)

# Print configuration summary
message(STATUS "AI Rx Example Configuration:")
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
