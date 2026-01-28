cmake_minimum_required(VERSION 3.18)

# Set the project name and CUDA support
project(modulation_mapping_example LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
endif()

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

# Create library for modulation mapping module
add_library(modulation_mapping_module
    modulation_mapping_module.cu
    modulation_mapping_pipeline.cu
)

# Set target properties for the library
set_target_properties(modulation_mapping_module PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Library include directories
target_include_directories(modulation_mapping_module PRIVATE 
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Library compile options
target_compile_options(modulation_mapping_module PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --use_fast_math
        -lineinfo
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
        -O3
        -march=native
    >
)

# Library compile definitions
target_compile_definitions(modulation_mapping_module PRIVATE
    CUDA_API_PER_THREAD_DEFAULT_STREAM
)

# Link CUDA libraries to the library
target_link_libraries(modulation_mapping_module
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
add_executable(modulation_mapping_example
    modulation_mapping_example.cpp
)

# Executable include directories
target_include_directories(modulation_mapping_example PRIVATE 
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Executable compile options
target_compile_options(modulation_mapping_example PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall
        -Wextra
        -O3
        -march=native
    >
)

# Link the modulation mapping library to the example
target_link_libraries(modulation_mapping_example
    modulation_mapping_module
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cufft
    CUDA::cublas
)

# Set output directory
set_target_properties(modulation_mapping_example PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

# Add custom target for running the example with different configurations
add_custom_target(run_modulation_qpsk
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/modulation_mapping_example --scheme QPSK --subcarriers 1200 --symbols 14
    DEPENDS modulation_mapping_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running QPSK modulation example"
)

add_custom_target(run_modulation_16qam
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/modulation_mapping_example --scheme 16QAM --subcarriers 1200 --symbols 14
    DEPENDS modulation_mapping_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running 16-QAM modulation example"
)

add_custom_target(run_modulation_ber_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/modulation_mapping_example --scheme QPSK --test-ber
    DEPENDS modulation_mapping_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running BER performance test"
)

add_custom_target(run_modulation_evm_test
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/modulation_mapping_example --scheme 16QAM --test-evm
    DEPENDS modulation_mapping_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running EVM test"
)

add_custom_target(run_modulation_constellation
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bin/modulation_mapping_example --scheme 16QAM --constellation
    DEPENDS modulation_mapping_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Displaying constellation diagram"
)

# Installation rules
install(TARGETS modulation_mapping_example
    RUNTIME DESTINATION bin
)

install(TARGETS modulation_mapping_module
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    modulation_mapping_module.hpp
    modulation_mapping_pipeline.hpp
    DESTINATION include
)

# Print configuration information
message(STATUS "Modulation Mapping Example Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA_VERSION: ${CUDAToolkit_VERSION}")
message(STATUS "  CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Aerial install path: ${AERIAL_INSTALL_PATH}")