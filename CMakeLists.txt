# NVIDIA Aerial Framework Examples
cmake_minimum_required(VERSION 3.20)

project(AerialExamples
    VERSION 1.0.0
    DESCRIPTION "NVIDIA Aerial Framework GPU-Accelerated 5G Signal Processing Examples"
    LANGUAGES CXX CUDA
)

# C++20 and CUDA requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(ENABLE_DEBUG_INFO "Enable debug information" OFF)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set default Aerial Framework path if not provided
if(NOT DEFINED AERIAL_FRAMEWORK_ROOT)
    set(AERIAL_FRAMEWORK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../aerial-framework-prod")
endif()

# Aerial Framework include directories
set(AERIAL_FRAMEWORK_INCLUDE_DIRS
    ${AERIAL_FRAMEWORK_ROOT}/include
    ${AERIAL_FRAMEWORK_ROOT}/include/pipeline
    ${AERIAL_FRAMEWORK_ROOT}/include/tensor
    ${AERIAL_FRAMEWORK_ROOT}/include/task
    ${AERIAL_FRAMEWORK_ROOT}/include/memory
    ${AERIAL_FRAMEWORK_ROOT}/include/utils
    ${AERIAL_FRAMEWORK_ROOT}/include/log
)

# Find framework libraries
find_library(FRAMEWORK_PIPELINE_LIB
    NAMES framework-pipeline
    PATHS ${AERIAL_FRAMEWORK_ROOT}/lib
    REQUIRED
)

find_library(FRAMEWORK_TENSOR_LIB
    NAMES framework-tensor
    PATHS ${AERIAL_FRAMEWORK_ROOT}/lib
    REQUIRED
)

find_library(FRAMEWORK_UTILS_LIB
    NAMES framework-utils
    PATHS ${AERIAL_FRAMEWORK_ROOT}/lib
    REQUIRED
)

find_library(FRAMEWORK_TENSORRT_LIB
    NAMES framework-tensorrt
    PATHS ${AERIAL_FRAMEWORK_ROOT}/lib
    REQUIRED
)

find_library(FRAMEWORK_TASK_LIB
    NAMES task
    PATHS ${AERIAL_FRAMEWORK_ROOT}/lib
    REQUIRED
)

# Create imported target for Aerial Framework
add_library(aerial_framework INTERFACE)
target_include_directories(aerial_framework INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
target_link_libraries(aerial_framework INTERFACE 
    ${FRAMEWORK_PIPELINE_LIB}
    ${FRAMEWORK_TENSOR_LIB}
    ${FRAMEWORK_UTILS_LIB}
    ${FRAMEWORK_TENSORRT_LIB}
    ${FRAMEWORK_TASK_LIB}
)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG")

# CUDA architecture configuration
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

# Global include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

# Common utilities library
add_subdirectory(common)

# Example modules
add_subdirectory(channel_estimation)
add_subdirectory(modulation_mapping)
add_subdirectory(fft_processing)
add_subdirectory(mimo_detection)
add_subdirectory(neural_beamforming)

# Tests
if(BUILD_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(GTest_FOUND)
        add_subdirectory(tests)
    else()
        message(WARNING "GTest not found, skipping unit tests")
    endif()
endif()

# Documentation
find_package(Doxygen QUIET)
if(Doxygen_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/docs/CMakeLists.txt")
    add_subdirectory(docs)
else()
    message(STATUS "Doxygen not found or no docs CMakeLists.txt. Skipping documentation build.")
endif()

# Installation
include(GNUInstallDirs)

install(DIRECTORY docs/
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    FILES_MATCHING PATTERN "*.md"
)

install(FILES
    README.md
    LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
)

# Package configuration (simplified)
include(CMakePackageConfigHelpers)

# Create a simple config file
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/AerialExamplesConfig.cmake"
    "# Aerial Examples Package Configuration\n"
    "find_dependency(CUDAToolkit REQUIRED)\n"
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/AerialExamplesConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/AerialExamplesConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/AerialExamplesConfigVersion.cmake
    DESTINATION lib/cmake/AerialExamples
)

# Summary
message(STATUS "")
message(STATUS "============== NVIDIA Aerial Examples Build Configuration ==============")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "Aerial Framework: ${AERIAL_FRAMEWORK_INCLUDE_DIRS}")
message(STATUS "========================================================================")
message(STATUS "")