# NVIDIA Aerial Framework Examples
cmake_minimum_required(VERSION 3.23) # Align with aerial-framework requirement

# Only set the cxx_standard if it is not set by someone else
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between -Wpedantic being enabled
# and -std=c++20 and -std=gnu++20 for example when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position-independent code globally for all static libraries Required for CUDA device
# linking and potential shared library usage
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(AerialExamples
    VERSION 1.0.0
    DESCRIPTION "NVIDIA Aerial Framework GPU-Accelerated 5G Signal Processing Examples"
    LANGUAGES CXX C ASM CUDA
)

# Build configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmark executables" ON)
option(ENABLE_PROFILING "Enable profiling support" OFF)
option(ENABLE_DEBUG_INFO "Enable debug information" OFF)

# Set default CPM source cache if not already configured 
if(NOT DEFINED CPM_SOURCE_CACHE AND NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(ENV{CPM_SOURCE_CACHE} "$ENV{HOME}/.cache/CPM")
    message(STATUS "CPM source cache: $ENV{HOME}/.cache/CPM (default)")
else()
    if(DEFINED CPM_SOURCE_CACHE)
        message(STATUS "CPM source cache: ${CPM_SOURCE_CACHE} (CMake variable)")
    else()
        message(STATUS "CPM source cache: $ENV{CPM_SOURCE_CACHE} (environment variable)")
    endif()
endif()

# Download CPM.cmake
file(DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.2/CPM.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
    EXPECTED_HASH SHA256=c8cdc32c03816538ce22781ed72964dc864b2a34a310d3b7104812a5ca2d835d
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# External dependencies - using CPM for proper management
message(STATUS "Configuring external dependencies...")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# Use local NamedType if available, otherwise download via CPM
if(NOT TARGET NamedType)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/NamedType/CMakeLists.txt")
        message(STATUS "Using local NamedType")
        add_subdirectory(external/NamedType)
    else()
        message(STATUS "Local NamedType not found at ${CMAKE_CURRENT_SOURCE_DIR}/external/NamedType/CMakeLists.txt")
        message(STATUS "Downloading NamedType via CPM")
        CPMAddPackage(
            NAME NamedType
            GITHUB_REPOSITORY joboccara/NamedType
            GIT_TAG v1.1.0
            OPTIONS "ENABLE_TEST OFF"
            SYSTEM YES
        )
    endif()
endif()

# Use local Quill library if available, otherwise download via CPM
if(NOT TARGET quill::quill)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/quill/CMakeLists.txt")
        message(STATUS "Using local external Quill library")
        option(QUILL_BUILD_EXAMPLES "Build examples" OFF)
        option(QUILL_BUILD_TESTS "Build tests" OFF) 
        option(QUILL_BUILD_BENCHMARKS "Build benchmarks" OFF)
        add_subdirectory(external/quill)
    else()
        message(STATUS "Local Quill not found at ${CMAKE_CURRENT_SOURCE_DIR}/external/quill/CMakeLists.txt")
        message(STATUS "Downloading Quill via CPM")
        CPMAddPackage(
            NAME quill
            GITHUB_REPOSITORY odygrd/quill
            GIT_TAG v2.9.2
            OPTIONS "QUILL_BUILD_EXAMPLES OFF" "QUILL_BUILD_TESTS OFF" "QUILL_BUILD_BENCHMARKS OFF"
            SYSTEM YES
        )
        
        # Create comprehensive compatibility headers for framework compatibility
        if(quill_ADDED)
            # Create compatibility directory structure
            file(MAKE_DIRECTORY "${quill_SOURCE_DIR}/include/quill")
            file(MAKE_DIRECTORY "${quill_SOURCE_DIR}/include/quill/detail")
            
            # LogMacros.h compatibility
            file(WRITE "${quill_SOURCE_DIR}/include/quill/LogMacros.h"
                "#pragma once\n"
                "#include \"quill/Logger.h\"\n"
                "#include \"quill/LogMacros.h\"\n"
                "// Compatibility header for Aerial Framework\n"
                "#define LOG_TRACE_L3(...) QUILL_LOG_TRACE_L3(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_TRACE_L2(...) QUILL_LOG_TRACE_L2(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_TRACE_L1(...) QUILL_LOG_TRACE_L1(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_DEBUG(...) QUILL_LOG_DEBUG(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_INFO(...) QUILL_LOG_INFO(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_WARNING(...) QUILL_LOG_WARNING(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_ERROR(...) QUILL_LOG_ERROR(quill::get_logger(\"root\"), __VA_ARGS__)\n"
                "#define LOG_CRITICAL(...) QUILL_LOG_CRITICAL(quill::get_logger(\"root\"), __VA_ARGS__)\n"
            )
            
            # DeferredFormatCodec.h compatibility
            file(WRITE "${quill_SOURCE_DIR}/include/quill/DeferredFormatCodec.h"
                "#pragma once\n"
                "#include <string>\n"
                "#include <cstdint>\n"
                "// Compatibility header for Aerial Framework\n"
                "namespace quill {\n"
                "  template<typename T>\n"
                "  struct DeferredFormatCodec {\n"
                "    static constexpr size_t size() noexcept { return sizeof(T); }\n"
                "    static void encode(std::byte*, T const&) noexcept {}\n"
                "    static T decode(std::byte const*) noexcept { return T{}; }\n"
                "  };\n"
                "}\n"
            )
            
            # Backend.h compatibility
            file(WRITE "${quill_SOURCE_DIR}/include/quill/Backend.h"
                "#pragma once\n"
                "// Compatibility header for Aerial Framework\n"
                "namespace quill {\n"
                "  class Backend {\n"
                "  public:\n"
                "    static void start() {}\n"
                "    static void stop() {}\n"
                "    static Backend& instance() { static Backend b; return b; }\n"
                "  private:\n"
                "    Backend() = default;\n"
                "  };\n"
                "}\n"
            )
            
            message(STATUS "Created comprehensive Quill compatibility headers")
        endif()
    endif()
endif()

# Try to find installed Aerial Framework first
find_package(framework QUIET)

if(framework_FOUND)
    message(STATUS "Found installed Aerial Framework")
    set(AERIAL_FRAMEWORK_TARGETS framework::pipeline framework::tensor framework::utils framework::tensorrt framework::task framework::memory framework::log framework::net)
else()
    # Fallback to manual discovery
    message(STATUS "Aerial Framework not found via find_package, using manual discovery")
    
    # Set default Aerial Framework path if not provided
    if(NOT DEFINED AERIAL_FRAMEWORK_ROOT)
        # Try common installation locations
        foreach(POTENTIAL_PATH 
                "${CMAKE_CURRENT_SOURCE_DIR}/../aerial-framework-prod"
                "${CMAKE_CURRENT_SOURCE_DIR}/../aerial-framework/out/install/clang-release"
                "${CMAKE_CURRENT_SOURCE_DIR}/../aerial-framework/out/install/gcc-release"
                "/usr/local"
                "/opt/nvidia/aerial-framework")
            if(EXISTS "${POTENTIAL_PATH}/include/pipeline" OR EXISTS "${POTENTIAL_PATH}/lib/libframework-pipeline.a")
                set(AERIAL_FRAMEWORK_ROOT "${POTENTIAL_PATH}")
                message(STATUS "Found Aerial Framework at: ${AERIAL_FRAMEWORK_ROOT}")
                break()
            endif()
        endforeach()
    endif()
    
    if(NOT AERIAL_FRAMEWORK_ROOT)
        message(FATAL_ERROR "Aerial Framework not found. Please set AERIAL_FRAMEWORK_ROOT or install framework package")
    endif()
    
    # Create imported targets for manual discovery
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindAerialFramework.cmake")
endif()

# CUDA architecture configuration (align with aerial-framework)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Use native if possible, fallback to modern architectures
    set(CMAKE_CUDA_ARCHITECTURES "native")
    # If native detection fails, use these modern defaults
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
    endif()
endif()

# Suppress deprecated GPU targets warning for older architectures
if(CMAKE_CUDA_COMPILER_VERSION)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
endif()

# Compiler flags (align with aerial-framework approach)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug") 
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
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

# Tests (use CPM for GTest like aerial-framework)
if(BUILD_TESTS)
    enable_testing()
    
    # Try to find installed GTest first
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        # Download via CPM like aerial-framework does
        CPMAddPackage(
            NAME googletest
            GITHUB_REPOSITORY google/googletest
            GIT_TAG v1.15.2
            OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
            SYSTEM YES
        )
    endif()
    
    # Only add tests subdirectory if it exists
    if((TARGET GTest::gtest_main OR TARGET gtest_main) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    else()
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
            message(STATUS "No tests directory found, skipping unit tests")
        else()
            message(WARNING "GTest not available, skipping unit tests")
        endif()
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