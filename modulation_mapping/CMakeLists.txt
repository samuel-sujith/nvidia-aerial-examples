# Modulation Mapping Pipeline Example
cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Modulation pipeline library
add_library(modulation_pipeline
    modulator.cu
    modulation_pipeline.cpp
)

target_link_libraries(modulation_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::curand
        aerial_framework
        common_utils
)

target_include_directories(modulation_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIRS}
)

set_target_properties(modulation_pipeline PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations for Release builds
target_compile_options(modulation_pipeline PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>
)

# Example executables
add_executable(modulation_example
    modulation_example.cpp
)

add_executable(modulation_mapping_example
    modulation_mapping_example.cpp
)

target_link_libraries(modulation_example
    PRIVATE
        modulation_pipeline
        CUDA::cudart
        aerial_framework
        common_utils
)

target_link_libraries(modulation_mapping_example
    PRIVATE
        modulation_pipeline
        CUDA::cudart
        CUDA::curand
        aerial_framework
        common_utils
)

set_target_properties(modulation_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(modulation_mapping_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS modulation_pipeline modulation_example modulation_mapping_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    modulator.hpp
    modulation_pipeline.hpp
    DESTINATION include/modulation_mapping
)

# Tests (if enabled)
if(BUILD_TESTS AND TARGET GTest::gtest_main AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_executable(modulation_mapping_tests
        tests/modulation_test.cpp
        tests/pipeline_test.cpp
    )
    
    target_link_libraries(modulation_mapping_tests
        modulation_pipeline
        modulator
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(modulation_mapping_tests)
else()
    message(STATUS "No tests directory found or GTest not available, skipping unit tests")
endif()