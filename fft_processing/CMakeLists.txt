# FFT Processing Pipeline Example
cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# FFT pipeline library
add_library(fft_pipeline
    fft_pipeline.cpp
)

target_link_libraries(fft_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::cufft
        CUDA::curand
        aerial_framework
        common_utils
)

target_include_directories(fft_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIRS}
)

set_target_properties(fft_pipeline PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations for Release builds
target_compile_options(fft_pipeline PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>
)

# Example executables
add_executable(fft_example
    fft_example.cpp
)

add_executable(fft_processing_example
    fft_processing_example.cpp
)

target_link_libraries(fft_example
    PRIVATE
        fft_pipeline
        CUDA::cudart
        CUDA::cufft
        aerial_framework
        common_utils
)

target_link_libraries(fft_processing_example
    PRIVATE
        fft_pipeline
        CUDA::cudart
        CUDA::cufft
        aerial_framework
        common_utils
)

set_target_properties(fft_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(fft_processing_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS fft_pipeline fft_example fft_processing_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    fft_module.hpp
    fft_pipeline.hpp
    DESTINATION include/fft_processing
)

# Unit tests (if testing framework is available)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS AND TARGET GTest::gtest_main)
    add_executable(fft_processing_tests
        tests/fft_test.cpp
        tests/pipeline_test.cpp
    )
    
    target_link_libraries(fft_processing_tests
        fft_pipeline
        fft_processor
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(fft_processing_tests)
endif()