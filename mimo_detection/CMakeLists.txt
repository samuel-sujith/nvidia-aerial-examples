# MIMO Detection Pipeline Example
cmake_minimum_required(VERSION 3.20)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# MIMO pipeline library
add_library(mimo_pipeline
    mimo_pipeline_impl.cu
)

target_link_libraries(mimo_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        CUDA::curand
        aerial_framework
        common_utils
)

target_include_directories(mimo_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIRS}
)

set_target_properties(mimo_pipeline PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Enable optimizations for Release builds
target_compile_options(mimo_pipeline PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>>
    $<$<CONFIG:Release>:$<$<COMPILE_LANGUAGE:CUDA>:-O3>>
)

# Example executables
add_executable(mimo_example
    mimo_example.cpp
)

add_executable(mimo_detection_example
    mimo_detection_example.cpp
)

target_link_libraries(mimo_example
    PRIVATE
        mimo_pipeline
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        aerial_framework
        common_utils
)

target_link_libraries(mimo_detection_example
    PRIVATE
        mimo_pipeline
        CUDA::cudart
        CUDA::cublas
        CUDA::cusolver
        aerial_framework
        common_utils
)

set_target_properties(mimo_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(mimo_detection_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS mimo_pipeline mimo_example mimo_detection_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    mimo_detector.hpp
    mimo_pipeline.hpp
    DESTINATION include/mimo_detection
)

# Unit tests (if testing framework is available)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS AND TARGET GTest::gtest_main)
    add_executable(mimo_detection_tests
        tests/mimo_test.cpp
        tests/pipeline_test.cpp
    )
    
    target_link_libraries(mimo_detection_tests
        mimo_pipeline
        mimo_detector
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(mimo_detection_tests)
endif()