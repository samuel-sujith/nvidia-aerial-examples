# FFT Processing Pipeline Example
# Inherits standards and CUDA setup from parent CMakeLists.txt

# FFT pipeline library
add_library(fft_pipeline
    fft_pipeline.cpp
)

target_link_libraries(fft_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::cufft
        CUDA::curand
        common_utils  # This now includes framework targets
)

target_include_directories(fft_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Inherit properties from parent - no need to set standards again
set_target_properties(fft_pipeline PROPERTIES
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

# Tests (if enabled and directory exists)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
else()
    message(STATUS "No tests found for fft_processing module")
endif()