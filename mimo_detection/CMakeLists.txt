# MIMO Detection Pipeline Example
# Inherits standards and CUDA setup from parent CMakeLists.txt

# MIMO pipeline library
add_library(mimo_pipeline
    mimo_pipeline_impl.cu
)

target_link_libraries(mimo_pipeline
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
    CUDA::curand
    common_utils  # This now includes framework targets
)

# Link Quill directly if available (needed for framework headers)
if(TARGET quill::quill)
    target_link_libraries(mimo_pipeline quill::quill)
endif()

target_include_directories(mimo_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Inherit properties from parent - no need to set standards again
set_target_properties(mimo_pipeline PROPERTIES
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
    mimo_detection_example.cpp)

target_link_libraries(mimo_example
    mimo_pipeline
)

target_link_libraries(mimo_detection_example
    mimo_pipeline
)

# Properties inherited from parent

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

# Tests (if enabled and directory exists)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
else()
    message(STATUS "No tests found for mimo_detection module")
endif()