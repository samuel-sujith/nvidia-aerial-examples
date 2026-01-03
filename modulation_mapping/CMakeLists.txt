# Modulation Mapping Pipeline Example
# Inherits standards and CUDA setup from parent CMakeLists.txt

# Modulation pipeline library
add_library(modulation_pipeline
    modulator.cu
    modulation_pipeline.cpp
)

target_link_libraries(modulation_pipeline
    PUBLIC
        CUDA::cudart
        CUDA::curand
        common_utils  # This now includes framework targets
)

target_include_directories(modulation_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Inherit properties from parent - no need to set standards again
set_target_properties(modulation_pipeline PROPERTIES
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
)

target_link_libraries(modulation_mapping_example
    PRIVATE
        modulation_pipeline
)

# Properties inherited from parent

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

# Tests (if enabled and directory exists)
if(BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
else()
    message(STATUS "No tests found for modulation_mapping module")
endif()