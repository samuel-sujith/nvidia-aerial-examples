# Neural Beamforming Pipeline with ML Integration
# Inherits standards and CUDA setup from parent CMakeLists.txt

add_library(neural_beamforming_pipeline STATIC
    neural_beamforming_pipeline.cpp
)

target_include_directories(neural_beamforming_pipeline
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(neural_beamforming_pipeline
    PRIVATE
        common_utils  # This now includes framework targets
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::cusolver
)

# Properties inherited from parent - no need to set standards again

# Set CUDA properties for neural network operations
set_property(TARGET neural_beamforming_pipeline PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Add CUDA compilation flags for ML optimization
target_compile_options(neural_beamforming_pipeline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -v
        --use_fast_math
        -lineinfo
    >
)

# Neural Beamforming Simple Example
add_executable(neural_beamforming_example
    neural_beamforming_example.cpp
)

target_link_libraries(neural_beamforming_example
    PRIVATE neural_beamforming_pipeline
)

# Comprehensive ML Example
add_executable(neural_beamforming_ml_example
    neural_beamforming_ml_example.cpp
)

target_link_libraries(neural_beamforming_ml_example
    PRIVATE neural_beamforming_pipeline
)

# Installation rules
install(TARGETS neural_beamforming_pipeline neural_beamforming_example neural_beamforming_ml_example
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES neural_beamforming_pipeline.hpp
    DESTINATION include/neural_beamforming
)

# Add custom targets for ML-specific operations
add_custom_target(neural_beamforming_train
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/neural_beamforming_ml_example --train
    DEPENDS neural_beamforming_ml_example
    COMMENT "Running neural beamforming training pipeline"
)

add_custom_target(neural_beamforming_benchmark
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/neural_beamforming_ml_example --benchmark
    DEPENDS neural_beamforming_ml_example
    COMMENT "Running neural beamforming benchmarks"
)

add_custom_target(neural_beamforming_export
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/neural_beamforming_ml_example --export-models
    DEPENDS neural_beamforming_ml_example
    COMMENT "Exporting TensorRT models for deployment"
)