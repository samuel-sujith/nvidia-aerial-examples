cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(path_loss_prediction LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT)

set(AERIAL_FRAMEWORK_INCLUDE_DIR "${AERIAL_FRAMEWORK_ROOT}/include")
if(NOT EXISTS ${AERIAL_FRAMEWORK_INCLUDE_DIR})
    message(FATAL_ERROR "Aerial framework headers not found at ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
endif()

add_library(path_loss_predictor
    path_loss_prediction_module.cu
    path_loss_prediction_pipeline.cpp
)

target_include_directories(path_loss_predictor PUBLIC
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(path_loss_predictor
    PUBLIC
        CUDA::cudart
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
        quill::quill
)

if(TensorRT_FOUND)
    if("${TensorRT_VERSION}" MATCHES "stub")
        target_compile_definitions(path_loss_predictor PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1)
    else()
        target_compile_definitions(path_loss_predictor PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0)
    endif()
    target_link_libraries(path_loss_predictor PUBLIC TensorRT::TensorRT)
    target_include_directories(path_loss_predictor PUBLIC ${TensorRT_INCLUDE_DIRS})
else()
    target_compile_definitions(path_loss_predictor PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0)
endif()

set_target_properties(path_loss_predictor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_HOST_COMPILATION_CPP ON
)

add_executable(path_loss_prediction_example
    path_loss_prediction_example.cpp
)

target_link_libraries(path_loss_prediction_example
    path_loss_predictor
    CUDA::cudart
    framework::log
    quill::quill
)

add_executable(path_loss_prediction_module_example
    path_loss_prediction_module_example.cpp
)

target_link_libraries(path_loss_prediction_module_example
    path_loss_predictor
    CUDA::cudart
    framework::log
    quill::quill
)

target_include_directories(path_loss_prediction_example PRIVATE
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(path_loss_prediction_module_example PRIVATE
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(path_loss_predictor PRIVATE -Wall -Wextra -fPIC)
    target_compile_options(path_loss_prediction_example PRIVATE -Wall -Wextra)
    target_compile_options(path_loss_prediction_module_example PRIVATE -Wall -Wextra)
endif()

install(TARGETS path_loss_predictor path_loss_prediction_example path_loss_prediction_module_example
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
