cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(user_scheduling
    VERSION 1.0.0
    DESCRIPTION "NVIDIA Aerial Framework - User Scheduling Example"
    LANGUAGES CUDA CXX)

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
include_directories(${AERIAL_FRAMEWORK_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")

add_library(user_scheduling_module
    user_scheduling_module.cu
    user_scheduling_module.hpp
)

target_include_directories(user_scheduling_module
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(user_scheduling_module
    PUBLIC
        CUDA::cudart
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

target_compile_features(user_scheduling_module
    PUBLIC cxx_std_20
)

set_target_properties(user_scheduling_module PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

add_library(user_scheduling_pipeline
    user_scheduling_pipeline.cpp
    user_scheduling_pipeline.hpp
)

target_include_directories(user_scheduling_pipeline
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(user_scheduling_pipeline
    PUBLIC
        user_scheduling_module
        CUDA::cudart
        framework::pipeline
        framework::tensor
        framework::task
        framework::log
        NamedType
)

if(TensorRT_FOUND)
    if("${TensorRT_VERSION}" MATCHES "stub")
        message(STATUS "TensorRT stub found - enabling user scheduling with fallback")
        target_compile_definitions(user_scheduling_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
        target_compile_definitions(user_scheduling_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=1
        )
    else()
        message(STATUS "TensorRT found - enabling user scheduling inference")
        target_compile_definitions(user_scheduling_module
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
        target_compile_definitions(user_scheduling_pipeline
            PUBLIC TENSORRT_AVAILABLE=1 TENSORRT_STUB=0
        )
    endif()
    target_link_libraries(user_scheduling_module PUBLIC TensorRT::TensorRT)
    target_link_libraries(user_scheduling_pipeline PUBLIC TensorRT::TensorRT)
    target_include_directories(user_scheduling_module PUBLIC ${TensorRT_INCLUDE_DIRS})
    target_include_directories(user_scheduling_pipeline PUBLIC ${TensorRT_INCLUDE_DIRS})
else()
    message(STATUS "TensorRT not found - user scheduling will use fallback")
    target_compile_definitions(user_scheduling_module
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
    target_compile_definitions(user_scheduling_pipeline
        PUBLIC TENSORRT_AVAILABLE=0 TENSORRT_STUB=0
    )
endif()

target_compile_features(user_scheduling_pipeline
    PUBLIC cxx_std_20
)

set_target_properties(user_scheduling_pipeline PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_HOST_COMPILATION_CPP ON
)

add_executable(user_scheduling_example
    user_scheduling_example.cpp
)

target_include_directories(user_scheduling_example
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${AERIAL_FRAMEWORK_INCLUDE_DIR}
)

target_link_libraries(user_scheduling_example
    PRIVATE
        user_scheduling_pipeline
        user_scheduling_module
        CUDA::cudart
)

target_compile_features(user_scheduling_example
    PRIVATE cxx_std_20
)

set_target_properties(user_scheduling_example PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_HOST_COMPILATION_CPP ON
)

install(TARGETS user_scheduling_example
    RUNTIME DESTINATION bin
)

install(TARGETS user_scheduling_module user_scheduling_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    user_scheduling_module.hpp
    user_scheduling_pipeline.hpp
    DESTINATION include
)

message(STATUS "User Scheduling Example Configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "  AERIAL_FRAMEWORK_INCLUDE_DIR: ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
if(TensorRT_FOUND)
    message(STATUS "  TensorRT Version: ${TensorRT_VERSION}")
    message(STATUS "  TensorRT Include: ${TensorRT_INCLUDE_DIRS}")
else()
    message(STATUS "  TensorRT: Not available - using fallback implementation")
endif()
