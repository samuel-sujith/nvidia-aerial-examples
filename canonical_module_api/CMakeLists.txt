cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(canonical_module_api LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(AERIAL_FRAMEWORK_INCLUDE_DIR "${AERIAL_FRAMEWORK_ROOT}/include")
if(NOT EXISTS ${AERIAL_FRAMEWORK_INCLUDE_DIR})
    message(FATAL_ERROR "Aerial framework headers not found at ${AERIAL_FRAMEWORK_INCLUDE_DIR}")
endif()

add_executable(canonical_module_api_example
    canonical_module_api_example.cpp
)

target_include_directories(canonical_module_api_example PRIVATE
    ${AERIAL_FRAMEWORK_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../channel_estimation
)

target_link_libraries(canonical_module_api_example
    PRIVATE
        channel_estimator
        CUDA::cudart
        framework::log
        quill::quill
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(canonical_module_api_example PRIVATE -Wall -Wextra)
endif()

install(TARGETS canonical_module_api_example
    RUNTIME DESTINATION bin
)
