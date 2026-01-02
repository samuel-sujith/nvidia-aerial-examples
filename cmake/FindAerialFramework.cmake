# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# FindAerialFramework.cmake
# 
# Manual discovery and import of Aerial Framework libraries when package config is not available

# Validate AERIAL_FRAMEWORK_ROOT was set
if(NOT AERIAL_FRAMEWORK_ROOT)
    message(FATAL_ERROR "AERIAL_FRAMEWORK_ROOT must be set for manual discovery")
endif()

# Determine include directory structure
if(EXISTS "${AERIAL_FRAMEWORK_ROOT}/include/framework")
    set(FRAMEWORK_INCLUDE_BASE "${AERIAL_FRAMEWORK_ROOT}/include/framework")
else()
    set(FRAMEWORK_INCLUDE_BASE "${AERIAL_FRAMEWORK_ROOT}/include")
endif()

# Aerial Framework include directories
set(AERIAL_FRAMEWORK_INCLUDE_DIRS
    ${FRAMEWORK_INCLUDE_BASE}
    ${FRAMEWORK_INCLUDE_BASE}/pipeline
    ${FRAMEWORK_INCLUDE_BASE}/tensor
    ${FRAMEWORK_INCLUDE_BASE}/task
    ${FRAMEWORK_INCLUDE_BASE}/memory
    ${FRAMEWORK_INCLUDE_BASE}/utils
    ${FRAMEWORK_INCLUDE_BASE}/log
    ${FRAMEWORK_INCLUDE_BASE}/net
    ${AERIAL_FRAMEWORK_ROOT}/include/wise_enum
)

# Find framework libraries with multiple naming conventions
function(find_framework_library VAR_NAME)
    set(LIB_NAMES ${ARGN})
    set(LIB_PATHS 
        ${AERIAL_FRAMEWORK_ROOT}/lib
        ${AERIAL_FRAMEWORK_ROOT}/lib64
        ${AERIAL_FRAMEWORK_ROOT}/out/build/*/framework/*/
        ${AERIAL_FRAMEWORK_ROOT}/framework/*/
    )
    
    foreach(LIB_NAME ${LIB_NAMES})
        find_library(${VAR_NAME}
            NAMES ${LIB_NAME}
            PATHS ${LIB_PATHS}
            PATH_SUFFIXES lib lib64
        )
        if(${VAR_NAME})
            break()
        endif()
    endforeach()
    
    if(NOT ${VAR_NAME})
        message(WARNING "Could not find library: ${LIB_NAMES}")
    endif()
endfunction()

# Find all framework libraries
find_framework_library(FRAMEWORK_PIPELINE_LIB framework-pipeline libframework-pipeline pipeline)
find_framework_library(FRAMEWORK_TENSOR_LIB framework-tensor libframework-tensor tensor)
find_framework_library(FRAMEWORK_UTILS_LIB framework-utils libframework-utils utils)
find_framework_library(FRAMEWORK_TENSORRT_LIB framework-tensorrt libframework-tensorrt tensorrt)
find_framework_library(FRAMEWORK_TASK_LIB framework-task libframework-task task)
find_framework_library(FRAMEWORK_MEMORY_LIB framework-memory libframework-memory memory)
find_framework_library(FRAMEWORK_LOG_LIB framework-log libframework-log log)
find_framework_library(FRAMEWORK_NET_LIB framework-net libframework-net net)

# Create imported targets
function(create_framework_target TARGET_NAME LIBRARY_VAR)
    if(${LIBRARY_VAR})
        add_library(framework::${TARGET_NAME} SHARED IMPORTED)
        set_target_properties(framework::${TARGET_NAME} PROPERTIES
            IMPORTED_LOCATION ${${LIBRARY_VAR}}
            INTERFACE_INCLUDE_DIRECTORIES "${AERIAL_FRAMEWORK_INCLUDE_DIRS}"
        )
        message(STATUS "Created target framework::${TARGET_NAME} -> ${${LIBRARY_VAR}}")
    else()
        # Create stub interface target if library not found
        add_library(framework::${TARGET_NAME} INTERFACE)
        target_include_directories(framework::${TARGET_NAME} INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
        message(WARNING "Created stub target framework::${TARGET_NAME} (library not found)")
    endif()
endfunction()

# Create all framework targets
create_framework_target(pipeline FRAMEWORK_PIPELINE_LIB)
create_framework_target(tensor FRAMEWORK_TENSOR_LIB)
create_framework_target(utils FRAMEWORK_UTILS_LIB) 
create_framework_target(tensorrt FRAMEWORK_TENSORRT_LIB)
create_framework_target(task FRAMEWORK_TASK_LIB)
create_framework_target(memory FRAMEWORK_MEMORY_LIB)
create_framework_target(log FRAMEWORK_LOG_LIB)
create_framework_target(net FRAMEWORK_NET_LIB)

# Set up dependencies between framework targets
if(TARGET framework::pipeline)
    target_link_libraries(framework::pipeline INTERFACE 
        framework::tensor framework::utils framework::memory)
endif()

if(TARGET framework::tensor)
    target_link_libraries(framework::tensor INTERFACE 
        framework::utils framework::memory)
endif()

if(TARGET framework::tensorrt)
    target_link_libraries(framework::tensorrt INTERFACE 
        framework::tensor framework::utils)
endif()

# Create convenience target that links everything
add_library(framework::all INTERFACE)
target_link_libraries(framework::all INTERFACE
    framework::pipeline
    framework::tensor  
    framework::utils
    framework::tensorrt
    framework::task
    framework::memory
    framework::log
    framework::net
    NamedType
    quill::quill
    CUDA::cudart
    CUDA::cuda_driver
)

# Export variables for compatibility
set(AERIAL_FRAMEWORK_TARGETS 
    framework::pipeline framework::tensor framework::utils 
    framework::tensorrt framework::task framework::memory 
    framework::log framework::net)