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

# Look for Quill headers in the framework installation
set(FRAMEWORK_QUILL_INCLUDE "")
foreach(QUILL_SEARCH_PATH 
        "${AERIAL_FRAMEWORK_ROOT}/include"
        "${AERIAL_FRAMEWORK_ROOT}/external/quill/include" 
        "${AERIAL_FRAMEWORK_ROOT}/../external/quill/include"
        "${AERIAL_FRAMEWORK_ROOT}/../../external/quill/include")
    if(EXISTS "${QUILL_SEARCH_PATH}/quill/DeferredFormatCodec.h")
        set(FRAMEWORK_QUILL_INCLUDE "${QUILL_SEARCH_PATH}")
        message(STATUS "Found framework's Quill headers at: ${QUILL_SEARCH_PATH}")
        break()
    endif()
endforeach()

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

# Add framework's Quill headers if found
if(FRAMEWORK_QUILL_INCLUDE)
    list(APPEND AERIAL_FRAMEWORK_INCLUDE_DIRS ${FRAMEWORK_QUILL_INCLUDE})
endif()

# Find framework libraries with multiple naming conventions
function(find_framework_library VAR_NAME)
    set(LIB_NAMES ${ARGN})
    set(LIB_PATHS 
        ${AERIAL_FRAMEWORK_ROOT}/lib
        ${AERIAL_FRAMEWORK_ROOT}/lib64
        ${AERIAL_FRAMEWORK_ROOT}/out/build/*/framework/*/
        ${AERIAL_FRAMEWORK_ROOT}/framework/*/
    )
    
    unset(${VAR_NAME} CACHE)  # Clear cache to ensure fresh search
    
    foreach(LIB_NAME ${LIB_NAMES})
        find_library(${VAR_NAME}
            NAMES ${LIB_NAME}
            PATHS ${LIB_PATHS}
            PATH_SUFFIXES lib lib64
            NO_DEFAULT_PATH
        )
        if(${VAR_NAME})
            message(STATUS "Found library ${LIB_NAME}: ${${VAR_NAME}}")
            break()
        endif()
    endforeach()
    
    if(NOT ${VAR_NAME})
        message(STATUS "Library not found: ${LIB_NAMES}")
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
    set(FULL_TARGET_NAME "framework_${TARGET_NAME}")
    
    if(${LIBRARY_VAR})
        add_library(${FULL_TARGET_NAME} UNKNOWN IMPORTED)
        set_target_properties(${FULL_TARGET_NAME} PROPERTIES
            IMPORTED_LOCATION ${${LIBRARY_VAR}}
        )
        target_include_directories(${FULL_TARGET_NAME} INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
        
        # Create the :: alias
        add_library(framework::${TARGET_NAME} ALIAS ${FULL_TARGET_NAME})
        message(STATUS "Created target framework::${TARGET_NAME} -> ${${LIBRARY_VAR}}")
    else()
        # Create stub interface target if library not found
        add_library(${FULL_TARGET_NAME} INTERFACE)
        target_include_directories(${FULL_TARGET_NAME} INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
        
        # Create the :: alias  
        add_library(framework::${TARGET_NAME} ALIAS ${FULL_TARGET_NAME})
        message(STATUS "Created stub target framework::${TARGET_NAME} (library not found)")
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

# Set up dependencies between framework targets (only if they exist)
if(TARGET framework::pipeline AND TARGET framework::tensor)
    target_link_libraries(framework_pipeline INTERFACE 
        framework::tensor framework::utils)
endif()

if(TARGET framework::tensor AND TARGET framework::utils)
    target_link_libraries(framework_tensor INTERFACE 
        framework::utils)
endif()

if(TARGET framework::tensorrt AND TARGET framework::tensor)
    target_link_libraries(framework_tensorrt INTERFACE 
        framework::tensor framework::utils)
endif()

# Create convenience target that links everything
add_library(framework_all INTERFACE)

# Link available targets
set(_available_targets)
foreach(_target pipeline tensor utils tensorrt task memory log net)
    if(TARGET framework::${_target})
        list(APPEND _available_targets framework::${_target})
    endif()
endforeach()

if(_available_targets)
    target_link_libraries(framework_all INTERFACE
        ${_available_targets}
        NamedType
        quill::quill
        CUDA::cudart
        CUDA::cuda_driver
    )
endif()

# Create the :: alias for the convenience target
add_library(framework::all ALIAS framework_all)

# Export variables for compatibility
set(AERIAL_FRAMEWORK_TARGETS)
foreach(_target pipeline tensor utils tensorrt task memory log net)
    if(TARGET framework::${_target})
        list(APPEND AERIAL_FRAMEWORK_TARGETS framework::${_target})
    endif()
endforeach()

message(STATUS "Available framework targets: ${AERIAL_FRAMEWORK_TARGETS}")