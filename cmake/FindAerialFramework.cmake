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

# Aerial Framework include directories (based on aerial-framework-prod structure)
set(AERIAL_FRAMEWORK_INCLUDE_DIRS
    ${AERIAL_FRAMEWORK_ROOT}/include
    ${AERIAL_FRAMEWORK_ROOT}/include/pipeline
    ${AERIAL_FRAMEWORK_ROOT}/include/tensor
    ${AERIAL_FRAMEWORK_ROOT}/include/task
    ${AERIAL_FRAMEWORK_ROOT}/include/memory
    ${AERIAL_FRAMEWORK_ROOT}/include/utils
    ${AERIAL_FRAMEWORK_ROOT}/include/log
    ${AERIAL_FRAMEWORK_ROOT}/include/net
    ${AERIAL_FRAMEWORK_ROOT}/include/wise_enum
    ${AERIAL_FRAMEWORK_ROOT}/include/tensorrt
)

# Get Quill include directories if available
if(TARGET quill::quill)
    get_target_property(QUILL_INCLUDE_DIRS quill::quill INTERFACE_INCLUDE_DIRECTORIES)
    if(QUILL_INCLUDE_DIRS)
        # Filter out generator expressions and extract real paths
        foreach(INCLUDE_DIR ${QUILL_INCLUDE_DIRS})
            string(REGEX MATCH "\\$<BUILD_INTERFACE:([^>]+)>" MATCH_RESULT "${INCLUDE_DIR}")
            if(CMAKE_MATCH_1)
                list(APPEND AERIAL_FRAMEWORK_INCLUDE_DIRS ${CMAKE_MATCH_1})
                message(STATUS "Added Quill BUILD_INTERFACE include: ${CMAKE_MATCH_1}")
            elseif(NOT INCLUDE_DIR MATCHES "\\$<.*>")
                # Add non-generator-expression paths directly
                list(APPEND AERIAL_FRAMEWORK_INCLUDE_DIRS ${INCLUDE_DIR})
                message(STATUS "Added Quill include: ${INCLUDE_DIR}")
            endif()
        endforeach()
    endif()
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

# Find all framework libraries (based on aerial-framework-prod lib structure)
find_framework_library(FRAMEWORK_PIPELINE_LIB libframework-pipeline.a)
find_framework_library(FRAMEWORK_TENSOR_LIB libframework-tensor.a)
find_framework_library(FRAMEWORK_UTILS_LIB libframework-utils.a)
find_framework_library(FRAMEWORK_TENSORRT_LIB libframework-tensorrt.a)
find_framework_library(FRAMEWORK_TASK_LIB libtask.a)
find_framework_library(FRAMEWORK_MEMORY_LIB libmemory.a)
find_framework_library(FRAMEWORK_LOG_LIB librt_log.a)
find_framework_library(FRAMEWORK_NET_LIB libnet.a)

# Create imported targets
function(create_framework_target TARGET_NAME LIBRARY_VAR)
    set(options OPTIONAL)
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    set(FULL_TARGET_NAME "framework_${TARGET_NAME}")
    
    if(${LIBRARY_VAR})
        add_library(${FULL_TARGET_NAME} UNKNOWN IMPORTED)
        set_target_properties(${FULL_TARGET_NAME} PROPERTIES
            IMPORTED_LOCATION ${${LIBRARY_VAR}}
        )
        target_include_directories(${FULL_TARGET_NAME} INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
        
        # Link Quill to all framework targets since framework headers depend on it
        if(TARGET quill::quill)
            target_link_libraries(${FULL_TARGET_NAME} INTERFACE quill::quill)
            # Also add Quill include directories directly for CUDA compilation
            get_target_property(QUILL_INCLUDES quill::quill INTERFACE_INCLUDE_DIRECTORIES)
            if(QUILL_INCLUDES)
                foreach(INCLUDE_DIR ${QUILL_INCLUDES})
                    string(REGEX MATCH "\\$<BUILD_INTERFACE:([^>]+)>" MATCH_RESULT "${INCLUDE_DIR}")
                    if(CMAKE_MATCH_1)
                        target_include_directories(${FULL_TARGET_NAME} INTERFACE ${CMAKE_MATCH_1})
                    elseif(NOT INCLUDE_DIR MATCHES "\\$<.*>")
                        target_include_directories(${FULL_TARGET_NAME} INTERFACE ${INCLUDE_DIR})
                    endif()
                endforeach()
            endif()
        endif()
        
        # Create the :: alias
        add_library(framework::${TARGET_NAME} ALIAS ${FULL_TARGET_NAME})
        message(STATUS "Created target framework::${TARGET_NAME} -> ${${LIBRARY_VAR}}")
    else()
        if(ARG_OPTIONAL)
            # Create interface target for optional libraries
            add_library(${FULL_TARGET_NAME} INTERFACE)
            target_include_directories(${FULL_TARGET_NAME} INTERFACE ${AERIAL_FRAMEWORK_INCLUDE_DIRS})
            add_library(framework::${TARGET_NAME} ALIAS ${FULL_TARGET_NAME})
            message(STATUS "Created optional interface target framework::${TARGET_NAME} (library not found)")
        else()
            message(FATAL_ERROR "Required framework library not found: ${TARGET_NAME}")
        endif()
    endif()
endfunction()

# Create all framework targets (some are optional)
create_framework_target(pipeline FRAMEWORK_PIPELINE_LIB)
create_framework_target(tensor FRAMEWORK_TENSOR_LIB)
create_framework_target(utils FRAMEWORK_UTILS_LIB) 
create_framework_target(tensorrt FRAMEWORK_TENSORRT_LIB OPTIONAL)
create_framework_target(task FRAMEWORK_TASK_LIB)
create_framework_target(memory FRAMEWORK_MEMORY_LIB OPTIONAL)
create_framework_target(log FRAMEWORK_LOG_LIB)
create_framework_target(net FRAMEWORK_NET_LIB OPTIONAL)

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
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    # Also link Quill if available
    if(TARGET quill::quill)
        target_link_libraries(framework_all INTERFACE quill::quill)
    endif()
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