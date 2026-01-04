# Common utilities - header only library
add_library(common_utils INTERFACE)

target_include_directories(common_utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Link with framework dependencies if available
if(TARGET framework::all)
    target_link_libraries(common_utils INTERFACE framework::all)
elseif(DEFINED AERIAL_FRAMEWORK_TARGETS)
    target_link_libraries(common_utils INTERFACE ${AERIAL_FRAMEWORK_TARGETS})
endif()

# Also link standard dependencies
target_link_libraries(common_utils INTERFACE
    NamedType
    CUDA::cudart
)

# Link Quill if available
if(TARGET quill::quill)
    target_link_libraries(common_utils INTERFACE quill::quill)
    # Also add Quill include directories directly for CUDA compilation
    get_target_property(QUILL_INCLUDES quill::quill INTERFACE_INCLUDE_DIRECTORIES)
    if(QUILL_INCLUDES)
        foreach(INCLUDE_DIR ${QUILL_INCLUDES})
            string(REGEX MATCH "\\$<BUILD_INTERFACE:([^>]+)>" MATCH_RESULT "${INCLUDE_DIR}")
            if(CMAKE_MATCH_1)
                target_include_directories(common_utils INTERFACE ${CMAKE_MATCH_1})
                message(STATUS "Added Quill include to common_utils: ${CMAKE_MATCH_1}")
            elseif(NOT INCLUDE_DIR MATCHES "\\$<.*>")
                target_include_directories(common_utils INTERFACE ${INCLUDE_DIR})
                message(STATUS "Added Quill include to common_utils: ${INCLUDE_DIR}")
            endif()
        endforeach()
    endif()
endif()

# Install headers
install(FILES
    data_generators.hpp
    perf_utils.hpp
    test_utils.hpp
    DESTINATION include/common
)