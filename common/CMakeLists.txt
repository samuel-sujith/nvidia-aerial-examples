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
endif()

# Install headers
install(FILES
    data_generators.hpp
    perf_utils.hpp
    test_utils.hpp
    DESTINATION include/common
)