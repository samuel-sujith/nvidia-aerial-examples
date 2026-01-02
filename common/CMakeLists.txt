# Common utilities - header only library
add_library(common_utils INTERFACE)

target_include_directories(common_utils INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Install headers
install(FILES
    data_generators.hpp
    perf_utils.hpp
    test_utils.hpp
    DESTINATION include/common
)