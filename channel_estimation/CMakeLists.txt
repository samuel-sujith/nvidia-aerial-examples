#
# Remove project-specific settings - inherit from parent
# The parent CMakeLists.txt already sets these properly

# Channel Estimator Library
add_library(channel_estimator SHARED
    channel_estimator.cu
    channel_estimator.hpp
)

# Use common utilities and framework targets
target_link_libraries(channel_estimator
    common_utils
    CUDA::cudart
    CUDA::cublas  # If using cuBLAS for MMSE
    CUDA::cufft   # If using FFT for frequency domain operations
)

# Link Quill directly if available (needed for framework headers)
if(TARGET quill::quill)
    target_link_libraries(channel_estimator quill::quill)
endif()

target_include_directories(channel_estimator PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set CUDA properties
set_property(TARGET channel_estimator PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Channel Estimation Pipeline Library  
add_library(channel_est_pipeline SHARED
    channel_est_pipeline.cpp
    channel_est_pipeline.hpp
)

target_link_libraries(channel_est_pipeline
    channel_estimator
    common_utils
    # Framework libraries will be linked via common_utils
)

# Link Quill directly if available (needed for framework headers)
if(TARGET quill::quill)
    target_link_libraries(channel_est_pipeline quill::quill)
endif()

target_include_directories(channel_est_pipeline PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Example executable
add_executable(channel_estimation_example
    channel_estimation_example.cpp
)

target_link_libraries(channel_estimation_example
    channel_est_pipeline
    channel_estimator
    CUDA::cudart
)

# Performance test executable (using existing example file)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/channel_estimation_benchmark.cpp")
    add_executable(channel_estimation_benchmark
        channel_estimation_benchmark.cpp
    )
    
    target_link_libraries(channel_estimation_benchmark
        channel_est_pipeline
        channel_estimator
        CUDA::cudart
    )
else()
    message(STATUS "channel_estimation_benchmark.cpp not found, skipping benchmark build")
endif()

# Unit tests (if testing framework is available)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS AND TARGET GTest::gtest_main)
    add_executable(channel_estimation_tests
        tests/channel_estimator_test.cpp
        tests/channel_pipeline_test.cpp
    )
    
    target_link_libraries(channel_estimation_tests
        channel_est_pipeline
        channel_estimator
        GTest::gtest_main
        CUDA::cudart
    )
    
    # Register with CTest
    include(GoogleTest)
    gtest_discover_tests(channel_estimation_tests)
endif()

# Installation
install(TARGETS channel_estimator channel_est_pipeline
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES
    channel_estimator.hpp
    channel_est_pipeline.hpp
    DESTINATION include/aerial/examples/channel_estimation
)

# Install executables
install(TARGETS channel_estimation_example
    RUNTIME DESTINATION bin
)

# Install benchmark if it was built
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/channel_estimation_benchmark.cpp")
    install(TARGETS channel_estimation_benchmark
        RUNTIME DESTINATION bin
    )
endif()

# Documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS AND CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            @ONLY
        )
        
        add_custom_target(channel_estimation_docs ALL
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM
        )
        
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
            DESTINATION share/doc/aerial-channel-estimation
        )
    endif()
endif()

# Packaging
set(CPACK_PACKAGE_NAME "aerial-channel-estimation-example")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION})
set(CPACK_PACKAGE_VENDOR "NVIDIA Corporation")

include(CPack)