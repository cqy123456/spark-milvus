cmake_minimum_required(VERSION 3.10)
project(vectorops_avx512)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find JNI
find_package(JNI REQUIRED)

# Compiler flags for AVX-512 (with fallback to AVX2)
# We'll compile multiple versions and select at runtime
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Try to detect CPU features at compile time
# For maximum compatibility, we'll create multiple builds:
# 1. AVX-512 version (if supported)
# 2. AVX2 version (fallback)
# 3. Scalar version (final fallback)

# Main library with AVX-512 support
add_library(vectorops_avx512 SHARED
    vector_ops_avx512.cpp
)

target_include_directories(vectorops_avx512 PRIVATE ${JNI_INCLUDE_DIRS})

# Set library output name
set_target_properties(vectorops_avx512 PROPERTIES
    OUTPUT_NAME "vectorops_avx512"
    PREFIX "lib"
)

# Installation
install(TARGETS vectorops_avx512
    LIBRARY DESTINATION lib
)
