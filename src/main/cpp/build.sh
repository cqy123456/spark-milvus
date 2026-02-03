#!/bin/bash
# Build script for vectorops_avx512 native library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/default-java}"

echo "Building vectorops_avx512 native library..."
echo "JAVA_HOME: ${JAVA_HOME}"

# Find JNI headers
if [ -z "$JNI_INCLUDE_DIRS" ]; then
    JNI_INCLUDE_DIRS="${JAVA_HOME}/include"
    if [ -d "${JAVA_HOME}/include/linux" ]; then
        JNI_INCLUDE_DIRS="${JAVA_HOME}/include:${JAVA_HOME}/include/linux"
    fi
fi

echo "JNI_INCLUDE_DIRS: ${JNI_INCLUDE_DIRS}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Detect CPU features
CPU_FEATURES=""
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    echo "CPU supports AVX-512, building with AVX-512 optimizations"
    # Need AVX-512DQ for _mm512_extractf32x8_ps, FMA for fused multiply-add
    CPU_FEATURES="-mavx512f -mavx512cd -mavx512dq -mavx512vl -mavx512bw -mfma"
elif grep -q avx2 /proc/cpuinfo 2>/dev/null; then
    echo "CPU supports AVX2, building with AVX2 optimizations"
    CPU_FEATURES="-mavx2 -mfma"
else
    echo "CPU does not support AVX-512 or AVX2, building scalar version"
    CPU_FEATURES=""
fi

# Compile
g++ -shared -fPIC -O3 ${CPU_FEATURES} \
    -I"${JAVA_HOME}/include" \
    -I"${JAVA_HOME}/include/linux" \
    -o libvectorops_avx512.so \
    "${SCRIPT_DIR}/vector_ops_avx512.cpp" \
    -std=c++17

echo "Build complete: ${BUILD_DIR}/libvectorops_avx512.so"
echo "To install, copy to your library path or set -Djava.library.path"
