#!/bin/bash

# Script to run SIFT1M Benchmark Test

echo "======================================================================"
echo "SIFT1M Benchmark: K-Means vs Mini-Batch K-Means"
echo "======================================================================"

cd "$(dirname "$0")"

# Check if HDF5 file exists
if [ ! -f "./data/sift-128-euclidean.hdf5" ]; then
    echo ""
    echo "ERROR: SIFT1M dataset (HDF5 format) not found."
    echo ""
    echo "Expected file: ./data/sift-128-euclidean.hdf5"
    echo ""
    echo "Please download the dataset:"
    echo "  wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5"
    echo ""
    exit 1
fi

echo ""
echo "Data file found:"
echo "  ✓ SIFT1M HDF5: $(ls -lh ./data/sift-128-euclidean.hdf5 | awk '{print $5}')"

echo ""
echo "Running benchmark test (this may take several minutes)..."
echo "----------------------------------------------------------------------"

# Set environment variables to avoid ASan issues
export ASAN_OPTIONS=verify_asan_link_order=0

# Option to use local-cluster mode for better memory isolation
# Set USE_LOCAL_CLUSTER=true to spawn separate executor processes
if [ "${USE_LOCAL_CLUSTER}" = "true" ]; then
    echo "Using local-cluster mode (separate executor processes)"
    export USE_LOCAL_CLUSTER=true
else
    echo "Using local mode (single process, multi-threaded)"
fi

# Run the benchmark test with increased memory
# -J options are for sbt JVM, we also need to configure test JVM
sbt -J-Xmx4g \
  'set Test / javaOptions ++= Seq("-Xmx16g", "-Xms8g", "-XX:+UseG1GC", "-XX:MaxGCPauseMillis=200")' \
  "testOnly com.zilliz.spark.connector.operations.clustering.Sift1MBenchMark"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Benchmark completed successfully!"
    echo "======================================================================"
else
    echo ""
    echo "======================================================================"
    echo "Benchmark failed!"
    echo "======================================================================"
    exit 1
fi
