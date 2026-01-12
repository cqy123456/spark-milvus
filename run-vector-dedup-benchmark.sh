#!/bin/bash

# Script to run Vector Deduplication Benchmark
# Usage:
#   ./run-vector-dedup-benchmark.sh [dataset] [config] [custom_k] [custom_threshold]
#
# Datasets:
#   sift1m-train  - SIFT1M training set (1M vectors)
#   sift1m-test   - SIFT1M test set (10K vectors)
#
# Configs:
#   conservative  - k=50, threshold=0.05 (fewer clusters, stricter threshold)
#   balanced      - k=100, threshold=0.1 (moderate settings)
#   aggressive    - k=200, threshold=0.2 (more clusters, looser threshold)
#
# Examples:
#   ./run-vector-dedup-benchmark.sh                          # Use defaults (sift1m-train, balanced)
#   ./run-vector-dedup-benchmark.sh sift1m-train conservative
#   ./run-vector-dedup-benchmark.sh sift1m-test balanced
#   ./run-vector-dedup-benchmark.sh sift1m-train balanced 150 0.15  # Custom k and threshold
#   SPARK_MASTER=local[*] ./run-vector-dedup-benchmark.sh    # Run locally

SPARK_MASTER="${SPARK_MASTER:-spark://10.15.2.60:7077}"
NUM_EXECUTORS="${NUM_EXECUTORS:-4}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-8g}"

# Arguments
DATASET="${1:-sift1m-train}"
CONFIG="${2:-balanced}"
CUSTOM_K="${3:-}"
CUSTOM_THRESHOLD="${4:-}"

echo "======================================================================"
echo "Vector Deduplication Benchmark"
echo "======================================================================"
echo ""
echo "Spark Configuration:"
echo "  Spark Master: $SPARK_MASTER"
echo "  Executors: $NUM_EXECUTORS"
echo "  Executor Cores: $EXECUTOR_CORES"
echo "  Executor Memory: $EXECUTOR_MEMORY"
echo "  Driver Memory: $DRIVER_MEMORY"
echo ""
echo "Benchmark Parameters:"
echo "  Dataset: $DATASET"
echo "  Config: $CONFIG"
if [ -n "$CUSTOM_K" ]; then
    echo "  Custom k: $CUSTOM_K"
fi
if [ -n "$CUSTOM_THRESHOLD" ]; then
    echo "  Custom threshold: $CUSTOM_THRESHOLD"
fi
echo ""

cd "$(dirname "$0")"

# Check if JAR exists
JAR_PATH="target/scala-2.12/spark-connector-assembly-0.2.1-SNAPSHOT.jar"
if [ ! -f "$JAR_PATH" ]; then
    echo "ERROR: JAR file not found: $JAR_PATH"
    echo "Please run: sbt assembly"
    exit 1
fi

# Set environment variables
export ASAN_OPTIONS=verify_asan_link_order=0

# Force Java 11 to match executor (fixes Java version mismatch)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Create Spark temp directories
SPARK_LOCAL_DIRS="/home/ubuntu/spark-workspace"
mkdir -p "$SPARK_LOCAL_DIRS"

echo "Running benchmark..."
echo "----------------------------------------------------------------------"

# Build arguments array
ARGS=("$DATASET" "$CONFIG")
if [ -n "$CUSTOM_K" ]; then
    ARGS+=("$CUSTOM_K")
    if [ -n "$CUSTOM_THRESHOLD" ]; then
        ARGS+=("$CUSTOM_THRESHOLD")
    fi
fi

# Run with spark-submit
spark-submit \
  --class com.zilliz.spark.benchmarks.VectorDedupBenchmark \
  --master "$SPARK_MASTER" \
  --driver-memory "$DRIVER_MEMORY" \
  --executor-memory "$EXECUTOR_MEMORY" \
  --executor-cores "$EXECUTOR_CORES" \
  --num-executors "$NUM_EXECUTORS" \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryoserializer.buffer.max=512m \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.network.timeout=800s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.local.dir="$SPARK_LOCAL_DIRS" \
  --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  --conf spark.executor.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  "$JAR_PATH" \
  "${ARGS[@]}"

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
