#!/bin/bash

# Script to run SIFT1M Benchmark as standalone Spark application
# Usage:
#   ./run-sift1m-benchmark.sh                        # Use default settings
#   SPARK_MASTER=spark://host:7077 ./run-sift1m-benchmark.sh  # Custom master
#   NUM_EXECUTORS=8 EXECUTOR_CORES=8 ./run-sift1m-benchmark.sh # Custom resources

SPARK_MASTER="${SPARK_MASTER:-spark://10.15.2.60:7077}"
NUM_EXECUTORS="${NUM_EXECUTORS:-4}"
EXECUTOR_CORES="${EXECUTOR_CORES:-2}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-8g}"
DRIVER_MEMORY="${DRIVER_MEMORY:-8g}"

echo "======================================================================"
echo "SIFT1M Benchmark: K-Means vs Mini-Batch K-Means"
echo "======================================================================"

cd "$(dirname "$0")"

# Check if Parquet files exist
if [ ! -d "./data/parquet" ] || \
   [ ! -f "./data/parquet/sift1m-train.parquet" ] || \
   [ ! -f "./data/parquet/sift1m-test.parquet" ] || \
   [ ! -f "./data/parquet/sift1m-groundtruth.parquet" ]; then
    echo ""
    echo "ERROR: SIFT1M dataset (Parquet format) not found."
    echo ""
    echo "Expected files:"
    echo "  - ./data/parquet/sift1m-train.parquet"
    echo "  - ./data/parquet/sift1m-test.parquet"
    echo "  - ./data/parquet/sift1m-groundtruth.parquet"
    echo ""
    echo "Please convert the HDF5 dataset to Parquet:"
    echo "  python scripts/convert_hdf5_to_parquet.py"
    echo ""
    exit 1
fi

echo ""
echo "Data files found:"
echo "  ✓ Training data: $(du -h ./data/parquet/sift1m-train.parquet | cut -f1)"
echo "  ✓ Test data: $(du -h ./data/parquet/sift1m-test.parquet | cut -f1)"
echo "  ✓ Ground truth: $(du -h ./data/parquet/sift1m-groundtruth.parquet | cut -f1)"

# Build assembly JAR if needed
JAR_PATH="target/scala-2.12/spark-connector-assembly-0.2.1-SNAPSHOT.jar"
if [ ! -f "$JAR_PATH" ]; then
    echo ""
    echo "Building assembly JAR..."
    sbt assembly
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build assembly JAR"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  Spark Master: $SPARK_MASTER"
echo "  Executors: $NUM_EXECUTORS"
echo "  Executor Cores: $EXECUTOR_CORES"
echo "  Executor Memory: $EXECUTOR_MEMORY"
echo "  Driver Memory: $DRIVER_MEMORY"

echo ""
echo "Running benchmark (this may take several minutes)..."
echo "----------------------------------------------------------------------"

# Set environment variables
export ASAN_OPTIONS=verify_asan_link_order=0

# Force Java 11 to match executor (fixes Java version mismatch causing EOFException)
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Create Spark temp directories if they don't exist
SPARK_LOCAL_DIRS="/home/ubuntu/spark-workspace"
mkdir -p "$SPARK_LOCAL_DIRS"
echo "Using Spark local directory: $SPARK_LOCAL_DIRS"

# Run with spark-submit
spark-submit \
  --class com.zilliz.spark.benchmarks.Sift1MBenchmark \
  --master "$SPARK_MASTER" \
  --driver-memory "$DRIVER_MEMORY" \
  --executor-memory "$EXECUTOR_MEMORY" \
  --executor-cores "$EXECUTOR_CORES" \
  --num-executors "$NUM_EXECUTORS" \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.default.parallelism=200 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryoserializer.buffer.max=512m \
  --conf spark.driver.maxResultSize=12g \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.rpc.message.maxSize=1024 \
  --conf spark.network.timeout=800s \
  --conf spark.executor.heartbeatInterval=60s \
  --conf spark.storage.memoryFraction=0.6 \
  --conf spark.shuffle.memoryFraction=0.3 \
  --conf spark.local.dir="$SPARK_LOCAL_DIRS" \
  --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  --conf spark.executor.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  "$JAR_PATH" \
  ./data/parquet

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
