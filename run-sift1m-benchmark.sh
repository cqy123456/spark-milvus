#!/bin/bash

# Script to run SIFT1M Benchmark as standalone Spark application
SPARK_MASTER="${SPARK_MASTER:-spark://cqy:7077}"

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
if [ ! -f "target/scala-2.13/spark-connector-assembly-0.2.1-SNAPSHOT.jar" ]; then
    echo ""
    echo "Building assembly JAR..."
    sbt assembly
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build assembly JAR"
        exit 1
    fi
fi

echo ""
echo "Running benchmark (this may take several minutes)..."
echo "----------------------------------------------------------------------"

# Set environment variables
export ASAN_OPTIONS=verify_asan_link_order=0

# Create Spark temp directories if they don't exist
SPARK_LOCAL_DIRS="/home/cqy/workspace/spark-temp"
mkdir -p "$SPARK_LOCAL_DIRS"
echo "Using Spark local directory: $SPARK_LOCAL_DIRS"

# Run with spark-submit
spark-submit \
  --class com.zilliz.spark.connector.benchmarks.Sift1MBenchmark \
  --master "$SPARK_MASTER" \
  --driver-memory 12g \
  --executor-memory 12g \
  --executor-cores 4 \
  --num-executors 4 \
  --conf spark.sql.shuffle.partitions=100 \
  --conf spark.default.parallelism=100 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  --conf spark.kryoserializer.buffer.max=1024m \
  --conf spark.driver.maxResultSize=4g \
  --conf spark.sql.adaptive.enabled=false \
  --conf spark.local.dir="$SPARK_LOCAL_DIRS" \
  --conf spark.driver.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  --conf spark.executor.extraJavaOptions="-Djava.io.tmpdir=$SPARK_LOCAL_DIRS" \
  target/scala-2.13/spark-connector-assembly-0.2.1-SNAPSHOT.jar \
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
