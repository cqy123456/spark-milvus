# Spark Milvus Connector - Benchmarks

This directory contains benchmark applications for testing and evaluating the Spark Milvus Connector performance.

## Directory Structure

```
benchmarks/
└── src/
    └── main/
        └── scala/
            └── com/
                └── zilliz/
                    └── spark/
                        └── benchmarks/
                            ├── Sift1MBenchmark.scala          # SIFT1M KMeans benchmark
                            └── VectorDedupBenchmark.scala    # Vector deduplication benchmark
```

## Available Benchmarks

### 1. SIFT1M Benchmark

Compares the performance of standard K-Means and Mini-Batch K-Means on the SIFT1M dataset (1 million 128-dimensional vectors).

**Package:** `com.zilliz.spark.benchmarks.Sift1MBenchmark`

**Run:**
```bash
./run-sift1m-benchmark.sh
```

**Features:**
- Tests both standard KMeans and Mini-Batch KMeans
- Evaluates clustering quality and performance
- Uses SIFT1M dataset with 1M training vectors

### 2. Vector Deduplication Benchmark

Tests the vector deduplication pipeline with various configurations and datasets.

**Package:** `com.zilliz.spark.benchmarks.VectorDedupBenchmark`

**Run:**
```bash
# Use predefined configurations
./run-vector-dedup-benchmark.sh sift1m-train balanced

# Available datasets:
#   - sift1m-train: SIFT1M training set (1M vectors)
#   - sift1m-test:  SIFT1M test set (10K vectors)

# Available configurations:
#   - conservative: k=50, threshold=0.05  (fewer clusters, stricter)
#   - balanced:     k=100, threshold=0.1  (moderate settings)
#   - aggressive:   k=200, threshold=0.2  (more clusters, looser)

# Custom parameters
./run-vector-dedup-benchmark.sh sift1m-train balanced 150 0.15
```

**Features:**
- Predefined dataset configurations
- Multiple deduplication strategies (conservative, balanced, aggressive)
- Supports custom parameter overrides
- Comprehensive performance metrics

## Predefined Configurations

### Datasets

| Name | Path | Dimension | Description |
|------|------|-----------|-------------|
| `sift1m-train` | `./data/parquet/sift1m-train.parquet` | 128 | SIFT1M training set with 1M vectors |
| `sift1m-test` | `./data/parquet/sift1m-test.parquet` | 128 | SIFT1M test set with 10K vectors |

### Deduplication Configs

| Name | k (clusters) | Threshold | Description |
|------|--------------|-----------|-------------|
| `conservative` | 50 | 0.05 | Fewer clusters, stricter threshold |
| `balanced` | 100 | 0.1 | Moderate settings |
| `aggressive` | 200 | 0.2 | More clusters, looser threshold |

## Building

The benchmarks are built as part of the main project:

```bash
# Build main project
sbt assembly

# The benchmarks subproject is automatically included
```

## Architecture

The benchmarks are organized as a separate SBT subproject that depends on the main `spark-connector` project:

- **Main Project (`root`)**: Core connector implementation in `src/main/scala/`
- **Benchmarks Project (`benchmarks`)**: Benchmark applications in `benchmarks/src/main/scala/`

This separation provides:
- Clear separation of concerns
- Independent benchmark configuration
- Easy maintenance and extension
- Standard SBT project structure

## Adding New Benchmarks

To add a new benchmark:

1. Create a new file in `benchmarks/src/main/scala/com/zilliz/spark/benchmarks/`
2. Use package `com.zilliz.spark.benchmarks`
3. Import main project classes: `import com.zilliz.spark.connector.apps._`
4. Create a corresponding shell script in the project root
5. Update this README with usage instructions

Example:
```scala
package com.zilliz.spark.benchmarks

import com.zilliz.spark.connector.apps.MyApp

object MyBenchmark {
  def main(args: Array[String]): Unit = {
    // Benchmark implementation
  }
}
```
