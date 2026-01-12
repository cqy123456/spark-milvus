package com.zilliz.spark.benchmarks

import com.zilliz.spark.connector.apps.VectorDedupApp
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Vector Deduplication Benchmark Configuration
 *
 * This benchmark tests the vector deduplication pipeline on various datasets.
 * It contains dataset-specific configurations and parameters that were previously
 * hardcoded in VectorDedupApp.
 */
object VectorDedupBenchmark {

  private val logger: Logger = LoggerFactory.getLogger(VectorDedupBenchmark.getClass)

  /**
   * Dataset Configuration
   */
  case class DatasetConfig(
    name: String,
    path: String,
    dimension: Int,
    description: String
  )

  /**
   * Deduplication Configuration
   */
  case class DedupConfig(
    k: Int,                            // Number of clusters for KMeans
    distanceThreshold: Float,          // Threshold for considering vectors as duplicates
    maxIterKMeans: Int = 20,           // Maximum iterations for KMeans
    distanceMetric: String = "l2",     // Distance metric: "l2" or "cosine" (for threshold graph)
    kmeansAlgorithm: String = "standard",  // KMeans algorithm: "standard" or "minibatch"
    kmeansInitMode: String = "k-means||",  // Init method: "k-means||" or "random"
    miniBatchSampleRatio: Double = 0.1,    // Sample ratio for mini-batch (0.0, 1.0]
    featuresCol: String = "features",
    idCol: String = "id"
  )

  /**
   * Predefined dataset configurations
   */
  object Datasets {
    val SIFT1M_TRAIN = DatasetConfig(
      name = "SIFT1M-Train",
      path = "./data/parquet/sift1m-train.parquet",
      dimension = 128,
      description = "SIFT1M training set with 1M 128-dimensional vectors"
    )

    val SIFT1M_TEST = DatasetConfig(
      name = "SIFT1M-Test",
      path = "./data/parquet/sift1m-test.parquet",
      dimension = 128,
      description = "SIFT1M test set with 10K 128-dimensional vectors"
    )
  }

  /**
   * Predefined deduplication configurations for different scenarios
   */
  object DedupConfigs {
    // Conservative: fewer clusters, stricter threshold
    val CONSERVATIVE = DedupConfig(
      k = 50,
      distanceThreshold = 0.05f,
      maxIterKMeans = 20
    )

    // Balanced: moderate settings
    val BALANCED = DedupConfig(
      k = 100,
      distanceThreshold = 0.1f,
      maxIterKMeans = 20
    )

    // Aggressive: more clusters, looser threshold
    val AGGRESSIVE = DedupConfig(
      k = 200,
      distanceThreshold = 0.2f,
      maxIterKMeans = 20
    )
  }

  /**
   * Benchmark result
   */
  case class BenchmarkResult(
    datasetName: String,
    configName: String,
    totalVectors: Long,
    representatives: Long,
    duplicatesRemoved: Long,
    deduplicationRate: Double,
    totalTimeSeconds: Double,
    clusteringTimeSeconds: Double,
    graphBuildingTimeSeconds: Double,
    componentFindingTimeSeconds: Double
  ) {
    def print(): Unit = {
      logger.info("=" * 80)
      logger.info("BENCHMARK RESULTS")
      logger.info("=" * 80)
      logger.info(s"Dataset: $datasetName")
      logger.info(s"Configuration: $configName")
      logger.info(s"Total vectors: $totalVectors")
      logger.info(s"Representatives: $representatives")
      logger.info(s"Duplicates removed: $duplicatesRemoved")
      logger.info(f"Deduplication rate: $deduplicationRate%.2f%%")
      logger.info(f"Total time: $totalTimeSeconds%.2f seconds")
      logger.info("=" * 80)
    }
  }

  /**
   * Run benchmark with specified dataset and deduplication config
   */
  def runBenchmark(
    spark: SparkSession,
    datasetConfig: DatasetConfig,
    dedupConfig: DedupConfig,
    configName: String = "Custom"
  ): BenchmarkResult = {
    logger.info("=" * 80)
    logger.info("Starting Vector Deduplication Benchmark")
    logger.info("=" * 80)
    logger.info(s"Dataset: ${datasetConfig.name} (${datasetConfig.path})")
    logger.info(s"Configuration: $configName")
    logger.info(s"  k (clusters): ${dedupConfig.k}")
    logger.info(s"  Distance threshold: ${dedupConfig.distanceThreshold}")
    logger.info(s"  Distance metric: ${dedupConfig.distanceMetric}")
    logger.info(s"  KMeans algorithm: ${dedupConfig.kmeansAlgorithm}")
    logger.info(s"  KMeans init mode: ${dedupConfig.kmeansInitMode}")
    if (dedupConfig.kmeansAlgorithm == "minibatch") {
      logger.info(s"  MiniBatch sample ratio: ${dedupConfig.miniBatchSampleRatio}")
    }
    logger.info(s"  Max iterations: ${dedupConfig.maxIterKMeans}")
    logger.info("=" * 80)

    // Load data
    logger.info(s"Loading data from: ${datasetConfig.path}")
    val inputDF = spark.read.parquet(datasetConfig.path)

    // Add ID column if not exists
    val dfWithId = if (!inputDF.columns.contains("id")) {
      import spark.implicits._
      inputDF.withColumn("id", monotonically_increasing_id())
    } else {
      inputDF
    }

    val totalVectors = dfWithId.count()
    logger.info(s"Loaded $totalVectors vectors")

    // Run deduplication
    val app = new VectorDedupApp(
      k = dedupConfig.k,
      distanceThreshold = dedupConfig.distanceThreshold,
      maxIterKMeans = dedupConfig.maxIterKMeans,
      local = true,  // Use default local Union-Find
      distanceMetric = dedupConfig.distanceMetric,
      kmeansAlgorithm = dedupConfig.kmeansAlgorithm,
      kmeansInitMode = dedupConfig.kmeansInitMode,
      miniBatchSampleRatio = dedupConfig.miniBatchSampleRatio,
      featuresCol = dedupConfig.featuresCol,
      idCol = dedupConfig.idCol
    )

    val startTime = System.currentTimeMillis()
    val resultDF = app.deduplicate(dfWithId)
    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0

    // Calculate statistics
    import spark.implicits._
    val representatives = resultDF.filter($"is_representative").count()
    val duplicatesRemoved = totalVectors - representatives
    val deduplicationRate = (duplicatesRemoved.toDouble / totalVectors * 100)

    val result = BenchmarkResult(
      datasetName = datasetConfig.name,
      configName = configName,
      totalVectors = totalVectors,
      representatives = representatives,
      duplicatesRemoved = duplicatesRemoved,
      deduplicationRate = deduplicationRate,
      totalTimeSeconds = totalTime,
      clusteringTimeSeconds = 0.0,  // Could be extracted from app metrics
      graphBuildingTimeSeconds = 0.0,
      componentFindingTimeSeconds = 0.0
    )

    result.print()

    // Show sample results
    println("\nSample results (first 20 rows):")
    resultDF.select("id", "cluster_id", "distance", "is_representative", "component_id")
      .orderBy("cluster_id", "distance")
      .show(20, truncate = false)

    result
  }

  /**
   * Main entry point for benchmarking
   */
  def main(args: Array[String]): Unit = {
    logger.info("=" * 80)
    logger.info("Vector Deduplication Benchmark Suite")
    logger.info("=" * 80)

    // Parse arguments
    val datasetName = if (args.nonEmpty) args(0) else "sift1m-train"
    val configName = if (args.length > 1) args(1) else "balanced"

    // Custom parameters (override preset configs)
    val customK = if (args.length > 2) Some(args(2).toInt) else None
    val customThreshold = if (args.length > 3) Some(args(3).toFloat) else None

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("VectorDeduplicationBenchmark")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try {
      // Select dataset
      val datasetConfig = datasetName.toLowerCase match {
        case "sift1m-train" => Datasets.SIFT1M_TRAIN
        case "sift1m-test" => Datasets.SIFT1M_TEST
        case _ =>
          logger.warn(s"Unknown dataset: $datasetName, using SIFT1M-Train")
          Datasets.SIFT1M_TRAIN
      }

      // Select or create deduplication config
      val baseDedupConfig = configName.toLowerCase match {
        case "conservative" => DedupConfigs.CONSERVATIVE
        case "balanced" => DedupConfigs.BALANCED
        case "aggressive" => DedupConfigs.AGGRESSIVE
        case _ =>
          logger.warn(s"Unknown config: $configName, using Balanced")
          DedupConfigs.BALANCED
      }

      // Apply custom overrides if provided
      val dedupConfig = baseDedupConfig.copy(
        k = customK.getOrElse(baseDedupConfig.k),
        distanceThreshold = customThreshold.getOrElse(baseDedupConfig.distanceThreshold)
      )

      val finalConfigName = if (customK.isDefined || customThreshold.isDefined) {
        s"$configName-custom"
      } else {
        configName
      }

      // Run benchmark
      runBenchmark(spark, datasetConfig, dedupConfig, finalConfigName)

    } catch {
      case e: Exception =>
        logger.error("Benchmark failed with exception", e)
        throw e
    } finally {
      spark.stop()
    }
  }
}
