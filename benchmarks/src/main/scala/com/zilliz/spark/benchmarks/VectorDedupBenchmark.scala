package com.zilliz.spark.benchmarks

import com.zilliz.spark.connector.apps.{VectorDedupApp, DedupTimingStats, StepTiming}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Vector Deduplication Benchmark Configuration
 *
 * This benchmark tests the vector deduplication pipeline on various datasets.
 * It contains dataset-specific configurations and parameters that were previously
 * hardcoded in VectorDedupApp.
 *
 * Environment Variables:
 *   BENCHMARK_DATASET       - Dataset name: sift1m-train, laion1m (default: sift1m-train)
 *   BENCHMARK_K             - Number of clusters (default: 100)
 *   BENCHMARK_THRESHOLD     - Distance threshold (default: 0.1)
 *   BENCHMARK_METRIC        - Distance metric: l2, cosine (default: l2)
 *   BENCHMARK_ALGORITHM     - KMeans algorithm: standard, minibatch (default: standard)
 *   BENCHMARK_SAMPLE_RATIO  - MiniBatch sample ratio (default: 0.1)
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
    normalize: Boolean = false,            // Whether to L2 normalize input vectors
    featuresCol: String = "features",
    idCol: String = "id"
  )

  /**
   * Predefined dataset configurations
   */
  object Datasets {
    val SIFT1M_TRAIN = DatasetConfig(
      name = "SIFT1M-Train",
      path = "./data/parquet/sift-128-euclidean-train.parquet",
      dimension = 128,
      description = "SIFT1M training set with 1M 128-dimensional vectors"
    )

    val SIFT1M_TEST = DatasetConfig(
      name = "SIFT1M-Test",
      path = "./data/parquet/sift-128-euclidean-test.parquet",
      dimension = 128,
      description = "SIFT1M test set with 10K 128-dimensional vectors"
    )

    val LAION1M = DatasetConfig(
      name = "LAION1M",
      path = "./data/parquet/laion1m-train.parquet",
      dimension = 768,
      description = "LAION1M CLIP embeddings with 1M 768-dimensional vectors (normalized)"
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

    // LAION1M with MiniBatch KMeans and cosine distance
    val LAION1M_MINIBATCH = DedupConfig(
      k = 256,
      distanceThreshold = 0.1f,
      maxIterKMeans = 20,
      distanceMetric = "cosine",
      kmeansAlgorithm = "minibatch",
      kmeansInitMode = "random",
      miniBatchSampleRatio = 0.1
    )
  }

  /**
   * Operator timing entry
   */
  case class OperatorTiming(
    name: String,
    durationSeconds: Double,
    description: String = ""
  )

  /**
   * Benchmark result with detailed operator timings
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
    componentFindingTimeSeconds: Double,
    operatorTimings: Seq[OperatorTiming] = Seq.empty,
    totalEdges: Long = 0L
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

      println("\n" + "=" * 80)
      println("VECTOR DEDUPLICATION BENCHMARK RESULTS")
      println("=" * 80)

      println(s"\nDataset: $datasetName")
      println(s"Configuration: $configName")
      println(s"Total vectors: $totalVectors")
      println(s"Representatives: $representatives")
      println(s"Duplicates removed: $duplicatesRemoved")
      println(f"Deduplication rate: $deduplicationRate%.2f%%")
      if (totalEdges > 0) {
        println(s"Total graph edges: $totalEdges")
      }

      if (operatorTimings.nonEmpty) {
        println("\n" + "-" * 80)
        println("OPERATOR TIMING BREAKDOWN")
        println("-" * 80)
        println(f"${"Operator"}%-45s ${"Duration (s)"}%15s ${"Percentage"}%12s")
        println("-" * 80)

        operatorTimings.foreach { timing =>
          val percentage = if (totalTimeSeconds > 0) (timing.durationSeconds / totalTimeSeconds) * 100 else 0.0
          println(f"${timing.name}%-45s ${timing.durationSeconds}%15.2f ${percentage}%11.1f%%")
          logger.info(f"${timing.name}: ${timing.durationSeconds}%.2f seconds (${percentage}%.1f%%)")
        }

        println("-" * 80)
        println(f"${"TOTAL"}%-45s ${totalTimeSeconds}%15.2f ${"100.0"}%11s%%")
      } else {
        println(f"\nTotal time: $totalTimeSeconds%.2f seconds")
      }

      println("=" * 80)
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

    // Parse arguments (can be overridden by environment variables)
    val datasetName = sys.env.getOrElse("BENCHMARK_DATASET",
      if (args.nonEmpty) args(0) else "sift1m-train")
    val configName = sys.env.getOrElse("BENCHMARK_CONFIG",
      if (args.length > 1) args(1) else "balanced")

    // Custom parameters from environment or arguments
    val customK = sys.env.get("BENCHMARK_K").map(_.toInt)
      .orElse(if (args.length > 2) Some(args(2).toInt) else None)
    val customThreshold = sys.env.get("BENCHMARK_THRESHOLD").map(_.toFloat)
      .orElse(if (args.length > 3) Some(args(3).toFloat) else None)
    val customMetric = sys.env.get("BENCHMARK_METRIC")
    val customAlgorithm = sys.env.get("BENCHMARK_ALGORITHM")
    val customSampleRatio = sys.env.get("BENCHMARK_SAMPLE_RATIO").map(_.toDouble)
    val customNormalize = sys.env.get("BENCHMARK_NORMALIZE").map(_.toBoolean)

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("VectorDeduplicationBenchmark")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .config("spark.sql.shuffle.partitions", "100")
      .config("spark.driver.maxResultSize", "4g")
      // Disable schema merging for Parquet to avoid SparkContext access during read
      .config("spark.sql.parquet.mergeSchema", "false")
      .config("spark.sql.files.ignoreCorruptFiles", "true")
      .getOrCreate()

    // Verify SparkContext is active before proceeding
    try {
      spark.sparkContext.setLogLevel("WARN")
      // Test that SparkContext is accessible and not stopped
      if (spark.sparkContext.isStopped) {
        throw new IllegalStateException("SparkContext was stopped during initialization. Check Spark Master connection.")
      }
      val _ = spark.sparkContext.defaultParallelism
      logger.info("SparkContext is active and ready")
    } catch {
      case e: IllegalStateException if e.getMessage.contains("stopped") =>
        logger.error("SparkContext is stopped. This usually means Spark Master connection failed.", e)
        try { spark.stop() } catch { case _: Exception => }
        throw new IllegalStateException(
          "SparkContext is not available. Please check:\n" +
          "1. Spark Master is running and accessible\n" +
          "2. Master URL is correct in spark-defaults.conf or --master argument\n" +
          "3. Network connectivity to Master (check firewall/security groups)", e)
      case e: Exception =>
        logger.error("SparkContext is not accessible. Check Spark Master connection.", e)
        try { spark.stop() } catch { case _: Exception => }
        throw new IllegalStateException("SparkContext is not available. Please check Spark Master connection and configuration.", e)
    }

    try {
      // Select dataset
      val datasetConfig = datasetName.toLowerCase match {
        case "sift1m-train" => Datasets.SIFT1M_TRAIN
        case "sift1m-test" => Datasets.SIFT1M_TEST
        case "laion1m" => Datasets.LAION1M
        case _ =>
          logger.warn(s"Unknown dataset: $datasetName, using SIFT1M-Train")
          Datasets.SIFT1M_TRAIN
      }

      // Select or create deduplication config
      val baseDedupConfig = configName.toLowerCase match {
        case "conservative" => DedupConfigs.CONSERVATIVE
        case "balanced" => DedupConfigs.BALANCED
        case "aggressive" => DedupConfigs.AGGRESSIVE
        case "laion1m-minibatch" | "laion1m" => DedupConfigs.LAION1M_MINIBATCH
        case _ =>
          logger.warn(s"Unknown config: $configName, using Balanced")
          DedupConfigs.BALANCED
      }

      // Apply custom overrides if provided
      val dedupConfig = baseDedupConfig.copy(
        k = customK.getOrElse(baseDedupConfig.k),
        distanceThreshold = customThreshold.getOrElse(baseDedupConfig.distanceThreshold),
        distanceMetric = customMetric.getOrElse(baseDedupConfig.distanceMetric),
        kmeansAlgorithm = customAlgorithm.getOrElse(baseDedupConfig.kmeansAlgorithm),
        miniBatchSampleRatio = customSampleRatio.getOrElse(baseDedupConfig.miniBatchSampleRatio),
        normalize = customNormalize.getOrElse(baseDedupConfig.normalize)
      )

      val finalConfigName = {
        val hasCustom = customK.isDefined || customThreshold.isDefined ||
          customMetric.isDefined || customAlgorithm.isDefined || customNormalize.isDefined
        if (hasCustom) s"$configName-custom" else configName
      }

      // Run benchmark with timing
      runBenchmarkWithTiming(spark, datasetConfig, dedupConfig, finalConfigName)

    } catch {
      case e: Exception =>
        logger.error("Benchmark failed with exception", e)
        throw e
    } finally {
      spark.stop()
    }
  }

  /**
   * Run benchmark with detailed operator timing statistics
   */
  def runBenchmarkWithTiming(
    spark: SparkSession,
    datasetConfig: DatasetConfig,
    dedupConfig: DedupConfig,
    configName: String = "Custom"
  ): BenchmarkResult = {
    import spark.implicits._

    val timings = scala.collection.mutable.ArrayBuffer[OperatorTiming]()
    val overallStart = System.currentTimeMillis()

    logger.info("=" * 80)
    logger.info("Starting Vector Deduplication Benchmark with Timing")
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
    logger.info(s"  Normalize input: ${dedupConfig.normalize}")
    logger.info("=" * 80)

    println("\n" + "=" * 80)
    println("VECTOR DEDUPLICATION BENCHMARK")
    println("=" * 80)
    println(s"Dataset: ${datasetConfig.name}")
    println(s"  Path: ${datasetConfig.path}")
    println(s"  Dimension: ${datasetConfig.dimension}")
    println(s"Configuration: $configName")
    println(s"  K (clusters): ${dedupConfig.k}")
    println(s"  Distance threshold: ${dedupConfig.distanceThreshold}")
    println(s"  Distance metric: ${dedupConfig.distanceMetric}")
    println(s"  KMeans algorithm: ${dedupConfig.kmeansAlgorithm}")
    if (dedupConfig.kmeansAlgorithm == "minibatch") {
      println(s"  MiniBatch sample ratio: ${dedupConfig.miniBatchSampleRatio}")
    }
    println(s"  Normalize input: ${dedupConfig.normalize}")
    println("=" * 80)

    // Step 0: Load data
    logger.info("Step 0: Loading data...")

    // Verify SparkContext is still active before reading data
    if (spark.sparkContext.isStopped) {
      throw new IllegalStateException("SparkContext was stopped. Cannot read data. Check Spark Master connection.")
    }

    val loadStart = System.currentTimeMillis()
    val inputDF = spark.read.parquet(datasetConfig.path)
    val dfWithIdRaw = if (!inputDF.columns.contains("id")) {
      inputDF.withColumn("id", monotonically_increasing_id())
    } else {
      inputDF
    }

    // Apply L2 normalization if configured
    val dfWithId = if (dedupConfig.normalize) {
      logger.info("Applying L2 normalization to input vectors...")
      println("Applying L2 normalization to input vectors...")
      normalizeVectors(dfWithIdRaw, dedupConfig.featuresCol)
    } else {
      dfWithIdRaw
    }

    dfWithId.cache()
    val totalVectors = dfWithId.count()
    val loadTime = (System.currentTimeMillis() - loadStart) / 1000.0
    val loadDesc = if (dedupConfig.normalize) s"Load and normalize $totalVectors vectors" else s"Load $totalVectors vectors"
    timings += OperatorTiming("0. Data Loading", loadTime, loadDesc)
    logger.info(f"Loaded $totalVectors vectors in $loadTime%.2f seconds")
    println(s"\nLoaded $totalVectors vectors in ${loadTime}s")

    // Create VectorDedupApp with timing hooks
    val app = new VectorDedupApp(
      k = dedupConfig.k,
      distanceThreshold = dedupConfig.distanceThreshold,
      maxIterKMeans = dedupConfig.maxIterKMeans,
      local = true,
      distanceMetric = dedupConfig.distanceMetric,
      kmeansAlgorithm = dedupConfig.kmeansAlgorithm,
      kmeansInitMode = dedupConfig.kmeansInitMode,
      miniBatchSampleRatio = dedupConfig.miniBatchSampleRatio,
      featuresCol = dedupConfig.featuresCol,
      idCol = dedupConfig.idCol
    )

    // Run deduplication with detailed timing
    val dedupStart = System.currentTimeMillis()
    val (resultDF, dedupTimingStats) = app.deduplicateWithTiming(dfWithId)
    val dedupTime = (System.currentTimeMillis() - dedupStart) / 1000.0

    // Add individual step timings from VectorDedupApp
    dedupTimingStats.stepTimings.foreach { stepTiming =>
      timings += OperatorTiming(
        s"${stepTiming.stepNumber}. ${stepTiming.stepName}",
        stepTiming.durationSeconds,
        s"Step ${stepTiming.stepNumber}"
      )
    }

    // Calculate statistics
    val representatives = resultDF.filter($"is_representative").count()
    val duplicatesRemoved = totalVectors - representatives
    val deduplicationRate = (duplicatesRemoved.toDouble / totalVectors * 100)

    val totalTime = (System.currentTimeMillis() - overallStart) / 1000.0

    val result = BenchmarkResult(
      datasetName = datasetConfig.name,
      configName = configName,
      totalVectors = totalVectors,
      representatives = representatives,
      duplicatesRemoved = duplicatesRemoved,
      deduplicationRate = deduplicationRate,
      totalTimeSeconds = totalTime,
      clusteringTimeSeconds = 0.0,
      graphBuildingTimeSeconds = 0.0,
      componentFindingTimeSeconds = 0.0,
      operatorTimings = timings.toSeq
    )

    result.print()

    // Show sample results
    println("\nSample results (first 20 rows):")
    resultDF.select("id", "cluster_id", "distance", "is_representative", "component_id")
      .orderBy("cluster_id", "distance")
      .show(20, truncate = false)

    // Cleanup
    dfWithId.unpersist()

    result
  }

  /**
   * L2 normalize vectors in the DataFrame
   */
  private def normalizeVectors(df: DataFrame, featuresCol: String): DataFrame = {
    import df.sparkSession.implicits._

    val normalizeUDF = udf { (features: Seq[Float]) =>
      if (features == null || features.isEmpty) {
        features
      } else {
        val norm = math.sqrt(features.map(x => x.toDouble * x.toDouble).sum).toFloat
        if (norm > 0.0f) {
          features.map(_ / norm)
        } else {
          features
        }
      }
    }

    df.withColumn(featuresCol, normalizeUDF(col(featuresCol)))
  }
}
