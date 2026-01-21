package com.zilliz.spark.benchmarks

import com.zilliz.spark.connector.operations.clustering.{KMeansOperation, MiniBatchKMeansOperation,MLlibKMeansOperation}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.clustering.KMeansModel
import java.nio.file.{Files, Paths}
import org.apache.log4j.{LogManager, Logger}

/**
 * SIFT1M Benchmark Application: K-Means vs Mini-Batch K-Means
 *
 * This benchmark compares the performance of standard K-Means and Mini-Batch K-Means
 * on the SIFT1M dataset (1 million 128-dimensional vectors).
 *
 * Usage:
 *   # First convert HDF5 to Parquet format:
 *   python scripts/convert_hdf5_to_parquet.py
 *
 *   # Run benchmark (Mini-Batch K-Means only, default):
 *   spark-submit --class com.zilliz.spark.connector.benchmarks.Sift1MBenchmark \
 *     --driver-memory 12g \
 *     --executor-memory 12g \
 *     target/scala-2.13/spark-connector-assembly-*.jar \
 *     [path/to/parquet/directory]
 *
 *   # Run benchmark with MLlib K-Means comparison:
 *   BENCHMARK_RUN_MLLIB=true spark-submit --class com.zilliz.spark.connector.benchmarks.Sift1MBenchmark \
 *     --driver-memory 12g \
 *     --executor-memory 12g \
 *     target/scala-2.13/spark-connector-assembly-*.jar \
 *     [path/to/parquet/directory]
 */
object Sift1MBenchmark {

  private val logger: Logger = LogManager.getLogger(this.getClass)

  // Configuration - read from environment variables with defaults
  val nlist = sys.env.get("BENCHMARK_K").map(_.toInt).getOrElse(1024)  // Number of clusters (IVF index parameter)
  val nprobe = sys.env.get("BENCHMARK_NPROBE").map(_.toInt).getOrElse(32)   // Number of clusters to search for recall calculation
  val dim = sys.env.get("BENCHMARK_DIM").map(_.toInt).getOrElse(128)     // Feature dimension
  val initMode = sys.env.get("BENCHMARK_INIT_MODE").getOrElse("random")  // Initialization mode: "random" or "k-means||"
  val runMllib = sys.env.get("BENCHMARK_RUN_MLLIB").map(_.toBoolean).getOrElse(false)  // Whether to run MLlib K-Means (default: false, only MiniBatch)

  // Dataset configuration from environment variables
  val dataset = sys.env.get("BENCHMARK_DATASET").getOrElse("sift")
  val metric = dataset match {
    case "sift" | "gist" => "euclidean"
    case "glove" => "angular"
    case _ => "euclidean"
  }
  
  // Helper function to construct parquet file paths
  def getParquetPath(parquetDir: String, fileType: String): String = {
    fileType match {
      case "train" => s"$parquetDir/$dataset-$dim-$metric-train.parquet"
      case "test" => s"$parquetDir/$dataset-$dim-$metric-test.parquet"
      case "groundtruth" | "neighbors" => s"$parquetDir/$dataset-$dim-$metric-neighbors.parquet"
      case _ => throw new IllegalArgumentException(s"Unknown file type: $fileType")
    }
  }

  def main(args: Array[String]): Unit = {
    logger.info("========== SIFT1M Benchmark Starting ==========")
    logger.info(s"Configuration: nlist=$nlist, nprobe=$nprobe, dim=$dim, initMode=$initMode, runMllib=$runMllib")

    // Get Parquet directory path from arguments or use default
    val parquetDir = if (args.nonEmpty) args(0) else "./data/parquet"
    logger.info(s"Parquet directory: $parquetDir")

    // Check if Parquet files exist
    val trainPath = getParquetPath(parquetDir, "train")
    val testPath = getParquetPath(parquetDir, "test")
    val gtPath = getParquetPath(parquetDir, "neighbors")

    logger.info("Checking Parquet files existence...")
    logger.info(s"  Train: $trainPath -> ${Files.exists(Paths.get(trainPath))}")
    logger.info(s"  Test: $testPath -> ${Files.exists(Paths.get(testPath))}")
    logger.info(s"  GroundTruth: $gtPath -> ${Files.exists(Paths.get(gtPath))}")

    if (!Files.exists(Paths.get(trainPath)) ||
        !Files.exists(Paths.get(testPath)) ||
        !Files.exists(Paths.get(gtPath))) {
      logger.error("SIFT1M dataset (Parquet format) not found!")
      System.err.println(
        s"""
        |SIFT1M dataset (Parquet format) not found.
        |
        |Expected files:
        |  - $trainPath
        |  - $testPath
        |  - $gtPath
        |
        |Please convert the HDF5 dataset to Parquet:
        |  python scripts/convert_hdf5_to_parquet.py
        |
        |Or provide the parquet directory as argument:
        |  spark-submit ... Sift1MBenchmark /path/to/parquet/dir
        """.stripMargin
      )
      sys.exit(1)
    }

    // Create Spark session
    logger.info("Creating Spark session...")
    val spark = SparkSession.builder()
      .appName("SIFT1M-Benchmark")
      .config("spark.sql.shuffle.partitions", "100")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      .config("spark.driver.maxResultSize", "4g")
      .config("spark.sql.autoBroadcastJoinThreshold", "-1")
      .config("spark.memory.fraction", "0.8")
      .config("spark.memory.storageFraction", "0.3")
      .config("spark.rdd.compress", "true")
      .config("spark.shuffle.compress", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("INFO")
    logger.info("Spark session created successfully")

    try {
      logger.info("Starting benchmark execution...")
      runBenchmark(spark, parquetDir)
      logger.info("Benchmark completed successfully")
    } catch {
      case e: Exception =>
        logger.error("Benchmark failed with exception", e)
        throw e
    } finally {
      logger.info("Stopping Spark session...")
      spark.stop()
      logger.info("========== SIFT1M Benchmark Finished ==========")
    }
  }

  def runBenchmark(spark: SparkSession, parquetDir: String): Unit = {
    logger.info("runBenchmark() started")
    println("=" * 80)
    val title = if (runMllib) "SIFT1M Benchmark: K-Means vs Mini-Batch K-Means" else "SIFT1M Benchmark: Mini-Batch K-Means"
    println(title)
    println("=" * 80)

    // Load test data and ground truth (small datasets, can fit in driver)
    logger.info("Loading test data and ground truth...")
    val (testVectors, groundTruth) = loadTestDataFromParquet(spark, parquetDir)
    logger.info(s"Test data loaded: ${testVectors.length} vectors, ${groundTruth.length} ground truth entries")

    // Load training data from Parquet (distributed by default)
    logger.info("Loading training data from Parquet...")
    val trainPath = getParquetPath(parquetDir, "train")
    val trainDF = spark.read.parquet(trainPath).select("features")

    // Verify Array[Float] format
    import org.apache.spark.sql.types._
    val featuresType = trainDF.schema("features").dataType
    logger.info(s"Parquet features column type: $featuresType")

    featuresType match {
      case ArrayType(FloatType, _) =>
        logger.info("✓ Features are in Array[Float] format as expected")
      case _ =>
        throw new IllegalArgumentException(s"Expected Array[Float], got: $featuresType. Please regenerate Parquet files.")
    }

    trainDF.persist()
    logger.info("Counting training vectors...")
    val numVectors = trainDF.count()
    println(s"✓ Training DataFrame has $numVectors vectors (Array[Float] format)")
    logger.info(s"Training DataFrame has $numVectors vectors (Array[Float] format)")

    // Suggest GC to free up memory
    logger.info("Triggering GC...")
    System.gc()
    Thread.sleep(1000)

    println("-" * 80)
    println(s"Configuration: nlist=$nlist, nprobe=$nprobe, dim=$dim")
    println("-" * 80)


    // ============================================================================
    // Test 2: Mini-Batch K-Means
    // ============================================================================
    logger.info("========== Test 2: Mini-Batch K-Means ==========")
    println("\n" + "=" * 80)
    println("Testing Mini-Batch K-Means...")
    println("=" * 80)

    logger.info(s"Creating MiniBatchKMeansOperation with k=$nlist, batchSize=0.1, numBatches=10, initMode=$initMode")
    val minibatchOp = new MiniBatchKMeansOperation(
      k = nlist,
      batchSize = 0.1,
      numBatches = 10,
      featuresCol = "features",
      initMode = initMode,
      predictionCol = "cluster_id"
    )

    // Training time
    println("Training Mini-Batch K-Means model...")
    logger.info("Starting Mini-Batch K-Means training...")
    val minibatchTrainStart = System.currentTimeMillis()
    val minibatchModel = minibatchOp.fit(trainDF)
    val minibatchTrainTime = (System.currentTimeMillis() - minibatchTrainStart) / 1000.0
    println(f"✓ Training time: $minibatchTrainTime%.2f seconds")
    logger.info(f"Mini-Batch K-Means training completed in $minibatchTrainTime%.2f seconds")

    // Assignment time
    println("Assigning vectors to clusters...")
    logger.info("Starting Mini-Batch K-Means cluster assignment...")
    val minibatchAssignStart = System.currentTimeMillis()
    // Use model.transform() with unified API
    val minibatchResult = minibatchModel.transform(trainDF)
    minibatchResult.cache()
    val minibatchAssignmentCount = minibatchResult.count()
    val minibatchAssignTime = (System.currentTimeMillis() - minibatchAssignStart) / 1000.0
    println(f"✓ Assignment time: $minibatchAssignTime%.2f seconds")
    logger.info(f"Mini-Batch K-Means assignment completed in $minibatchAssignTime%.2f seconds, assigned $minibatchAssignmentCount vectors")

    // Loss
    logger.info("Calculating Mini-Batch K-Means loss...")
    val minibatchLoss = calculateLossFloat(minibatchResult, minibatchModel)
    println(f"✓ K-Means loss (inertia): $minibatchLoss%.2f")
    logger.info(f"Mini-Batch K-Means loss (inertia): $minibatchLoss%.2f")

    // Recall - use training DataFrame
    logger.info("Collecting Mini-Batch K-Means cluster assignments...")
    val minibatchAssignments = minibatchResult.select("cluster_id").collect().map(_.getInt(0))
    logger.info(f"Collected ${minibatchAssignments.length} assignments, starting recall calculation...")
    // Use model.clusterCenters with unified API
    val minibatchRecall = calculateRecallFloat(
      minibatchModel.clusterCenters,
      trainDF,
      testVectors,
      groundTruth,
      minibatchAssignments,
      nprobe
    )
    println(f"✓ Recall@1 (nprobe=$nprobe): $minibatchRecall%.4f (${minibatchRecall * 100}%.2f%%)")
    logger.info(f"Mini-Batch K-Means Recall@1 (nprobe=$nprobe): $minibatchRecall%.4f (${minibatchRecall * 100}%.2f%%)")

    // ============================================================================
    // Test 1: MLlib K-Means (Standard Implementation) - OPTIONAL
    // ============================================================================
    val (kmeansTrainTime, kmeansAssignTime, kmeansLoss, kmeansRecall, kmeansResult) = if (runMllib) {
      logger.info("========== Test 1: MLlib K-Means ==========")
      println("\n" + "=" * 80)
      println("Testing MLlib K-Means (Standard Implementation)...")
      println("=" * 80)

      logger.info(s"Creating MLlibKMeansOperation with k=$nlist, maxIter=10, initMode=$initMode")
      val kmeansOp = new MLlibKMeansOperation(
        k = nlist,
        maxIter = 10,
        featuresCol = "features",
        initMode = initMode,
        predictionCol = "cluster_id"
      )

      // Training time
      println("Training MLlib K-Means model...")
      logger.info("Starting MLlib K-Means training...")
      val kmeansTrainStart = System.currentTimeMillis()
      val kmeansModel = kmeansOp.fit(trainDF)
      val kmeansTrainTime = (System.currentTimeMillis() - kmeansTrainStart) / 1000.0
      println(f"✓ Training time: $kmeansTrainTime%.2f seconds")
      logger.info(f"MLlib K-Means training completed in $kmeansTrainTime%.2f seconds")

      // Assignment time
      println("Assigning vectors to clusters...")
      logger.info("Starting MLlib K-Means cluster assignment...")
      val kmeansAssignStart = System.currentTimeMillis()
      // Use model.transform() for MLlib KMeans
      val kmeansResult = kmeansModel.transform(trainDF)
      kmeansResult.cache()
      val kmeansAssignmentCount = kmeansResult.count()
      val kmeansAssignTime = (System.currentTimeMillis() - kmeansAssignStart) / 1000.0
      println(f"✓ Assignment time: $kmeansAssignTime%.2f seconds")
      logger.info(f"MLlib K-Means assignment completed in $kmeansAssignTime%.2f seconds, assigned $kmeansAssignmentCount vectors")

      // Loss
      logger.info("Calculating MLlib K-Means loss...")
      val kmeansLoss = calculateLossFloat(kmeansResult, kmeansModel)
      println(f"✓ K-Means loss (inertia): $kmeansLoss%.2f")
      logger.info(f"MLlib K-Means loss (inertia): $kmeansLoss%.2f")

      // Recall - use training DataFrame
      logger.info("Collecting MLlib K-Means cluster assignments...")
      val kmeansAssignments = kmeansResult.select("cluster_id").collect().map(_.getInt(0))
      logger.info(f"Collected ${kmeansAssignments.length} assignments, starting recall calculation...")
      // Use model.clusterCenters
      val kmeansRecall = calculateRecallFloat(
        kmeansModel.clusterCenters,
        trainDF,
        testVectors,
        groundTruth,
        kmeansAssignments,
        nprobe
      )
      println(f"✓ Recall@1 (nprobe=$nprobe): $kmeansRecall%.4f (${kmeansRecall * 100}%.2f%%)")
      logger.info(f"MLlib K-Means Recall@1 (nprobe=$nprobe): $kmeansRecall%.4f (${kmeansRecall * 100}%.2f%%)")

      // Free training vectors memory after K-Means recall
      logger.info("Triggering GC after K-Means testing...")
      System.gc()
      Thread.sleep(1000)

      (kmeansTrainTime, kmeansAssignTime, kmeansLoss, kmeansRecall, Some(kmeansResult))
    } else {
      logger.info("Skipping MLlib K-Means test (BENCHMARK_RUN_MLLIB=false)")
      println("\n" + "=" * 80)
      println("Skipping MLlib K-Means test")
      println("=" * 80)
      (0.0, 0.0, 0.0, 0.0, None)
    }

    // ============================================================================
    // Comparison Summary
    // ============================================================================
    logger.info("========== Generating Comparison Summary ==========")
    println("\n" + "=" * 80)
    println("BENCHMARK RESULTS SUMMARY")
    println("=" * 80)
    println(s"Dataset: SIFT1M ($numVectors vectors, $dim dimensions)")
    println(s"Configuration: nlist=$nlist, nprobe=$nprobe")
    println("-" * 80)

    if (runMllib) {
      // Comparison mode: show both MLlib and MiniBatch
      println(f"${"Metric"}%-40s ${"MLlib K-Means"}%15s ${"Mini-Batch"}%15s ${"Speedup"}%10s")
      println("-" * 80)

      val trainSpeedup = kmeansTrainTime / minibatchTrainTime
      println(f"${"Training Time (s)"}%-40s ${kmeansTrainTime}%15.2f ${minibatchTrainTime}%15.2f ${trainSpeedup}%9.2fx")
      logger.info(f"Training speedup: ${trainSpeedup}%.2fx")

      val assignSpeedup = kmeansAssignTime / minibatchAssignTime
      println(f"${"Assignment Time (s)"}%-40s ${kmeansAssignTime}%15.2f ${minibatchAssignTime}%15.2f ${assignSpeedup}%9.2fx")
      logger.info(f"Assignment speedup: ${assignSpeedup}%.2fx")

      val lossRatio = minibatchLoss / kmeansLoss
      println(f"${"K-Means Loss (inertia)"}%-40s ${kmeansLoss}%15.2f ${minibatchLoss}%15.2f ${lossRatio}%9.2fx")
      logger.info(f"Loss ratio (MiniBatch/KMeans): ${lossRatio}%.2fx")

      val recallRatio = minibatchRecall / kmeansRecall
      println(f"${"Recall@1 (nprobe=$nprobe)"}%-40s ${kmeansRecall * 100}%14.2f%% ${minibatchRecall * 100}%14.2f%% ${recallRatio}%9.2fx")
      logger.info(f"Recall ratio (MiniBatch/KMeans): ${recallRatio}%.2fx")

      println("-" * 80)
      println(f"Overall Training Speedup: ${trainSpeedup}%.2fx")
      val lossDiffPercent = (lossRatio - 1.0) * 100
      println(f"Loss Difference: ${if (lossDiffPercent > 0) "+" else ""}$lossDiffPercent%.2f%%")
      logger.info(f"Loss difference: ${if (lossDiffPercent > 0) "+" else ""}$lossDiffPercent%.2f%%")
      val recallDiffPercent = (recallRatio - 1.0) * 100
      println(f"Recall Difference: ${if (recallDiffPercent > 0) "+" else ""}$recallDiffPercent%.2f%%")
      logger.info(f"Recall difference: ${if (recallDiffPercent > 0) "+" else ""}$recallDiffPercent%.2f%%")
    } else {
      // Mini-Batch only mode: show only MiniBatch results
      println(f"${"Metric"}%-40s ${"Mini-Batch K-Means"}%20s")
      println("-" * 80)
      println(f"${"Training Time (s)"}%-40s ${minibatchTrainTime}%20.2f")
      println(f"${"Assignment Time (s)"}%-40s ${minibatchAssignTime}%20.2f")
      println(f"${"K-Means Loss (inertia)"}%-40s ${minibatchLoss}%20.2f")
      println(f"${"Recall@1 (nprobe=$nprobe)"}%-40s ${minibatchRecall * 100}%19.2f%%")
      logger.info(f"Mini-Batch K-Means Results - Train: $minibatchTrainTime%.2fs, Assign: $minibatchAssignTime%.2fs, Loss: $minibatchLoss%.2f, Recall: ${minibatchRecall * 100}%.2f%%")
    }
    println("=" * 80)

    // Cleanup
    logger.info("Unpersisting cached DataFrames...")
    trainDF.unpersist()
    minibatchResult.unpersist()
    kmeansResult.foreach(_.unpersist())
    logger.info("runBenchmark() completed")
  }

  /**
   * Load test vectors and ground truth from Parquet files
   */
  def loadTestDataFromParquet(spark: SparkSession, parquetDir: String): (Array[Array[Float]], Array[Array[Int]]) = {
    logger.info(s"loadTestDataFromParquet() called with parquetDir=$parquetDir")
    println(s"\nLoading test data from Parquet: $parquetDir")

    // Load test vectors
    val testPath = getParquetPath(parquetDir, "test")
    logger.info(s"Reading test vectors from: $testPath")

    val testVectors = try {
      val testDF = spark.read.parquet(testPath).select("features")
      logger.info(s"Test DataFrame schema: ${testDF.schema}")
      logger.info(s"Test DataFrame partitions: ${testDF.rdd.getNumPartitions}")
      logger.info(s"Test DataFrame count: ${testDF.count()}")

      logger.info("Collecting test vectors...")
      val vectors = testDF.collect().map { row =>
        // Parquet stores as Float directly (Array[Float])
        val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
        features
      }
      println(s"✓ Loaded ${vectors.length} test vectors of dimension ${vectors(0).length}")
      logger.info(s"Loaded ${vectors.length} test vectors of dimension ${vectors(0).length}")
      vectors
    } catch {
      case e: Exception =>
        logger.error(s"Failed to load test vectors from $testPath", e)
        throw e
    }

    // Load ground truth
    val gtPath = getParquetPath(parquetDir, "neighbors")
    logger.info(s"Reading ground truth from: $gtPath")
    val gtDF = spark.read.parquet(gtPath).select("neighbors")
    logger.info("Collecting ground truth...")
    val groundTruth = gtDF.collect().map { row =>
      // Parquet stores neighbors as Array[Int]
      val neighbors = row.getAs[scala.collection.mutable.WrappedArray[Int]](0).toArray
      neighbors
    }
    println(s"✓ Loaded ${groundTruth.length} ground truth entries with ${groundTruth(0).length} neighbors each")
    logger.info(s"Loaded ${groundTruth.length} ground truth entries with ${groundTruth(0).length} neighbors each")

    // Log first ground truth entry to understand the structure
    if (groundTruth.length > 0) {
      val firstGT = groundTruth(0).take(10)
      logger.info(s"First ground truth entry (first 10 neighbors): ${firstGT.mkString(", ")}")
      println(s"Ground truth structure: Each query has ${groundTruth(0).length} neighbors")
      println(s"First query's top-10 neighbors: ${firstGT.mkString(", ")}")
    }

    (testVectors, groundTruth)
  }

  /**
   * Load training data from Parquet as DataFrame (distributed by default)
   *
   * DEPRECATED: This function was used for MLlib K-Means which requires Vector format.
   * Now we use Array[Float] format throughout for consistency.
   *
   * IMPORTANT: Maintains Float precision throughout
   * - Raw data: Float (from Parquet - already float32)
   * - Convert to Double for MLlib Vector (MLlib requires Double)
   * - This ensures training uses the same precision as recall calculation
   */
  def loadTrainDataFromParquet(spark: SparkSession, parquetDir: String): DataFrame = {
    val trainPath = s"$parquetDir/sift1m-train.parquet"
    logger.info(s"loadTrainDataFromParquet() called with path=$trainPath")
    println(s"\nLoading training data from Parquet: $trainPath")

    logger.info(s"Reading Parquet file: $trainPath")
    val df = spark.read.parquet(trainPath)

    // Check current partitions
    val currentPartitions = df.rdd.getNumPartitions
    println(s"Current partitions after reading Parquet: $currentPartitions")
    logger.info(s"Current partitions after reading Parquet: $currentPartitions")

    // Convert Array[Float] to Spark Vector (which requires Double)
    // Parquet stores as Float (float32), convert to Double for MLlib
    import org.apache.spark.sql.functions._
    logger.info("Creating UDF to convert Array[Float] to Spark Vector...")
    val toVectorFloat = udf((features: scala.collection.mutable.WrappedArray[Float]) => {
      // Convert Float -> Double for MLlib Vector
      val doublePrecision = features.map(_.toDouble)
      Vectors.dense(doublePrecision.toArray)
    })

    logger.info("Transforming DataFrame with Vector conversion and repartitioning to 100 partitions...")
    val vectorDF = df.withColumn("features", toVectorFloat(col("features")))
      .select("features")
      .repartition(100)  // Force repartition to use all executors

    val finalPartitions = vectorDF.rdd.getNumPartitions
    println(s"✓ Training DataFrame loaded and repartitioned to $finalPartitions partitions")
    logger.info(s"Training DataFrame loaded and repartitioned to $finalPartitions partitions")

    // Verify by checking first vector
    val firstVec = vectorDF.limit(1).collect()(0).getAs[Vector](0).toArray
    logger.info(s"First training vector (first 5 dims): ${firstVec.take(5).mkString(", ")}")
    logger.info("Note: Training data loaded from Float Parquet, converted to Vector[Double] for MLlib")

    vectorDF
  }

  /**
   * Load training data for recall calculation as Array
   */
  def loadTrainArrayForRecall(spark: SparkSession, parquetDir: String): Array[Array[Float]] = {
    val trainPath = s"$parquetDir/sift1m-train.parquet"
    logger.info(s"loadTrainArrayForRecall() called with path=$trainPath")
    println("\nLoading training vectors for recall calculation...")

    logger.info(s"Reading training data from Parquet: $trainPath")
    val trainDF = spark.read.parquet(trainPath)
    logger.info("Collecting training vectors into array...")
    val trainVectors = trainDF.collect().map { row =>
      // Parquet now stores as Float directly (Array[Float])
      row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
    }
    println(s"✓ Loaded ${trainVectors.length} training vectors")
    logger.info(s"Loaded ${trainVectors.length} training vectors into memory")
    trainVectors
  }

  /**
   * Calculate K-Means loss (inertia / WCSS)
   * Works with both Vector and Array[Float] features
   */
  def calculateLoss(predictions: DataFrame, model: KMeansModel, featuresCol: String = "features"): Double = {
    logger.info("calculateLoss() called")
    val clusterCenters = model.clusterCenters
    logger.info(s"Model has ${clusterCenters.length} cluster centers")

    // Check feature type
    val featuresType = predictions.schema(featuresCol).dataType
    logger.info(s"Features type: $featuresType")

    logger.info("Computing squared distances sum...")
    val distances = featuresType match {
      case _: org.apache.spark.sql.types.ArrayType =>
        // Array[Float] format - use mutable.WrappedArray which is what Spark returns
        predictions.select(featuresCol, "cluster_id").rdd.map { row =>
          val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](0)
          val clusterId = row.getInt(1)
          val center = clusterCenters(clusterId).toArray.map(_.toFloat)

          var sum = 0.0
          for (i <- features.indices) {
            val diff = features(i) - center(i)
            sum += diff * diff
          }
          sum
        }.sum()

      case _ =>
        // Vector format
        predictions.select(featuresCol, "cluster_id").rdd.map { row =>
          val features = row.getAs[Vector](0)
          val clusterId = row.getInt(1)
          val center = clusterCenters(clusterId)

          var sum = 0.0
          for (i <- 0 until features.size) {
            val diff = features(i) - center(i)
            sum += diff * diff
          }
          sum
        }.sum()
    }

    logger.info(f"Total loss (inertia): $distances%.2f")
    distances
  }

  /**
   * Calculate K-Means loss (inertia / WCSS) for custom Float KMeansModel
   * Works with Array[Float] features
   *
   * Input DataFrame schema (from model.transform()):
   *   - features: Array[Float] - feature vectors
   *   - cluster_id: Int - assigned cluster ID
   *   - distance: Float - squared distance to cluster center (computed by transform)
   *
   * Returns: sum of all squared distances (Within-Set Sum of Squared Errors)
   */
  def calculateLossFloat(
    predictions: DataFrame,
    model: com.zilliz.spark.connector.operations.clustering.KMeansModel,
    featuresCol: String = "features"
  ): Double = {
    logger.info("calculateLossFloat() called")
    val clusterCenters = model.clusterCenters
    logger.info(s"Model has ${clusterCenters.length} cluster centers")

    // Check if distance column exists (from new transform implementation)
    val hasDistanceCol = predictions.columns.contains("distance")
    logger.info(s"Using distance column for loss calculation: $hasDistanceCol")

    val distances = if (hasDistanceCol) {
      // Use pre-computed distances from transform() - much faster!
      logger.info("Computing loss using pre-computed distance column...")
      predictions.select("distance").rdd.map(_.getFloat(0).toDouble).sum()
    } else {
      // Fallback: recompute distances (for backward compatibility)
      logger.info("Computing squared distances sum (distance column not found, recomputing)...")
      predictions.select(featuresCol, "cluster_id").rdd.map { row =>
        val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](0)
        val clusterId = row.getInt(1)
        val center = clusterCenters(clusterId)

        var sum = 0.0
        for (i <- features.indices) {
          val diff = features(i) - center(i)
          sum += diff * diff
        }
        sum
      }.sum()
    }

    logger.info(f"Total loss (inertia): $distances%.2f")
    distances
  }

  /**
   * Calculate Recall@1 using IVF-like search with Float centers
   * Optimized version: distributed computation to avoid OOM on driver
   */
  def calculateRecallFloat(
      clusterCenters: Array[Array[Float]],
      trainDF: DataFrame,
      testVectors: Array[Array[Float]],
      groundTruth: Array[Array[Int]],
      clusterAssignments: Array[Int],
      nprobe: Int
  ): Double = {
    logger.info(s"calculateRecallFloat() called with nprobe=$nprobe")
    logger.info(s"clusterCenters: ${clusterCenters.length}, testVectors: ${testVectors.length}")
    println(s"\nCalculating Recall@1 with nprobe=$nprobe (distributed mode)...")

    val spark = trainDF.sparkSession
    val numQueries = testVectors.length

    // Build inverted index: cluster_id -> set of vector indices for O(1) lookup
    logger.info("Building inverted index from cluster assignments...")
    val invertedIndex = Array.fill(clusterCenters.length)(scala.collection.mutable.HashSet[Int]())
    for (i <- clusterAssignments.indices) {
      invertedIndex(clusterAssignments(i)) += i
    }
    val clusterSizes = invertedIndex.map(_.size)
    logger.info(s"Built inverted index for ${clusterCenters.length} clusters")
    logger.info(s"Cluster sizes - min: ${clusterSizes.min}, max: ${clusterSizes.max}, avg: ${clusterSizes.sum / clusterSizes.length}")

    // Add cluster_id column to training DataFrame for filtering
    logger.info("Adding cluster assignments to training DataFrame...")
    import spark.implicits._

    // Create RDD with (row_index, cluster_id, features) from trainDF
    // Use zipWithIndex to get sequential 0-based indices that match clusterAssignments array
    val trainWithIndexRDD = trainDF.rdd.zipWithIndex().map { case (row, idx) =>
      val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
      val clusterId = clusterAssignments(idx.toInt)
      (idx.toInt, clusterId, features)
    }

    val trainWithCluster = trainWithIndexRDD
      .toDF("vector_id", "cluster_id", "features")
      .cache()  // Cache since we'll access it multiple times

    logger.info(s"Training data with clusters: ${trainWithCluster.count()} vectors")

    // Broadcast data for executors
    val bcCenters = spark.sparkContext.broadcast(clusterCenters)
    val bcTestVectors = spark.sparkContext.broadcast(testVectors)
    val bcGroundTruth = spark.sparkContext.broadcast(groundTruth)

    logger.info(s"Processing $numQueries queries in distributed mode...")

    // Process queries in batches to avoid too many small tasks
    val batchSize = 100
    val numBatches = (numQueries + batchSize - 1) / batchSize
    var totalCorrect = 0

    for (batchIdx <- 0 until numBatches) {
      val startIdx = batchIdx * batchSize
      val endIdx = Math.min(startIdx + batchSize, numQueries)
      val queryIndices = (startIdx until endIdx).toArray

      logger.info(s"Processing query batch ${batchIdx + 1}/$numBatches (queries $startIdx to ${endIdx - 1})...")

      // Broadcast query batch
      val bcQueryBatch = spark.sparkContext.broadcast(queryIndices)

      // Step 1: Find top nprobe clusters for each query (cheap operation)
      val queryTopClusters = queryIndices.map { queryIdx =>
        val query = testVectors(queryIdx)
        val topClusters = clusterCenters.indices.map { clusterId =>
          val center = clusterCenters(clusterId)
          var sumSq = 0.0f
          var i = 0
          while (i < query.length) {
            val diff = query(i) - center(i)
            sumSq += diff * diff
            i += 1
          }
          (clusterId, sumSq)
        }.sortBy(_._2).take(nprobe).map(_._1).toSet
        (queryIdx, topClusters)
      }.toMap

      val bcQueryTopClusters = spark.sparkContext.broadcast(queryTopClusters)

      // Step 2: Filter training data to only relevant clusters and search
      // OPTIMIZED: Only process vectors in selected clusters
      val allSelectedClusters = queryTopClusters.values.flatten.toSet
      val bcSelectedClusters = spark.sparkContext.broadcast(allSelectedClusters)

      val searchResults = trainWithCluster
        .filter($"cluster_id".isin(allSelectedClusters.toSeq: _*))  // Pre-filter by cluster
        .rdd
        .mapPartitions { iter =>
          val localTestVectors = bcTestVectors.value
          val localQueryIndices = bcQueryBatch.value
          val localQueryTopClusters = bcQueryTopClusters.value

          // Collect partition data: (vector_id, cluster_id, features)
          val partitionData = iter.map { row =>
            val vectorId = row.getInt(row.fieldIndex("vector_id"))
            val clusterId = row.getInt(row.fieldIndex("cluster_id"))
            val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](row.fieldIndex("features")).toArray
            (vectorId, clusterId, features)
          }.toArray

          // For each query, search only in its top clusters
          localQueryIndices.iterator.flatMap { queryIdx =>
            val query = localTestVectors(queryIdx)
            val topClusters = localQueryTopClusters(queryIdx)

            // Filter vectors in this partition that belong to top clusters
            val relevantVectors = partitionData.filter { case (_, clusterId, _) =>
              topClusters.contains(clusterId)
            }

            if (relevantVectors.nonEmpty) {
              // Find best match in this partition
              var bestDistance = Float.MaxValue
              var bestVectorId = -1

              relevantVectors.foreach { case (vectorId, _, trainVec) =>
                var sumSq = 0.0f
                var i = 0
                while (i < query.length) {
                  val diff = query(i) - trainVec(i)
                  sumSq += diff * diff
                  i += 1
                }
                val dist = Math.sqrt(sumSq).toFloat

                if (dist < bestDistance) {
                  bestDistance = dist
                  bestVectorId = vectorId  // Store the actual vector ID!
                }
              }

              Some((queryIdx, bestVectorId, bestDistance))
            } else {
              None
            }
          }
        }.collect()

      bcQueryBatch.destroy()
      bcQueryTopClusters.destroy()
      bcSelectedClusters.destroy()

      // Aggregate results: find global best for each query
      val queryResults = searchResults.groupBy(_._1).map { case (queryIdx, candidates) =>
        val (_, bestVecId, _) = candidates.minBy(_._3)
        (queryIdx, bestVecId)
      }

      // Count correct predictions
      val batchCorrect = queryResults.count { case (queryIdx, predictedIdx) =>
        predictedIdx == groundTruth(queryIdx)(0)
      }

      totalCorrect += batchCorrect

      logger.info(s"Batch ${batchIdx + 1}/$numBatches: ${batchCorrect}/${endIdx - startIdx} correct")
      println(s"Processed queries $startIdx-${endIdx - 1}: ${batchCorrect}/${endIdx - startIdx} correct")
    }

    // Cleanup broadcasts
    bcCenters.destroy()
    bcTestVectors.destroy()
    bcGroundTruth.destroy()
    trainWithCluster.unpersist()

    println(s"\rProcessed $numQueries queries - Done!")
    val recall = totalCorrect.toDouble / numQueries
    println(f"Recall@1 (nprobe=$nprobe): $recall%.4f (${recall * 100}%.2f%%)")
    logger.info(f"Recall calculation completed: $totalCorrect correct out of $numQueries queries, recall=$recall%.4f")

    recall
  }

  /**
   * Calculate Recall@1 using IVF-like search (deprecated - use calculateRecallFloat)
   */
  def calculateRecall(
      clusterCenters: Array[Vector],
      trainDF: DataFrame,
      testVectors: Array[Array[Float]],
      groundTruth: Array[Array[Int]],
      clusterAssignments: Array[Int],
      nprobe: Int
  ): Double = {
    // Convert Vector centers to Float arrays and call calculateRecallFloat
    val floatCenters = clusterCenters.map(_.toArray.map(_.toFloat))
    calculateRecallFloat(floatCenters, trainDF, testVectors, groundTruth, clusterAssignments, nprobe)
  }

  /**
   * Old calculateRecall implementation - kept for reference
   */
  def calculateRecallOld(
      clusterCenters: Array[Vector],
      trainDF: DataFrame,
      testVectors: Array[Array[Float]],
      groundTruth: Array[Array[Int]],
      clusterAssignments: Array[Int],
      nprobe: Int
  ): Double = {
    logger.info(s"calculateRecall() called with nprobe=$nprobe")

    // Extract training vectors as Float arrays from DataFrame
    logger.info("Extracting training vectors from DataFrame as Float arrays...")
    val trainVectors = trainDF.collect().map { row =>
      // trainDF contains Array[Float], Spark returns it as mutable.WrappedArray
      row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
    }

    logger.info(s"clusterCenters: ${clusterCenters.length}, trainVectors: ${trainVectors.length}, testVectors: ${testVectors.length}")
    println(s"\nCalculating Recall@1 with nprobe=$nprobe...")

    var correctCount = 0
    val numQueries = testVectors.length
    //var numQueries = 1 // For debugging, limit to 1 query

    // Build inverted index
    logger.info("Building inverted index...")
    val invertedIndex = Array.fill(clusterCenters.length)(scala.collection.mutable.ArrayBuffer[Int]())
    for (i <- clusterAssignments.indices) {
      invertedIndex(clusterAssignments(i)) += i
    }
    val clusterSizes = invertedIndex.map(_.length)
    logger.info(s"Inverted index built: ${clusterCenters.length} clusters, min=${clusterSizes.min}, max=${clusterSizes.max}, avg=${clusterSizes.sum / clusterSizes.length}")

    logger.info(s"Starting recall calculation for $numQueries queries...")

    // Print header for first 10 queries
    if (numQueries > 0) {
      logger.info("=" * 80)
      logger.info("First 10 Query Results Comparison (GT = Ground Truth, Result = Search Result)")
      logger.info("=" * 80)
      println("\n" + "=" * 80)
      println("First 10 Query Results:")
      println(f"${"Query"}%8s  ${"Ground Truth"}%15s  ${"Search Result"}%15s  ${"Match"}%8s  ${"Distance"}%12s")
      println("-" * 80)
    }

    for (queryIdx <- 0 until numQueries) {
      val query = testVectors(queryIdx)

      // Find nprobe nearest cluster centers
      // Convert cluster centers to Float arrays for consistent distance calculation
      val clusterDistances = clusterCenters.indices.map { clusterId =>
        val center = clusterCenters(clusterId).toArray.map(_.toFloat)  // Convert Double to Float
        val dist = euclideanDistanceFloat(query, center)  // Use Float version
        (clusterId, dist)
      }.sortBy(_._2).take(nprobe)

      // Debug: Log first query's cluster selection
      if (queryIdx == 0) {
        val topClusters = clusterDistances.take(5).map { case (cid, d) => f"$cid(d=$d%.2f)" }.mkString(", ")
        logger.info(f"Query 0: Selected top-5 clusters from $nprobe: $topClusters")

        // Also check which cluster the true nearest neighbor belongs to
        val trueNN = groundTruth(0)(0)
        val trueCluster = clusterAssignments(trueNN)
        val trueClusterRank = clusterDistances.indexWhere(_._1 == trueCluster)
        logger.info(f"Query 0: True NN=$trueNN is in cluster=$trueCluster, rank in selected clusters=$trueClusterRank (${if (trueClusterRank >= 0) "FOUND" else "MISSING"})")

        // Log the first few dimensions to verify data consistency
        logger.info(f"Query 0 first 5 dims: ${query.take(5).mkString(", ")}")
        val center0 = clusterCenters(clusterDistances(0)._1).toArray.map(_.toFloat)
        logger.info(f"Nearest cluster center first 5 dims: ${center0.take(5).mkString(", ")}")

        // Verify that training vector matches the Float precision
        val trainVec0 = trainVectors(0)
        logger.info(f"Training vector 0 first 5 dims: ${trainVec0.take(5).mkString(", ")}")
        logger.info("Data type verification: All vectors use Float precision for consistency")
      }

      // Search within nprobe clusters
      var bestDistance = Double.MaxValue
      var bestIdx = -1

      for ((clusterId, _) <- clusterDistances) {
        for (vecIdx <- invertedIndex(clusterId)) {
          val dist = euclideanDistanceFloat(query, trainVectors(vecIdx))
          if (dist < bestDistance) {
            bestDistance = dist
            bestIdx = vecIdx
          }
        }
      }

      val trueNearest = groundTruth(queryIdx)(0)
      val isMatch = bestIdx == trueNearest
      if (isMatch) {
        correctCount += 1
      }

      // Print first 10 results
      if (queryIdx < 10) {
        val matchStr = if (isMatch) "✓" else "✗"
        // Show more ground truth neighbors for debugging
        val gtNeighbors = groundTruth(queryIdx).take(5).mkString(", ")
        println(f"$queryIdx%8d  $trueNearest%15d  $bestIdx%15d  $matchStr%8s  $bestDistance%12.4f")
        logger.info(f"Query $queryIdx%4d: GT[0]=$trueNearest%8d (GT top-5: [$gtNeighbors]), Result=$bestIdx%8d, Match=$isMatch, Distance=$bestDistance%.4f")

        // Additional check: is our result in the top-K ground truth?
        val gtLength = groundTruth(queryIdx).length
        val isInTopK = groundTruth(queryIdx).contains(bestIdx)
        if (!isMatch && isInTopK) {
          val gtRank = groundTruth(queryIdx).indexOf(bestIdx)
          logger.info(f"  Note: Result $bestIdx is in GT at rank $gtRank (GT has $gtLength neighbors)")
        }
      }

      if ((queryIdx + 1) % 1000 == 0) {
        print(s"\rProcessed ${queryIdx + 1}/$numQueries queries...")
        logger.info(s"Progress: ${queryIdx + 1}/$numQueries queries processed, current recall: ${correctCount.toDouble / (queryIdx + 1)}")
      }
    }

    if (numQueries > 0) {
      println("=" * 80)
    }

    println(s"\rProcessed $numQueries/$numQueries queries - Done!")
    val recall = correctCount.toDouble / numQueries
    println(f"Recall@1 (nprobe=$nprobe): $recall%.4f (${recall * 100}%.2f%%)")
    logger.info(f"Recall calculation completed: $correctCount correct out of $numQueries queries, recall=$recall%.4f")

    recall
  }

  /**
   * Compute cluster assignments using Float precision
   * This ensures consistency with recall calculation
   */
  def computeClusterAssignmentsFloat(
      trainDF: DataFrame,
      clusterCenters: Array[Vector]
  ): Array[Int] = {
    logger.info("Computing cluster assignments with Float precision...")

    val spark = trainDF.sparkSession
    val broadcastCenters = spark.sparkContext.broadcast(clusterCenters)

    val assignments = trainDF.rdd.map { row =>
      val features = row.getAs[Vector](0)
      val featuresFloat = features.toArray.map(_.toFloat)

      // Find closest cluster using Float precision
      var closestCluster = 0
      var minDistance = Double.MaxValue

      for (i <- broadcastCenters.value.indices) {
        val centerFloat = broadcastCenters.value(i).toArray.map(_.toFloat)
        var sum = 0.0
        for (j <- featuresFloat.indices) {
          val diff = featuresFloat(j) - centerFloat(j)
          sum += diff * diff
        }

        if (sum < minDistance) {
          minDistance = sum
          closestCluster = i
        }
      }

      closestCluster
    }.collect()

    broadcastCenters.destroy()
    logger.info(s"Computed ${assignments.length} cluster assignments with Float precision")

    assignments
  }

  def euclideanDistanceFloat(v1: Array[Float], v2: Array[Float]): Double = {
    var sum = 0.0
    for (i <- v1.indices) {
      val diff = v1(i) - v2(i)
      sum += diff * diff
    }
    math.sqrt(sum)
  }

  def euclideanDistance(v1: Array[Float], v2: Array[Double]): Double = {
    var sum = 0.0
    for (i <- v1.indices) {
      val diff = v1(i) - v2(i)
      sum += diff * diff
    }
    math.sqrt(sum)
  }
}
