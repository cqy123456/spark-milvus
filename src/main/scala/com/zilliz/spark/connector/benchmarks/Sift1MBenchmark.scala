package com.zilliz.spark.connector.benchmarks

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
 *   # Then run benchmark:
 *   spark-submit --class com.zilliz.spark.connector.benchmarks.Sift1MBenchmark \
 *     --driver-memory 12g \
 *     --executor-memory 12g \
 *     target/scala-2.13/spark-connector-assembly-*.jar \
 *     [path/to/parquet/directory]
 */
object Sift1MBenchmark {

  private val logger: Logger = LogManager.getLogger(this.getClass)

  // Configuration
  val nlist = 1024  // Number of clusters (IVF index parameter)
  val nprobe = 32   // Number of clusters to search for recall calculation
  val dim = 128     // SIFT feature dimension

  def main(args: Array[String]): Unit = {
    logger.info("========== SIFT1M Benchmark Starting ==========")
    logger.info(s"Configuration: nlist=$nlist, nprobe=$nprobe, dim=$dim")

    // Get Parquet directory path from arguments or use default
    val parquetDir = if (args.nonEmpty) args(0) else "./data/parquet"
    logger.info(s"Parquet directory: $parquetDir")

    // Check if Parquet files exist
    val trainPath = s"$parquetDir/sift1m-train.parquet"
    val testPath = s"$parquetDir/sift1m-test.parquet"
    val gtPath = s"$parquetDir/sift1m-groundtruth.parquet"

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
    println("SIFT1M Benchmark: K-Means vs Mini-Batch K-Means")
    println("=" * 80)

    // Load test data and ground truth (small datasets, can fit in driver)
    logger.info("Loading test data and ground truth...")
    val (testVectors, groundTruth) = loadTestDataFromParquet(spark, parquetDir)
    logger.info(s"Test data loaded: ${testVectors.length} vectors, ${groundTruth.length} ground truth entries")

    // Load training data from Parquet (distributed by default)
    logger.info("Loading training data from Parquet...")
    val trainPath = s"$parquetDir/sift1m-train.parquet"
    val rawTrainDF = spark.read.parquet(trainPath)

    // Check and ensure Array[Float] format for MiniBatchKMeans
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.types._
    val featuresType = rawTrainDF.schema("features").dataType
    logger.info(s"Parquet features column type: $featuresType")

    val trainDF = featuresType match {
      case ArrayType(FloatType, _) =>
        logger.info("Features already in Array[Float] format")
        rawTrainDF
      case ArrayType(DoubleType, _) =>
        logger.info("Converting features from Array[Double] to Array[Float]")
        val toFloatArray = udf((arr: Seq[Double]) => arr.map(_.toFloat))
        rawTrainDF.withColumn("features", toFloatArray(col("features")))
      case _ =>
        throw new IllegalArgumentException(s"Unexpected features type: $featuresType")
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

    logger.info(s"Creating MiniBatchKMeansOperation with k=$nlist, batchSize=0.1, numBatches=10")
    val minibatchOp = new MiniBatchKMeansOperation(
      k = nlist,
      batchSize = 0.1,
      numBatches = 10,
      featuresCol = "features",
      initMode = "random",
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
    // Test 1: Standard K-Means
    // ============================================================================
    logger.info("========== Test 1: Standard K-Means ==========")
    println("\n" + "=" * 80)
    println("Testing Standard K-Means...")
    println("=" * 80)

    logger.info(s"Creating KMeansOperation with k=$nlist, maxIter=10")
    val kmeansOp = new MLlibKMeansOperation(
      k = nlist,
      maxIter = 10,
      featuresCol = "features",
      initMode = "random",
      predictionCol = "cluster_id"
    )

     // Training time
    println("Training Mini-Batch K-Means model...")
    logger.info("Starting Mini-Batch K-Means training...")
    val kmeansTrainStart = System.currentTimeMillis()
    val kmeansModel = kmeansOp.fit(trainDF)
    val kmeansTrainTime = (System.currentTimeMillis() - kmeansTrainStart) / 1000.0
    println(f"✓ Training time: $kmeansTrainTime%.2f seconds")
    logger.info(f"Mini-Batch K-Means training completed in $kmeansTrainTime%.2f seconds")

    // Assignment time
    println("Assigning vectors to clusters...")
    logger.info("Starting Mini-Batch K-Means cluster assignment...")
    val kmeansAssignStart = System.currentTimeMillis()
    // Use kmeansOp.transform() instead of model.transform() for Array[Float] support
    val kmeansResult = kmeansModel.transform(trainDF)
    kmeansResult.cache()
    val kmeansAssignmentCount = kmeansResult.count()
    val kmeansAssignTime = (System.currentTimeMillis() - kmeansAssignStart) / 1000.0
    println(f"✓ Assignment time: $kmeansAssignTime%.2f seconds")
    logger.info(f"Mini-Batch K-Means assignment completed in $kmeansAssignTime%.2f seconds, assigned $kmeansAssignmentCount vectors")

    // Loss
    logger.info("Calculating K-Means loss...")
    val kmeansLoss = calculateLossFloat(kmeansResult, kmeansModel)
    println(f"✓ K-Means loss (inertia): $kmeansLoss%.2f")
    logger.info(f"K-Means loss (inertia): $kmeansLoss%.2f")

    // Recall - use training DataFrame
    logger.info("Collecting K-Means cluster assignments...")
    val kmeansAssignments = kmeansResult.select("cluster_id").collect().map(_.getInt(0))
    logger.info(f"Collected ${kmeansAssignments.length} assignments, starting recall calculation...")
    // Use model.clusterCenters - now both operations return unified KMeansModel
    val kmeansRecall = calculateRecallFloat(
      kmeansModel.clusterCenters,
      trainDF,
      testVectors,
      groundTruth,
      kmeansAssignments,
      nprobe
    )
    println(f"✓ Recall@1 (nprobe=$nprobe): $kmeansRecall%.4f (${kmeansRecall * 100}%.2f%%)")
    logger.info(f"Mini-Batch K-Means Recall@1 (nprobe=$nprobe): $kmeansRecall%.4f (${kmeansRecall * 100}%.2f%%)")


    // Free training vectors memory after K-Means recall
    logger.info("Triggering GC after K-Means testing...")
    System.gc()
    Thread.sleep(1000)

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
    println(f"${"Metric"}%-40s ${"K-Means"}%15s ${"Mini-Batch"}%15s ${"Speedup"}%10s")
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
    println("=" * 80)

    // Cleanup
    logger.info("Unpersisting cached DataFrames...")
    trainDF.unpersist()
    //kmeansResult.unpersist()
    minibatchResult.unpersist()
    logger.info("runBenchmark() completed")
  }

  /**
   * Load test vectors and ground truth from Parquet files
   */
  def loadTestDataFromParquet(spark: SparkSession, parquetDir: String): (Array[Array[Float]], Array[Array[Int]]) = {
    logger.info(s"loadTestDataFromParquet() called with parquetDir=$parquetDir")
    println(s"\nLoading test data from Parquet: $parquetDir")

    // Load test vectors
    val testPath = s"$parquetDir/sift1m-test.parquet"
    logger.info(s"Reading test vectors from: $testPath")
    val testDF = spark.read.parquet(testPath)
    logger.info("Collecting test vectors...")
    val testVectors = testDF.collect().map { row =>
      // Parquet stores as double, convert to float
      val features = row.getAs[Seq[Double]](0).map(_.toFloat).toArray
      features
    }
    println(s"✓ Loaded ${testVectors.length} test vectors of dimension ${testVectors(0).length}")
    logger.info(s"Loaded ${testVectors.length} test vectors of dimension ${testVectors(0).length}")

    // Load ground truth
    val gtPath = s"$parquetDir/sift1m-groundtruth.parquet"
    logger.info(s"Reading ground truth from: $gtPath")
    val gtDF = spark.read.parquet(gtPath)
    logger.info("Collecting ground truth...")
    val groundTruth = gtDF.collect().map { row =>
      // Parquet stores as long for integers, convert to int
      val neighbors = row.getAs[Seq[Long]](0).map(_.toInt).toArray
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
   * IMPORTANT: Maintains Float precision throughout
   * - Raw data: Double (from Parquet)
   * - Convert to Float first (to match original SIFT1M float32 precision)
   * - Then convert to Double for MLlib Vector (MLlib requires Double)
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

    // Convert to Spark Vector with Float precision
    // 1. Parquet stores as Double
    // 2. Convert to Float to match SIFT1M original precision
    // 3. Convert back to Double for MLlib (which requires Double)
    // This ensures the same precision loss as in recall calculation
    import org.apache.spark.sql.functions._
    logger.info("Creating UDF to convert features to Spark Vector with Float precision...")
    val toVectorFloat = udf((features: Seq[Double]) => {
      // Convert Double -> Float -> Double to maintain Float precision
      val floatPrecision = features.map(_.toFloat).map(_.toDouble)
      Vectors.dense(floatPrecision.toArray)
    })

    logger.info("Transforming DataFrame with Vector conversion and repartitioning to 100 partitions...")
    val vectorDF = df.withColumn("features", toVectorFloat(col("features")))
      .select("features")
      .repartition(100)  // Force repartition to use all executors

    val finalPartitions = vectorDF.rdd.getNumPartitions
    println(s"✓ Training DataFrame loaded and repartitioned to $finalPartitions partitions")
    logger.info(s"Training DataFrame loaded and repartitioned to $finalPartitions partitions")

    // Verify Float precision by checking first vector
    val firstVec = vectorDF.limit(1).collect()(0).getAs[Vector](0).toArray
    logger.info(s"First training vector (first 5 dims): ${firstVec.take(5).mkString(", ")}")
    logger.info("Note: Training data now uses Float precision (Double->Float->Double conversion)")

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
      // Parquet stores as double, convert to float
      row.getAs[Seq[Double]](0).map(_.toFloat).toArray
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
   */
  def calculateLossFloat(
    predictions: DataFrame,
    model: com.zilliz.spark.connector.operations.clustering.KMeansModel,
    featuresCol: String = "features"
  ): Double = {
    logger.info("calculateLossFloat() called")
    val clusterCenters = model.clusterCenters
    logger.info(s"Model has ${clusterCenters.length} cluster centers")

    logger.info("Computing squared distances sum...")
    val distances = predictions.select(featuresCol, "cluster_id").rdd.map { row =>
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

    logger.info(f"Total loss (inertia): $distances%.2f")
    distances
  }

  /**
   * Calculate Recall@1 using IVF-like search with Float centers
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

    // Build inverted index
    logger.info("Building inverted index...")
    val invertedIndex = Array.fill(clusterCenters.length)(scala.collection.mutable.ArrayBuffer[Int]())
    for (i <- clusterAssignments.indices) {
      invertedIndex(clusterAssignments(i)) += i
    }
    val clusterSizes = invertedIndex.map(_.length)
    logger.info(s"Built inverted index for ${clusterCenters.length} clusters")
    logger.info(s"Cluster sizes - min: ${clusterSizes.min}, max: ${clusterSizes.max}, avg: ${clusterSizes.sum / clusterSizes.length}")

    // Process each query
    for (queryIdx <- 0 until numQueries) {
      val query = testVectors(queryIdx)
      val trueNN = groundTruth(queryIdx)(0) // Ground truth nearest neighbor

      // Compute distances to all cluster centers (Float precision)
      val clusterDistances = clusterCenters.indices.map { clusterId =>
        val center = clusterCenters(clusterId)
        val distance = euclideanDistanceFloat(query, center)
        (clusterId, distance)
      }.sortBy(_._2).take(nprobe)

      // Debug logging for first query
      if (queryIdx == 0) {
        logger.info(s"Query 0: Selected ${clusterDistances.length} clusters (nprobe=$nprobe)")
        val trueCluster = clusterAssignments(trueNN)
        val trueClusterRank = clusterDistances.indexWhere(_._1 == trueCluster)
        logger.info(f"Query 0: True NN=$trueNN is in cluster=$trueCluster, rank in selected clusters=$trueClusterRank (${if (trueClusterRank >= 0) "FOUND" else "MISSING"})")

        logger.info(f"Query 0 first 5 dims: ${query.take(5).mkString(", ")}")
        val center0 = clusterCenters(clusterDistances(0)._1)
        logger.info(f"Nearest cluster center first 5 dims: ${center0.take(5).mkString(", ")}")

        val trainVec0 = trainVectors(0)
        logger.info(f"Training vector 0 first 5 dims: ${trainVec0.take(5).mkString(", ")}")
        logger.info("Data type verification: All vectors use Float precision for consistency")
      }

      // Search within nprobe clusters
      var bestDistance = Double.MaxValue
      var bestIdx = -1

      for ((clusterId, _) <- clusterDistances) {
        for (vecIdx <- invertedIndex(clusterId)) {
          val trainVec = trainVectors(vecIdx)
          val distance = euclideanDistanceFloat(query, trainVec)

          if (distance < bestDistance) {
            bestDistance = distance
            bestIdx = vecIdx
          }
        }
      }

      // Check if we found the true nearest neighbor
      if (bestIdx == trueNN) {
        correctCount += 1
      }

      // Progress indicator
      if ((queryIdx + 1) % 1000 == 0 || queryIdx == 0) {
        print(s"\rProcessed ${queryIdx + 1}/$numQueries queries...")
      }
    }

    println(s"\rProcessed $numQueries/$numQueries queries - Done!")
    val recall = correctCount.toDouble / numQueries
    println(f"Recall@1 (nprobe=$nprobe): $recall%.4f (${recall * 100}%.2f%%)")
    logger.info(f"Recall calculation completed: $correctCount correct out of $numQueries queries, recall=$recall%.4f")

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
