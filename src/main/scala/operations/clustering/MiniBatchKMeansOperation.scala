package com.zilliz.spark.connector.operations.clustering

import com.zilliz.spark.connector.utils.VectorOps
import org.apache.spark.ml.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType, StructField}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable

/**
 * Mini-Batch K-Means clustering operation with incremental learning
 *
 * Algorithm:
 * 1. Initialize cluster centers using K-Means++ on a small sample (if initMode="k-means||")
 *    or random sampling (if initMode="random")
 * 2. For each mini-batch iteration:
 *    a. Sample a batch of data points
 *    b. Assign each point to nearest cluster center
 *    c. Update cluster centers using incremental formula:
 *       center_new = center_old + eta * (batch_mean - center_old)
 *       where eta = batch_count / (total_count_for_cluster)
 * 3. Final prediction uses trained centers on full dataset
 *
 * IMPORTANT: This operation works directly with Array[Float] (Milvus format).
 * Conversion to Vector only happens when using MLlib KMeans for initialization (initMode="k-means||")
 *
 * @param k Number of clusters
 * @param batchSize Fraction of data to sample per batch (0.0-1.0)
 * @param numBatches Number of mini-batch iterations
 * @param initMaxIter Max iterations for K-Means++ initialization
 * @param seed Random seed
 * @param featuresCol Features column name
 * @param predictionCol Prediction column name
 * @param initMode Initialization mode: "k-means||" (k-means++) or "random"
 */
class MiniBatchKMeansOperation(
  val k: Int,
  val batchSize: Double = 0.1,
  val numBatches: Int = 10,
  val initMaxIter: Int = 1,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val initMode: String = "k-means||"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MiniBatchKMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(batchSize > 0.0 && batchSize <= 1.0, "Batch size must be between 0.0 and 1.0")
  require(numBatches > 0, "Number of batches must be positive")
  require(initMode == "k-means||" || initMode == "random",
    s"initMode must be 'k-means||' or 'random', got: $initMode")

  /**
   * Train Mini-Batch K-Means and return the model
   * Accepts DataFrame with Array[Float] (Milvus format)
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training Mini-Batch K-Means: k=$k, batchSize=$batchSize, numBatches=$numBatches, initMode=$initMode")

    val spark = df.sparkSession
    val startTime = System.currentTimeMillis()

    // Validate input format
    val featuresType = df.schema(featuresCol).dataType
    require(featuresType == ArrayType(FloatType, false) || featuresType == ArrayType(FloatType, true),
      s"Input DataFrame must have Array[Float] column, but got $featuresType")

    // Cache data for faster sampling
    df.cache()
    val totalCount = df.count()
    logger.info(s"Total data points: $totalCount")

    // Step 1: Initialize cluster centers
    logger.info(s"Step 1: Initializing cluster centers with mode: $initMode...")
    val initStart = System.currentTimeMillis()

    var centers: Array[Array[Float]] = if (initMode == "random") {
      // Random initialization: directly sample k random points as centers
      logger.info("Using random initialization: sampling k random points as initial centers")
      val randomSample = df.sample(false, Math.min(0.1, batchSize * 2), seed)
        .limit(k)
        .select(featuresCol)
        .collect()
        .map(_.getAs[Seq[Float]](0).toArray)

      if (randomSample.length < k) {
        logger.warn(s"Only sampled ${randomSample.length} points, expected $k. Using all sampled points.")
      }
      randomSample
    } else {
      // K-Means++ initialization: native Float implementation
      logger.info("Using k-means++ initialization (native Float implementation)")

      val initSample = df.sample(false, Math.min(0.1, batchSize * 2), seed)
      initializeKMeansPlusPlus(initSample, k, seed)
    }

    val initEnd = System.currentTimeMillis()
    logger.info(s"Initialization completed in ${initEnd - initStart} ms")
    logger.info(s"Initialized ${centers.length} cluster centers")

    // Track total points assigned to each cluster (for learning rate calculation)
    val clusterCounts = Array.fill(k)(1) // Avoid division by zero

    // Statistics
    var totalIterationTime: Long = 0
    var totalSamplingTime: Long = 0
    var totalAssignmentTime: Long = 0
    var totalUpdateTime: Long = 0

    // Step 2: Mini-Batch incremental updates
    logger.info("Starting mini-batch incremental training...")

    for (iteration <- 0 until numBatches) {
      val iterStart = System.currentTimeMillis()
      logger.info(s"=== Iteration ${iteration + 1}/$numBatches ===")

      // 2.1 Sampling phase
      val samplingStart = System.currentTimeMillis()
      val batch = df.sample(false, batchSize, seed + iteration).cache()
      val batchCount = batch.count()
      val samplingEnd = System.currentTimeMillis()
      val samplingTime = samplingEnd - samplingStart

      if (batchCount > 0) {
        // Only process non-empty batches

      logger.info(s"  [Sampling] Time: ${samplingTime} ms, Batch size: $batchCount samples")
      totalSamplingTime += samplingTime

      // 2.2 Assignment and statistics phase
      val assignmentStart = System.currentTimeMillis()

      // Broadcast centers for efficient access
      val broadcastCenters = spark.sparkContext.broadcast(centers)

      // Compute cluster statistics using mapPartitions + reduceByKey
      // MEMORY OPTIMIZED: Process in sub-batches to avoid loading entire partition into memory
      val clusterStatsRDD = batch.rdd.mapPartitions { iter =>
        val localCenters = broadcastCenters.value
        val dim = localCenters(0).length
        val localStats = Array.fill(localCenters.length)(
          (new Array[Float](dim), 0)
        )

        // MEMORY OPTIMIZATION: Process partition in sub-batches to limit peak memory
        // This prevents OOM when partition contains too many points
        val subBatchSize = 5000  // Process 5000 points at a time
        val partitionIter = iter.map { row =>
          row.getAs[Seq[Float]](featuresCol).toArray
        }

        while (partitionIter.hasNext) {
          // Collect sub-batch
          val subBatch = partitionIter.take(subBatchSize).toArray

          if (subBatch.nonEmpty) {
            // OPTIMIZED: Find nearest center for each point using ND4J with SIMD
            // findNearestTargets uses adaptive batch size to prevent OOM
            val nearestCenters = VectorOps.findNearestTargets(subBatch, localCenters)

            // Assign each point to closest cluster and accumulate stats
            var i = 0
            while (i < subBatch.length) {
              val (closestCluster, minDist) = nearestCenters(i)

              // Accumulate sum and count
              val (sum, count) = localStats(closestCluster)
              val point = subBatch(i)
              var k = 0
              while (k < dim) {
                sum(k) += point(k)
                k += 1
              }
              localStats(closestCluster) = (sum, count + 1)

              i += 1
            }
          }
        }

        // Emit (clusterId, stats) pairs - only for non-empty clusters
        Iterator.tabulate(localCenters.length) { clusterId =>
          (clusterId, localStats(clusterId))
        }.filter(_._2._2 > 0)  // Filter out empty clusters to reduce shuffle
      }.reduceByKey { (v1, v2) =>
        // Merge statistics for the same cluster
        val (sum1, count1) = v1
        val (sum2, count2) = v2
        for (i <- sum1.indices) {
          sum1(i) += sum2(i)
        }
        (sum1, count1 + count2)
      }.collectAsMap()  // Use collectAsMap for direct lookup

      broadcastCenters.destroy()
      batch.unpersist()

      val assignmentEnd = System.currentTimeMillis()
      val assignmentTime = assignmentEnd - assignmentStart
      totalAssignmentTime += assignmentTime

      logger.info(s"  [Assignment] Time: ${assignmentTime} ms")

      // 2.3 Update phase
      val updateStart = System.currentTimeMillis()

      // Incremental update formula:
      // center_new = (1 - eta) * center_old + eta * batch_mean
      // where eta = batch_count / (total_count_for_cluster)

      clusterStatsRDD.foreach { case (clusterId, (sum, batchClusterCount)) =>
        // Compute batch mean for this cluster
        val batchMean = sum.map(_ / batchClusterCount)

        // Update total count for this cluster
        clusterCounts(clusterId) += batchClusterCount

        // Compute learning rate
        val eta = Math.min(batchClusterCount.toDouble / clusterCounts(clusterId), 1.0).toFloat

        // Incremental update
        val oldCenter = centers(clusterId)
        val newCenter = new Array[Float](oldCenter.length)

        for (i <- oldCenter.indices) {
          newCenter(i) = (1.0f - eta) * oldCenter(i) + eta * batchMean(i)
        }

        centers(clusterId) = newCenter
      }

      val updateEnd = System.currentTimeMillis()
      val updateTime = updateEnd - updateStart
      totalUpdateTime += updateTime


      val iterTime = System.currentTimeMillis() - iterStart
      totalIterationTime += iterTime

      logger.info(s"  [Total] Iteration time: ${iterTime} ms (sampling:${samplingTime} + assignment:${assignmentTime} + update:${updateTime})")
      } else {
        logger.warn("  Batch is empty, skipping")
        batch.unpersist()
      }
    }

    df.unpersist()

    val totalTime = System.currentTimeMillis() - startTime

    // Log final statistics
    logger.info("=" * 60)
    logger.info("Mini-Batch K-Means Training Completed")
    logger.info("=" * 60)
    logger.info(s"Initialization time:  ${initEnd - initStart} ms")
    logger.info(s"Actual iterations:    $numBatches")
    logger.info(s"Total iteration time: $totalIterationTime ms")
    logger.info(s"  - Sampling:         $totalSamplingTime ms (avg: ${totalSamplingTime / numBatches} ms)")
    logger.info(s"  - Assignment:       $totalAssignmentTime ms (avg: ${totalAssignmentTime / numBatches} ms)")
    logger.info(s"  - Update:           $totalUpdateTime ms (avg: ${totalUpdateTime / numBatches} ms)")
    logger.info(s"Total training time:  $totalTime ms")
    logger.info("=" * 60)

    // Return KMeansModel with trained centers
    new KMeansModel(centers, featuresCol, predictionCol)
  }

  /**
   * Find closest cluster center to a point (Float precision)
   * Uses ND4J batch distance computation for optimal SIMD performance
   */
  private def findClosestClusterFloat(point: Array[Float], centers: Array[Array[Float]]): Int = {
    // Use ND4J batch computation to calculate distances to all centers at once
    // This leverages SIMD instructions and is much faster than sequential computation
    val distances = VectorOps.batchEuclideanDistance(point, centers)

    // Find index of minimum distance
    var closestCluster = 0
    var minDistance = distances(0)

    for (i <- 1 until distances.length) {
      if (distances(i) < minDistance) {
        minDistance = distances(i)
        closestCluster = i
      }
    }

    closestCluster
  }

  /**
   * Compute squared Euclidean distance with Float precision
   * Uses ND4J for SIMD-optimized computation
   */
  private def squaredDistanceFloat(v1: Array[Float], v2: Array[Float]): Float = {
    val dist = VectorOps.euclideanDistance(v1, v2)
    dist * dist  // Return squared distance
  }

  /**
   * K-Means++ initialization using MLlib's k-means|| algorithm
   *
   * MLlib's k-means|| is a distributed, scalable version of k-means++ that:
   * 1. Runs in O(log n) rounds instead of O(k) sequential iterations
   * 2. Samples ~k candidates per round in parallel across the cluster
   * 3. Much more efficient for large k values
   *
   * We run MLlib KMeans with maxIter=0 to only perform initialization,
   * then extract the initialized centers for our mini-batch training.
   */
  private def initializeKMeansPlusPlus(
    df: DataFrame,
    k: Int,
    seed: Long
  ): Array[Array[Float]] = {
    logger.info(s"Using MLlib k-means|| for initialization (k=$k)")

    // Convert Array[Float] to Vector[Double] for MLlib
    val toVector = udf((arr: Seq[Float]) => {
      Vectors.dense(arr.map(_.toDouble).toArray)
    })

    val vectorDF = df.withColumn("__mllib_init_features", toVector(col(featuresCol)))
    vectorDF.cache()

    // Use MLlib KMeans with maxIter=1 to get k-means|| initialized centers
    // initSteps controls the number of k-means|| rounds (default 2, we use 5 for better quality)
    val mllibKMeans = new MLlibKMeans()
      .setK(k)
      .setMaxIter(initMaxIter)  // Use the configured initMaxIter (default 1)
      .setSeed(seed)
      .setInitMode("k-means||")
      .setInitSteps(5)  // More steps for better initialization quality
      .setFeaturesCol("__mllib_init_features")
      .setPredictionCol("__mllib_init_pred")

    val mllibModel = mllibKMeans.fit(vectorDF)

    // Convert Vector[Double] centers back to Array[Float]
    val centers = mllibModel.clusterCenters.map { vector =>
      vector.toArray.map(_.toFloat)
    }

    vectorDF.unpersist()

    logger.info(s"MLlib k-means|| initialization completed: ${centers.length} centers")
    centers
  }
}

object MiniBatchKMeansOperation {

  def apply(k: Int, batchSize: Double = 0.1, numBatches: Int = 10): MiniBatchKMeansOperation = {
    new MiniBatchKMeansOperation(k = k, batchSize = batchSize, numBatches = numBatches)
  }

  def apply(
    k: Int,
    batchSize: Double,
    numBatches: Int,
    featuresCol: String,
    predictionCol: String,
    initMode: String
  ): MiniBatchKMeansOperation = {
    new MiniBatchKMeansOperation(
      k = k,
      batchSize = batchSize,
      numBatches = numBatches,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      initMode = initMode
    )
  }
}
