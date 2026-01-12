package com.zilliz.spark.connector.operations.clustering

import com.zilliz.spark.connector.utils.VectorOps
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
      // Similar to KMeansOperation and MLlib KMeans for better performance
      val clusterStatsRDD = batch.rdd.mapPartitions { iter =>
        val localCenters = broadcastCenters.value
        val dim = localCenters(0).length
        val localStats = Array.fill(localCenters.length)(
          (new Array[Float](dim), 0)
        )

        iter.foreach { row =>
          val features = row.getAs[Seq[Float]](featuresCol).toArray
          val closestCluster = findClosestClusterFloat(features, localCenters)

          // Accumulate sum and count
          val (sum, count) = localStats(closestCluster)
          for (i <- features.indices) {
            sum(i) += features(i)
          }
          localStats(closestCluster) = (sum, count + 1)
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
   * K-Means++ initialization: simplified version for mini-batch
   * Selects k centers using distance-weighted sampling
   */
  private def initializeKMeansPlusPlus(
    df: DataFrame,
    k: Int,
    seed: Long
  ): Array[Array[Float]] = {
    val spark = df.sparkSession
    val rand = new scala.util.Random(seed)

    // Select first center randomly
    val firstCenter = df.sample(false, 0.01, seed)
      .limit(1)
      .select(featuresCol)
      .collect()
      .head
      .getAs[Seq[Float]](0)
      .toArray

    var centers = Array(firstCenter)
    logger.info(s"K-means++ initialization: selected first center")

    // Iteratively select remaining k-1 centers
    for (step <- 1 until k) {
      val bcCenters = spark.sparkContext.broadcast(centers)

      // Compute squared distances to nearest center
      val costs = df.select(featuresCol).rdd.map { row =>
        val point = row.getAs[Seq[Float]](0).toArray
        var minDist = Float.MaxValue
        for (center <- bcCenters.value) {
          val dist = squaredDistanceFloat(point, center)
          if (dist < minDist) {
            minDist = dist
          }
        }
        (point, minDist)
      }.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER)

      // Use treeAggregate to compute total cost in parallel
      val totalCost = costs.treeAggregate(0.0)(
        seqOp = (sum, pointCost) => sum + pointCost._2,
        combOp = (sum1, sum2) => sum1 + sum2,
        depth = 2
      )

      // Sample new center proportional to squared distance
      val threshold = rand.nextDouble() * totalCost
      var cumulative = 0.0
      var newCenter: Array[Float] = null

      costs.toLocalIterator.foreach { case (point, cost) =>
        if (newCenter == null) {
          cumulative += cost
          if (cumulative >= threshold) {
            newCenter = point
          }
        }
      }

      costs.unpersist()
      bcCenters.destroy()

      if (newCenter != null) {
        centers = centers :+ newCenter
        logger.info(s"K-means++ initialization: selected center ${step + 1}/$k")
      } else {
        logger.warn(s"K-means++ initialization: failed to select center ${step + 1}, using random fallback")
        val randomPoint = df.sample(false, 0.01, seed + step)
          .limit(1)
          .select(featuresCol)
          .collect()
          .head
          .getAs[Seq[Float]](0)
          .toArray
        centers = centers :+ randomPoint
      }
    }

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
