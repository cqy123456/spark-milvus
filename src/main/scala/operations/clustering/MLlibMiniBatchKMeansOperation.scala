package com.zilliz.spark.connector.operations.clustering

import org.apache.spark.ml.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable

/**
 * MLlib-based Mini-Batch K-Means clustering operation
 *
 * This operation uses MLlib KMeans for k-means++ initialization (when initMode="k-means||"),
 * then performs mini-batch incremental updates. It works with Milvus Array[Float] format:
 * - Input: Array[Float] (Milvus format)
 * - K-means++ init: Converts to Vector[Double] for MLlib, then back to Float
 * - Mini-batch updates: Native Float operations
 * - Output: KMeansModel with Array[Float] centers
 *
 * Use this when you want MLlib's proven k-means++ initialization with mini-batch training.
 * For pure Float implementation without MLlib, use MiniBatchKMeansOperation instead.
 *
 * @param k Number of clusters
 * @param batchSize Fraction of data to sample per batch (0.0-1.0)
 * @param numBatches Number of mini-batch iterations
 * @param initMaxIter Max iterations for K-Means++ initialization (only used if initMode="k-means||")
 * @param seed Random seed
 * @param featuresCol Features column name
 * @param predictionCol Prediction column name
 * @param initMode Initialization mode: "k-means||" (uses MLlib) or "random" (native Float)
 */
class MLlibMiniBatchKMeansOperation(
  val k: Int,
  val batchSize: Double = 0.1,
  val numBatches: Int = 10,
  val initMaxIter: Int = 1,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val initMode: String = "k-means||"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MLlibMiniBatchKMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(batchSize > 0.0 && batchSize <= 1.0, "Batch size must be between 0.0 and 1.0")
  require(numBatches > 0, "Number of batches must be positive")
  require(initMode == "k-means||" || initMode == "random",
    s"initMode must be 'k-means||' or 'random', got: $initMode")

  /**
   * Train Mini-Batch K-Means and return the model
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training MLlib Mini-Batch K-Means: k=$k, batchSize=$batchSize, numBatches=$numBatches, initMode=$initMode")

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
      // K-Means++ initialization: use MLlib KMeans
      logger.info("Using k-means++ initialization via MLlib KMeans (converting to Vector temporarily)")

      val toVector = udf((arr: Seq[Float]) => {
        Vectors.dense(arr.map(_.toDouble).toArray)
      })

      val vectorDF = df.withColumn("__mllib_features", toVector(col(featuresCol)))
      val initSample = vectorDF.sample(false, Math.min(0.1, batchSize * 2), seed)

      val initKMeans = new MLlibKMeans()
        .setK(k)
        .setMaxIter(initMaxIter)
        .setInitMode(initMode)
        .setSeed(seed)
        .setFeaturesCol("__mllib_features")
        .setPredictionCol(predictionCol)

      val initModel = initKMeans.fit(initSample)

      // Convert Vector centers back to Array[Float]
      initModel.clusterCenters.map(v => v.toArray.map(_.toFloat))
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

      // Compute cluster statistics using mapPartitions + reduce
      val clusterStatsRDD = batch.rdd.mapPartitions { iter =>
        val localCenters = broadcastCenters.value
        val localStats = mutable.HashMap[Int, (Array[Float], Int)]()

        iter.foreach { row =>
          val features = row.getAs[Seq[Float]](featuresCol).toArray
          val closestCluster = findClosestClusterFloat(features, localCenters)

          localStats.get(closestCluster) match {
            case Some((sum, count)) =>
              for (i <- sum.indices) {
                sum(i) += features(i)
              }
              localStats(closestCluster) = (sum, count + 1)
            case None =>
              localStats(closestCluster) = (features.clone(), 1)
          }
        }

        localStats.iterator
      }.reduceByKey((v1: (Array[Float], Int), v2: (Array[Float], Int)) => {
        val (sum1, count1) = v1
        val (sum2, count2) = v2
        for (i <- sum1.indices) {
          sum1(i) += sum2(i)
        }
        (sum1, count1 + count2)
      }).collect()

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

      logger.info(s"  [Update] Time: ${updateTime} ms, Updated ${clusterStatsRDD.length} centers")

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
    logger.info("MLlib Mini-Batch K-Means Training Completed")
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
   */
  private def findClosestClusterFloat(point: Array[Float], centers: Array[Array[Float]]): Int = {
    var closestCluster = 0
    var minDistance = squaredDistanceFloat(point, centers(0))

    for (i <- 1 until centers.length) {
      val distance = squaredDistanceFloat(point, centers(i))
      if (distance < minDistance) {
        minDistance = distance
        closestCluster = i
      }
    }

    closestCluster
  }

  /**
   * Compute squared Euclidean distance with Float precision
   */
  private def squaredDistanceFloat(v1: Array[Float], v2: Array[Float]): Float = {
    var sum = 0.0f

    for (i <- v1.indices) {
      val diff = v1(i) - v2(i)
      sum += diff * diff
    }

    sum
  }
}

object MLlibMiniBatchKMeansOperation {

  def apply(k: Int, batchSize: Double = 0.1, numBatches: Int = 10): MLlibMiniBatchKMeansOperation = {
    new MLlibMiniBatchKMeansOperation(k = k, batchSize = batchSize, numBatches = numBatches)
  }

  def apply(
    k: Int,
    batchSize: Double,
    numBatches: Int,
    featuresCol: String,
    predictionCol: String,
    initMode: String
  ): MLlibMiniBatchKMeansOperation = {
    new MLlibMiniBatchKMeansOperation(
      k = k,
      batchSize = batchSize,
      numBatches = numBatches,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      initMode = initMode
    )
  }
}
