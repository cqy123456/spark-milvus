package com.zilliz.spark.connector.operations.clustering

import com.zilliz.spark.connector.utils.VectorOps
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, DataTypes, FloatType}
import org.slf4j.{Logger, LoggerFactory}
import scala.collection.mutable
import scala.util.Random

/**
 * Float-native K-Means clustering operation
 *
 * Implementation based on Apache Spark MLlib KMeans but using Float precision throughout.
 * Reference: https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala
 *
 * Key differences from MLlib KMeans:
 * - Works directly with Array[Float] (Milvus format)
 * - Uses Float precision for all calculations
 * - No type conversions needed
 * - Optimized for Milvus vector data
 *
 * @param k Number of clusters
 * @param maxIter Maximum number of iterations
 * @param epsilon Convergence threshold
 * @param seed Random seed for initialization
 * @param featuresCol Name of the features column (must contain Array[Float])
 * @param predictionCol Name of the output prediction column (cluster_id)
 * @param initMode Initialization mode: "k-means||" (k-means++) or "random"
 * @param initSteps Number of steps for k-means|| initialization
 */
class KMeansOperation(
  val k: Int,
  val maxIter: Int = 100,
  val epsilon: Float = 1e-4f,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val initMode: String = "k-means||",
  val initSteps: Int = 2,
  val distanceMetric: String = "cosine"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[KMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(maxIter > 0, "Maximum iterations must be positive")
  require(epsilon > 0, "Epsilon must be positive")
  require(featuresCol.nonEmpty, "Features column name cannot be empty")
  require(predictionCol.nonEmpty, "Prediction column name cannot be empty")
  require(initMode == "k-means||" || initMode == "random",
    s"initMode must be 'k-means||' or 'random', got: $initMode")
  require(distanceMetric == "cosine" || distanceMetric == "l2",
    s"distanceMetric must be 'cosine' or 'l2', got: $distanceMetric")

  private val useCosine = distanceMetric == "cosine"

  /**
   * Normalize vector to unit length (L2 normalization)
   */
  private def normalizeL2(v: Array[Float]): Array[Float] = {
    val norm = math.sqrt(squaredNorm(v)).toFloat
    if (norm > 0) v.map(_ / norm) else v
  }

  /**
   * Compute squared L2 norm of a vector
   */
  private def squaredNorm(v: Array[Float]): Double = {
    var sum = 0.0
    var i = 0
    while (i < v.length) {
      sum += v(i).toDouble * v(i).toDouble
      i += 1
    }
    sum
  }

  /**
   * Compute dot product of two vectors
   */
  private def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
    var sum = 0.0f
    var i = 0
    while (i < v1.length) {
      sum += v1(i) * v2(i)
      i += 1
    }
    sum
  }

  /**
   * Find closest center using dot product (for cosine distance with normalized centers)
   * When centers are normalized, argmax(dot(point, center)) = argmin(cosine_distance)
   * Returns (closestIdx, cosineDistance)
   */
  private def findClosestCenterByDotProduct(
    point: Array[Float],
    centers: Array[Array[Float]]
  ): (Int, Float) = {
    var closestIdx = 0
    var maxDotProduct = dotProduct(point, centers(0))

    var i = 1
    while (i < centers.length) {
      val dp = dotProduct(point, centers(i))
      if (dp > maxDotProduct) {
        maxDotProduct = dp
        closestIdx = i
      }
      i += 1
    }

    // Compute cosine distance for the cost calculation
    // cosine_distance = 1 - cosine_similarity
    // cosine_similarity = dot(point, center) / (||point|| * ||center||)
    // Since ||center|| = 1, we have: cosine_similarity = dot(point, center) / ||point||
    val pointNorm = math.sqrt(squaredNorm(point)).toFloat
    val cosineSimilarity = if (pointNorm > 0) maxDotProduct / pointNorm else 0f
    val cosineDistance = 1f - cosineSimilarity

    (closestIdx, cosineDistance)
  }

  /**
   * Find closest center index using dot product (for cosine distance)
   */
  private def findClosestCenterIdxByDotProduct(point: Array[Float], centers: Array[Array[Float]]): Int = {
    var closestIdx = 0
    var maxDotProduct = dotProduct(point, centers(0))

    var i = 1
    while (i < centers.length) {
      val dp = dotProduct(point, centers(i))
      if (dp > maxDotProduct) {
        maxDotProduct = dp
        closestIdx = i
      }
      i += 1
    }
    closestIdx
  }

  /**
   * Train K-Means model and return cluster centers
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training Float K-Means: k=$k, maxIter=$maxIter, initMode=$initMode, epsilon=$epsilon, distanceMetric=$distanceMetric")

    val spark = df.sparkSession
    val startTime = System.currentTimeMillis()

    // Validate input format
    val featuresType = df.schema(featuresCol).dataType
    require(featuresType == ArrayType(FloatType, false) || featuresType == ArrayType(FloatType, true),
      s"Input DataFrame must have Array[Float] column, but got $featuresType")

    // Cache data for faster access
    df.cache()
    val totalCount = df.count()
    logger.info(s"Total data points: $totalCount")

    // Step 1: Initialize cluster centers
    logger.info(s"Initializing cluster centers with mode: $initMode...")
    val initStart = System.currentTimeMillis()

    val centers: Array[Array[Float]] = if (initMode == "random") {
      initializeRandom(df, k, seed)
    } else {
      initializeKMeansPlusPlus(df, k, seed, initSteps)
    }

    val initTime = System.currentTimeMillis() - initStart
    logger.info(s"Initialization completed in $initTime ms")

    // Normalize centers if using cosine distance
    if (useCosine) {
      logger.info("Normalizing initial centers for cosine distance")
      for (i <- centers.indices) {
        centers(i) = normalizeL2(centers(i))
      }
    }

    // Step 2: Lloyd's iterations
    logger.info("Starting Lloyd's iterations...")
    val (finalCenters, iterations, finalCost) = lloydIterations(df, centers, maxIter, epsilon, spark)

    val totalTime = System.currentTimeMillis() - startTime

    df.unpersist()

    // Log statistics
    logger.info("=" * 60)
    logger.info("Float K-Means Training Completed")
    logger.info("=" * 60)
    logger.info(s"Initialization time: $initTime ms")
    logger.info(s"Iterations completed: $iterations")
    logger.info(s"Final cost (WSSSE): $finalCost")
    logger.info(s"Total training time: $totalTime ms")
    logger.info("=" * 60)

    new KMeansModel(finalCenters, featuresCol, predictionCol, distanceMetric)
  }

  /**
   * Random initialization: randomly select k points as initial centers
   */
  private def initializeRandom(df: DataFrame, k: Int, seed: Long): Array[Array[Float]] = {
    logger.info(s"Using random initialization to select $k centers")

    val sample = df.sample(false, Math.min(0.1, k.toDouble / df.count()), seed)
      .limit(k)
      .select(featuresCol)
      .collect()
      .map(_.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray)

    if (sample.length < k) {
      logger.warn(s"Only sampled ${sample.length} points, expected $k")
    }

    sample
  }

  /**
   * K-Means++ (k-means||) initialization
   * Based on: "Scalable K-Means++" (Bahmani et al., 2012)
   */
  private def initializeKMeansPlusPlus(
    df: DataFrame,
    k: Int,
    seed: Long,
    steps: Int
  ): Array[Array[Float]] = {
    logger.info(s"Using k-means++ initialization with $steps steps")

    val spark = df.sparkSession
    val rand = new Random(seed)

    // Select first center randomly
    val firstCenter = df.sample(false, 0.01, seed)
      .limit(1)
      .select(featuresCol)
      .collect()
      .head
      .getAs[scala.collection.mutable.WrappedArray[Float]](0)
      .toArray

    var centers = Array(firstCenter)
    logger.info(s"Selected first center")

    // Iteratively select centers based on distance distribution
    for (step <- 0 until steps) {
      val bcCenters = spark.sparkContext.broadcast(centers)

      // Compute cost (squared distance to nearest center) for each point
      val costs = df.select(featuresCol).rdd.map { row =>
        val point = row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
        val minDist = findMinDistance(point, bcCenters.value)
        (point, minDist)
      }.persist(org.apache.spark.storage.StorageLevel.MEMORY_AND_DISK_SER)

      // Use treeAggregate to compute total cost in parallel
      val totalCost = costs.treeAggregate(0.0)(
        seqOp = (sum, pointCost) => sum + pointCost._2,
        combOp = (sum1, sum2) => sum1 + sum2,
        depth = 2
      )
      logger.info(s"Step ${step + 1}: Total cost = $totalCost")

      // Sample new centers proportional to their cost
      val numToSample = Math.max(1, (2 * k * Math.log(totalCost + 1)).toInt)
      val newCenters = costs.filter { case (_, cost) =>
        rand.nextDouble() < (2.0 * cost * k / totalCost)
      }.take(numToSample).map(_._1)

      centers = centers ++ newCenters
      costs.unpersist()
      bcCenters.destroy()

      logger.info(s"Step ${step + 1}: Selected ${newCenters.length} new centers, total = ${centers.length}")
    }

    // Reduce to exactly k centers using k-means on the candidates
    if (centers.length > k) {
      logger.info(s"Reducing ${centers.length} candidate centers to $k using k-means")
      kmeansPlusPlusReduce(centers, k, seed)
    } else {
      centers
    }
  }

  /**
   * Reduce candidate centers to k centers using weighted k-means
   */
  private def kmeansPlusPlusReduce(
    candidates: Array[Array[Float]],
    k: Int,
    seed: Long
  ): Array[Array[Float]] = {
    val rand = new Random(seed)

    // Initialize with k random candidates
    var centers = rand.shuffle(candidates.toSeq).take(k).toArray

    // Run a few iterations of Lloyd's on candidates
    for (_ <- 0 until 10) {
      val assignments = candidates.map(point => findClosestCenter(point, centers))

      // Recompute centers
      centers = (0 until k).map { clusterId =>
        val clusterPoints = candidates.zip(assignments).filter(_._2 == clusterId).map(_._1)
        if (clusterPoints.nonEmpty) {
          computeMean(clusterPoints)
        } else {
          centers(clusterId) // Keep old center if cluster is empty
        }
      }.toArray
    }

    centers
  }

  /**
   * Lloyd's iterations: assign points to nearest center and update centers
   */
  private def lloydIterations(
    df: DataFrame,
    initialCenters: Array[Array[Float]],
    maxIter: Int,
    epsilon: Float,
    spark: SparkSession
  ): (Array[Array[Float]], Int, Double) = {

    var centers = initialCenters
    var converged = false
    var iteration = 0
    var cost = 0.0

    while (iteration < maxIter && !converged) {
      iteration += 1
      logger.info(s"Iteration $iteration/$maxIter")

      val bcCenters = spark.sparkContext.broadcast(centers)

      // Assign each point to nearest center and compute statistics
      // Use treeAggregate to avoid collectAsMap overhead
      val localUseCosine = useCosine  // Capture for serialization
      val stats = df.select(featuresCol).rdd.treeAggregate(
        Array.fill(centers.length)((new Array[Double](centers(0).length), 0, 0.0))
      )(
        seqOp = (localStats, row) => {
          val point = row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
          val localCenters = bcCenters.value

          // Use dot product for cosine distance, L2 for euclidean
          val (closestCenter, distance) = if (localUseCosine) {
            // For cosine: argmax(dot(point, center)) when centers are normalized
            var closestIdx = 0
            var maxDp = {
              var sum = 0.0f
              var j = 0
              while (j < point.length) { sum += point(j) * localCenters(0)(j); j += 1 }
              sum
            }
            var i = 1
            while (i < localCenters.length) {
              var dp = 0.0f
              var j = 0
              while (j < point.length) { dp += point(j) * localCenters(i)(j); j += 1 }
              if (dp > maxDp) { maxDp = dp; closestIdx = i }
              i += 1
            }
            // Compute cosine distance for cost
            var pointNormSq = 0.0
            var k = 0
            while (k < point.length) { pointNormSq += point(k).toDouble * point(k).toDouble; k += 1 }
            val pointNorm = math.sqrt(pointNormSq).toFloat
            val cosineSim = if (pointNorm > 0) maxDp / pointNorm else 0f
            (closestIdx, (1f - cosineSim).toFloat)
          } else {
            findClosestCenterWithDistance(point, localCenters)
          }

          // Accumulate sum, count, and cost
          val (sum, count, _) = localStats(closestCenter)
          for (i <- point.indices) {
            sum(i) += point(i)
          }
          localStats(closestCenter) = (sum, count + 1, localStats(closestCenter)._3 + distance)
          localStats
        },
        combOp = (stats1, stats2) => {
          // Merge two stats arrays
          stats1.zip(stats2).map { case ((sum1, count1, cost1), (sum2, count2, cost2)) =>
            for (i <- sum1.indices) {
              sum1(i) += sum2(i)
            }
            (sum1, count1 + count2, cost1 + cost2)
          }
        },
        depth = 2  // Tree depth for parallel aggregation
      )

      bcCenters.destroy()

      // Compute total cost
      cost = stats.map(_._3).sum

      // Update centers
      var newCenters = stats.zipWithIndex.map { case ((sum, count, _), idx) =>
        if (count > 0) {
          sum.map(s => (s / count).toFloat)
        } else {
          centers(idx) // Keep old center if cluster is empty
        }
      }

      // Normalize centers after update if using cosine distance
      if (useCosine) {
        newCenters = newCenters.map { center =>
          val norm = math.sqrt(center.map(x => x.toDouble * x.toDouble).sum).toFloat
          if (norm > 0) center.map(_ / norm) else center
        }
      }

      // Check convergence
      val maxShift = centers.zip(newCenters).map { case (oldCenter, newCenter) =>
        squaredDistance(oldCenter, newCenter)
      }.max

      converged = maxShift <= epsilon * epsilon
      centers = newCenters

      logger.info(f"  Cost: $cost%.2f, Max shift: ${Math.sqrt(maxShift)}%.6f, Converged: $converged")
    }

    (centers, iteration, cost)
  }

  /**
   * Find minimum squared distance from point to any center
   */
  private def findMinDistance(point: Array[Float], centers: Array[Array[Float]]): Float = {
    var minDist = Float.MaxValue
    for (center <- centers) {
      val dist = squaredDistance(point, center)
      if (dist < minDist) {
        minDist = dist
      }
    }
    minDist
  }

  /**
   * Find closest center index
   * Uses ND4J batch distance computation for optimal SIMD performance
   */
  private def findClosestCenter(point: Array[Float], centers: Array[Array[Float]]): Int = {
    // Use ND4J batch computation to calculate distances to all centers at once
    val distances = VectorOps.batchEuclideanDistance(point, centers)

    // Find index of minimum distance
    var closestIdx = 0
    var minDist = distances(0)

    for (i <- 1 until distances.length) {
      if (distances(i) < minDist) {
        minDist = distances(i)
        closestIdx = i
      }
    }

    closestIdx
  }

  /**
   * Find closest center index with distance
   * Uses ND4J batch distance computation for optimal SIMD performance
   */
  private def findClosestCenterWithDistance(
    point: Array[Float],
    centers: Array[Array[Float]]
  ): (Int, Float) = {
    // Use ND4J batch computation to calculate distances to all centers at once
    val distances = VectorOps.batchEuclideanDistance(point, centers)

    // Find index of minimum distance
    var closestIdx = 0
    var minDist = distances(0)

    for (i <- 1 until distances.length) {
      if (distances(i) < minDist) {
        minDist = distances(i)
        closestIdx = i
      }
    }

    (closestIdx, minDist)
  }

  /**
   * Compute squared Euclidean distance between two Float arrays
   * Uses ND4J for SIMD-optimized computation
   */
  private def squaredDistance(v1: Array[Float], v2: Array[Float]): Float = {
    val dist = VectorOps.euclideanDistance(v1, v2)
    dist * dist  // Return squared distance
  }

  /**
   * Compute mean of multiple vectors
   */
  private def computeMean(vectors: Array[Array[Float]]): Array[Float] = {
    val dim = vectors(0).length
    val sum = new Array[Double](dim)

    vectors.foreach { vec =>
      for (i <- vec.indices) {
        sum(i) += vec(i)
      }
    }

    sum.map(s => (s / vectors.length).toFloat)
  }
}

/**
 * K-Means model trained with Float precision
 */
class KMeansModel(
  val clusterCenters: Array[Array[Float]],
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val distanceMetric: String = "cosine"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[KMeansModel])
  private val useCosine = distanceMetric == "cosine"

  /**
   * Transform DataFrame by assigning cluster IDs and computing distances
   *
   * OPTIMIZED: Uses mapPartitions with ND4J batch distance matrix computation
   * for SIMD-accelerated performance. pairwiseSquaredDistances handles memory
   * management internally, so no need for external batching.
   *
   * Input DataFrame schema:
   *   - features: Array[Float] - feature vectors
   *   - (any other columns from input)
   *
   * Output DataFrame schema:
   *   - features: Array[Float] - feature vectors (unchanged)
   *   - (any other columns from input)
   *   - cluster_id: Int - assigned cluster ID (0 to k-1)
   *   - distance: Float - squared distance to assigned cluster center
   *
   * Example:
   *   Input:  | features: Array[Float]           |
   *           | [1.0, 2.0, 3.0, ...]             |
   *
   *   Output: | features: Array[Float]           | cluster_id: Int | distance: Float |
   *           | [1.0, 2.0, 3.0, ...]             | 5               | 123.45          |
   */
  def transform(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(clusterCenters)

    // Define output schema
    val outputSchema = df.schema
      .add(predictionCol, DataTypes.IntegerType, false)
      .add("distance", DataTypes.FloatType, false)

    // MEMORY OPTIMIZED: Use mapPartitions with sub-batch processing
    // Each partition processes vectors in small batches, releasing memory after each batch
    val localUseCosine = useCosine  // Capture for serialization
    val localFeaturesCol = featuresCol  // Capture for serialization
    val subBatchSize = 5000  // Process 5000 vectors at a time per partition

    val resultRDD = df.rdd.mapPartitions { iter =>
      val localCenters = bcCenters.value
      val cachedTargets = VectorOps.createCachedTargets(localCenters, localUseCosine)

      try {
        iter.grouped(subBatchSize).flatMap { batch =>
          val rows = batch.toArray
          val vectors = rows.map(_.getAs[scala.collection.mutable.WrappedArray[Float]](localFeaturesCol).toArray)
          val assignments = cachedTargets.findNearest(vectors)
          rows.zip(assignments).map { case (row, (clusterId, rawValue)) =>
            val distance = if (localUseCosine) {
              val features = row.getAs[scala.collection.mutable.WrappedArray[Float]](localFeaturesCol).toArray
              var pointNormSq = 0.0
              var k = 0
              while (k < features.length) {
                pointNormSq += features(k).toDouble * features(k).toDouble
                k += 1
              }
              val pointNorm = math.sqrt(pointNormSq).toFloat
              val cosineSim = if (pointNorm > 0) rawValue / pointNorm else 0f
              1f - cosineSim
            } else {
              rawValue
            }
            Row.fromSeq(row.toSeq :+ clusterId :+ distance)
          }
        }
      } finally {
        cachedTargets.close()
      }
    }

    // Convert RDD back to DataFrame with output schema
    val result = spark.createDataFrame(resultRDD, outputSchema)

    // NOTE: Do NOT destroy bcCenters here!
    // DataFrame is lazily evaluated - destroying the broadcast before
    // the DataFrame is materialized (e.g., via count(), collect(), write())
    // will cause "Broadcast was destroyed" errors.
    // Let Spark manage broadcast lifecycle automatically.

    result
  }

  /**
   * Find closest cluster for a point
   * Uses ND4J batch distance computation for optimal SIMD performance
   */
  private def findClosestCluster(point: Array[Float], centers: Array[Array[Float]]): Int = {
    // Use ND4J batch computation to calculate distances to all centers at once
    val distances = VectorOps.batchEuclideanDistance(point, centers)

    // Find index of minimum distance
    var closestIdx = 0
    var minDist = distances(0)

    for (i <- 1 until distances.length) {
      if (distances(i) < minDist) {
        minDist = distances(i)
        closestIdx = i
      }
    }

    closestIdx
  }

  /**
   * Find closest cluster for a point and return cluster ID with distance
   * Uses ND4J batch distance computation for optimal SIMD performance
   */
  private def findClosestClusterWithDistance(point: Array[Float], centers: Array[Array[Float]]): (Int, Float) = {
    // Use ND4J batch computation to calculate distances to all centers at once
    val distances = VectorOps.batchEuclideanDistance(point, centers)

    // Find index of minimum distance
    var closestIdx = 0
    var minDist = distances(0)

    for (i <- 1 until distances.length) {
      if (distances(i) < minDist) {
        minDist = distances(i)
        closestIdx = i
      }
    }

    // Return squared distance for consistency with original implementation
    (closestIdx, minDist * minDist)
  }

  /**
   * Compute squared Euclidean distance
   * Uses ND4J for SIMD-optimized computation
   */
  private def squaredDistance(v1: Array[Float], v2: Array[Float]): Float = {
    val dist = VectorOps.euclideanDistance(v1, v2)
    dist * dist  // Return squared distance
  }

  /**
   * Compute total cost (Within-Set Sum of Squared Errors)
   */
  def computeCost(df: DataFrame): Double = {
    val spark = df.sparkSession
    val bcCenters = spark.sparkContext.broadcast(clusterCenters)

    df.select(featuresCol).rdd.map { row =>
      val point = row.getAs[scala.collection.mutable.WrappedArray[Float]](0).toArray
      val closestIdx = findClosestCluster(point, bcCenters.value)
      squaredDistance(point, bcCenters.value(closestIdx)).toDouble
    }.sum()
  }
}

object KMeansOperation {

  def apply(k: Int, maxIter: Int = 100): KMeansOperation = {
    new KMeansOperation(k = k, maxIter = maxIter)
  }

  def apply(
    k: Int,
    maxIter: Int,
    seed: Long,
    featuresCol: String,
    predictionCol: String,
    initMode: String
  ): KMeansOperation = {
    new KMeansOperation(
      k = k,
      maxIter = maxIter,
      seed = seed,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      initMode = initMode
    )
  }
}
