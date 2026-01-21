package com.zilliz.spark.connector.apps

import com.zilliz.spark.connector.operations.clustering.{KMeansOperation,MiniBatchKMeansOperation}
import com.zilliz.spark.connector.operations.graph.ConnectedComponentsOperation
import com.zilliz.spark.connector.utils.VectorOps
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.HashPartitioner
import org.slf4j.{Logger, LoggerFactory}

/**
 * Timing statistics for each step in the deduplication pipeline
 */
case class StepTiming(
  stepName: String,
  stepNumber: Int,
  durationSeconds: Double
)

/**
 * Complete timing statistics for the deduplication pipeline
 */
case class DedupTimingStats(
  stepTimings: Seq[StepTiming],
  totalDurationSeconds: Double
) {
  def getStepDuration(stepNumber: Int): Option[Double] = {
    stepTimings.find(_.stepNumber == stepNumber).map(_.durationSeconds)
  }

  def toMap: Map[String, Double] = {
    stepTimings.map(s => s"step${s.stepNumber}_${s.stepName}" -> s.durationSeconds).toMap +
      ("total" -> totalDurationSeconds)
  }

  override def toString: String = {
    val sb = new StringBuilder
    sb.append("=" * 60 + "\n")
    sb.append("Deduplication Pipeline Timing Statistics\n")
    sb.append("=" * 60 + "\n")
    stepTimings.foreach { s =>
      sb.append(f"  Step ${s.stepNumber}: ${s.stepName}%-30s ${s.durationSeconds}%10.2f sec\n")
    }
    sb.append("-" * 60 + "\n")
    sb.append(f"  Total:${" " * 33}${totalDurationSeconds}%10.2f sec\n")
    sb.append("=" * 60 + "\n")
    sb.toString
  }
}

/**
 * Vector Deduplication Application
 *
 * Pipeline:
 * 1. KMeans Clustering: Cluster vectors into k groups
 * 2. Repartition by Cluster: Move vectors to executors by cluster_id, sort by distance
 * 3. Build Graph + Union-Find: Within each bucket, find similar vectors and compute
 *    connected components using bucket-local Union-Find (no global iteration needed!)
 * 4. Deduplicate: Keep one representative from each component
 *
 * Input DataFrame schema:
 *   - id: Long - unique vector ID
 *   - features: Array[Float] - feature vectors
 *
 * Output DataFrame schema:
 *   - id: Long - unique vector ID
 *   - features: Array[Float] - feature vectors (deduplicated)
 *   - cluster_id: Int - assigned cluster
 *   - duplicate_group_id: Long - connected component ID (same ID = duplicates)
 *
 * Directory Structure (when cacheDir is provided):
 * vector_dedup/
 * ├── kmeans_results/
 * │   ├── kmeans_centroids.parquet       # Cluster centroids
 * │   └── assign_by_nearest_center/      # Embeddings organized by cluster
 * │       └── nearest_cent={0..n-1}/     # Subdirectories for each cluster
 * │           └── *.parquet               # Cluster member embeddings
 * ├── pairwise_results/                  # Threshold graph edges
 * │   └── *.parquet                      # Similarity scores by cluster
 * └── output/
 *     ├── duplicates/                    # Duplicate identification results
 *     │   └── *.parquet                  # Vector IDs marked as duplicates
 *     └── deduplicated/                  # Final clean dataset
 *         └── *.parquet                  # Deduplicated vectors
 *
 * @param local If true, use block-level Union-Find (good for local/dense graphs).
 *              If false, use global LargeStar algorithm (good for distributed/sparse graphs).
 * @param distanceMetric Distance metric for threshold graph: "l2" or "cosine". KMeans always uses L2.
 * @param kmeansAlgorithm KMeans algorithm: "standard" (full batch) or "minibatch" (mini-batch).
 * @param kmeansInitMode Initialization method: "k-means||" (k-means++) or "random".
 * @param miniBatchSampleRatio Sampling ratio for mini-batch KMeans (0.0, 1.0]. Only used when kmeansAlgorithm="minibatch".
 * @param cacheDir Optional directory for caching intermediate results. If None, no caching is performed.
 * @param outputDir Optional directory for saving final results. If None, results are not saved.
 *
 * Usage:
 *   val app = new VectorDedupApp(
 *     k = 100,
 *     distanceThreshold = 0.1f,
 *     local = true,
 *     distanceMetric = "l2",
 *     kmeansAlgorithm = "minibatch",
 *     kmeansInitMode = "k-means||",
 *     miniBatchSampleRatio = 0.1,
 *     cacheDir = Some("/path/to/cache"),
 *     outputDir = Some("/path/to/output")
 *   )
 *   val dedupedDF = app.deduplicate(inputDF)
 */
class VectorDedupApp(
  val k: Int,
  val distanceThreshold: Float = 0.1f,
  val maxIterKMeans: Int = 20,
  val local: Boolean = true,
  val distanceMetric: String = "l2",
  val kmeansAlgorithm: String = "standard",
  val kmeansInitMode: String = "k-means||",
  val miniBatchSampleRatio: Double = 0.1,
  val featuresCol: String = "features",
  val idCol: String = "id",
  val cacheDir: Option[String] = None,
  val outputDir: Option[String] = None
) extends Serializable {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(classOf[VectorDedupApp])

  require(k > 0, "Number of clusters must be positive")
  require(distanceThreshold > 0, "Distance threshold must be positive")
  require(distanceMetric == "l2" || distanceMetric == "cosine",
    s"distanceMetric must be 'l2' or 'cosine', got: $distanceMetric")
  require(kmeansAlgorithm == "standard" || kmeansAlgorithm == "minibatch",
    s"kmeansAlgorithm must be 'standard' or 'minibatch', got: $kmeansAlgorithm")
  require(kmeansInitMode == "random" || kmeansInitMode == "k-means||",
    s"kmeansInitMode must be 'random' or 'k-means||', got: $kmeansInitMode")
  require(miniBatchSampleRatio > 0.0 && miniBatchSampleRatio <= 1.0,
    s"miniBatchSampleRatio must be in (0.0, 1.0], got: $miniBatchSampleRatio")

  /**
   * Main deduplication pipeline
   *
   * Input DataFrame:
   *   | id: Long | features: Array[Float]           |
   *   | 1        | [1.0, 2.0, 3.0, ...]             |
   *   | 2        | [1.1, 2.1, 3.1, ...]             |
   *
   * Output DataFrame:
   *   | id: Long | features: Array[Float] | cluster_id: Int | is_representative: Boolean | component_id: Long |
   *   | 1        | [1.0, 2.0, ...]        | 5               | true                       | 1001               |
   *   | 2        | [1.1, 2.1, ...]        | 5               | false                      | 1001               |
   *
   * Where:
   *   - is_representative: true if this vector is the representative of its duplicate group
   *   - component_id: ID of the connected component (duplicate group)
   */
  // Store the latest timing statistics
  @transient private var _lastTimingStats: Option[DedupTimingStats] = None

  /**
   * Get the timing statistics from the last deduplication run
   */
  def lastTimingStats: Option[DedupTimingStats] = _lastTimingStats

  def deduplicate(df: DataFrame): DataFrame = {
    val (result, _) = deduplicateWithTiming(df)
    result
  }

  /**
   * Main deduplication pipeline with detailed timing statistics
   *
   * @return A tuple of (deduplicated DataFrame, timing statistics)
   */
  def deduplicateWithTiming(df: DataFrame): (DataFrame, DedupTimingStats) = {
    logger.info("=" * 80)
    logger.info("Starting Vector Deduplication Pipeline")
    logger.info("=" * 80)
    logger.info(s"Configuration: k=$k, distanceThreshold=$distanceThreshold, maxIterKMeans=$maxIterKMeans")
    logger.info(s"Distance Metric: $distanceMetric (for threshold graph)")
    logger.info(s"Connected Components Algorithm: ${if (local) "Block-level Union-Find (local)" else "LargeStar (distributed)"}")
    cacheDir.foreach(dir => logger.info(s"Cache directory: $dir"))
    outputDir.foreach(dir => logger.info(s"Output directory: $dir"))

    val spark = df.sparkSession
    val pipelineStartTime = System.currentTimeMillis()
    val stepTimings = scala.collection.mutable.ArrayBuffer[StepTiming]()

    // Validate input
    validateInput(df)

    val totalVectors = df.count()
    logger.info(s"Total input vectors: $totalVectors")

    // Step 1: KMeans Clustering
    var stepStart = System.currentTimeMillis()
    val clusteredDF = performKMeansClustering(df)
    saveKMeansResults(clusteredDF)
    var stepDuration = (System.currentTimeMillis() - stepStart) / 1000.0
    stepTimings += StepTiming("KMeans Clustering", 1, stepDuration)
    logger.info(f"Step 1 (KMeans Clustering) completed in $stepDuration%.2f seconds")

    // Step 2: Repartition by Cluster and Sort by Distance
    // NOTE: For S3 data > disk capacity, avoid .cache() here.
    //       This prevents writing entire dataset to disk.
    stepStart = System.currentTimeMillis()
    val sortedDF = repartitionAndSortByClusters(clusteredDF)
    // Trigger computation without explicit caching
    sortedDF.count()
    stepDuration = (System.currentTimeMillis() - stepStart) / 1000.0
    stepTimings += StepTiming("Repartition & Sort", 2, stepDuration)
    logger.info(f"Step 2 (Repartition & Sort) completed in $stepDuration%.2f seconds")

    // Step 3: Build Threshold Graph + Find Connected Components + Deduplicate (all in one pass)
    // This combines graph building, Union-Find, and deduplication into a single mapPartitions.
    // Since each bucket is independent (no cross-bucket edges), we can:
    // 1. Run Union-Find locally within each partition
    // 2. Compute min distance per component locally
    // 3. Mark representatives directly
    // All without any shuffle operations!
    stepStart = System.currentTimeMillis()
    val dedupedDF = buildGraphFindComponentsAndDeduplicate(sortedDF)
    dedupedDF.cache()

    // Compute statistics
    val stats = dedupedDF.agg(
      count("*").as("total"),
      sum(when(col("is_representative"), 1).otherwise(0)).as("representatives"),
      countDistinct("component_id").as("components")
    ).head()

    val total = stats.getAs[Long]("total")
    val representatives = stats.getAs[Long]("representatives")
    val componentCount = stats.getAs[Long]("components")
    val duplicates = total - representatives
    val deduplicationRate = duplicates.toDouble / total * 100

    stepDuration = (System.currentTimeMillis() - stepStart) / 1000.0
    stepTimings += StepTiming("Graph + Union-Find + Dedup", 3, stepDuration)
    logger.info(f"Step 3 (Graph + Union-Find + Dedup) completed in $stepDuration%.2f seconds")
    logger.info(s"Distinct components: $componentCount")

    // Log deduplication statistics
    logger.info(s"Deduplication complete:")
    logger.info(s"  Total vectors: $total")
    logger.info(s"  Representatives: $representatives")
    logger.info(s"  Duplicates removed: $duplicates")
    logger.info(f"  Deduplication rate: $deduplicationRate%.2f%%")

    // Log component size distribution
    logger.info("Component size distribution:")
    dedupedDF
      .groupBy("component_id")
      .agg(count("*").as("component_size"))
      .groupBy("component_size")
      .agg(count("*").as("num_components"))
      .orderBy("component_size")
      .show(20, false)

    // Save final results
    saveFinalResults(dedupedDF)

    // Unpersist cached DataFrames
    sortedDF.unpersist()
    dedupedDF.unpersist()

    val totalDuration = (System.currentTimeMillis() - pipelineStartTime) / 1000.0
    val timingStats = DedupTimingStats(stepTimings.toSeq, totalDuration)
    _lastTimingStats = Some(timingStats)

    logger.info("=" * 80)
    logger.info("Vector Deduplication Pipeline Completed")
    logger.info("=" * 80)
    logger.info(timingStats.toString)

    (dedupedDF, timingStats)
  }

  /**
   * Step 1: Perform KMeans Clustering
   *
   * Input:  | id: Long | features: Array[Float]           |
   *
   * Output: | id: Long | features: Array[Float]           | cluster_id: Int | distance: Float |
   */
  private def performKMeansClustering(df: DataFrame): DataFrame = {
    logger.info("Step 1: Performing KMeans clustering...")
    logger.info(s"  Algorithm: $kmeansAlgorithm")
    logger.info(s"  Init mode: $kmeansInitMode")
    if (kmeansAlgorithm == "minibatch") {
      logger.info(s"  Sample ratio: $miniBatchSampleRatio")
    }
    val startTime = System.currentTimeMillis()

    // Train KMeans model based on selected algorithm
    val trainingStartTime = System.currentTimeMillis()
    val model = kmeansAlgorithm match {
      case "standard" =>
        val kmeans = new KMeansOperation(
          k = k,
          maxIter = maxIterKMeans,
          featuresCol = featuresCol,
          initMode = kmeansInitMode,
          predictionCol = "cluster_id"
        )
        logger.info(s"Training standard KMeans with k=$k, maxIter=$maxIterKMeans...")
        kmeans.fit(df.select(featuresCol))

      case "minibatch" =>
        val kmeans = new MiniBatchKMeansOperation(
          k = k,
          batchSize = miniBatchSampleRatio,
          numBatches = (1.0 / miniBatchSampleRatio).toInt max 1,
          featuresCol = featuresCol,
          initMode = kmeansInitMode,
          predictionCol = "cluster_id"
        )
        logger.info(s"Training mini-batch KMeans with k=$k, sampleRatio=$miniBatchSampleRatio...")
        kmeans.fit(df.select(featuresCol))
    }
    val trainingTime = (System.currentTimeMillis() - trainingStartTime) / 1000.0
    logger.info(f"✓ KMeans training completed in $trainingTime%.2f seconds")

    // Transform: add cluster_id and distance columns
    logger.info("Assigning cluster IDs and computing distances...")
    val transformStartTime = System.currentTimeMillis()
    val clusteredDF = model.transform(df)
    clusteredDF.cache().count()
    val transformTime = (System.currentTimeMillis() - transformStartTime) / 1000.0
    logger.info(f"✓ Transform and distance computation completed in $transformTime%.2f seconds")

    val clusteringTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ KMeans clustering (total) completed in $clusteringTime%.2f seconds")

    // Log cluster statistics
    //logClusterStatistics(clusteredDF)

    clusteredDF
  }

  /**
   * Step 2: Repartition by Cluster ID and Sort by Distance
   *
   * This ensures:
   * 1. All vectors in the same cluster are on the same executor
   * 2. Within each cluster, vectors are sorted by distance to center (closest first)
   *
   * Input:  | id: Long | features: Array[Float]           | cluster_id: Int | distance: Float |
   *
   * Output: | id: Long | features: Array[Float]           | cluster_id: Int | distance: Float |
   *         (repartitioned by cluster_id, sorted by distance within each partition)
   */
  private def repartitionAndSortByClusters(df: DataFrame): DataFrame = {
    logger.info("Step 2: Repartitioning by cluster_id and sorting by distance...")
    val startTime = System.currentTimeMillis()

    val spark = df.sparkSession

    // Get cluster count for determining partitions
    val clusterCount = df.select("cluster_id").distinct().count().toInt
    logger.info(s"Number of clusters: $clusterCount")

    // Use DataFrame API for memory-efficient repartitioning and sorting
    // This avoids loading entire partitions into memory (unlike RDD toSeq.sortBy)
    logger.info(s"Repartitioning into $clusterCount partitions by cluster_id...")

    // repartition by cluster_id ensures same cluster goes to same partition
    // sortWithinPartitions uses external sort, avoiding OOM on large partitions
    val sortedDF = df
      .repartition(clusterCount, col("cluster_id"))
      .sortWithinPartitions(col("cluster_id"), col("distance").asc)

    val repartitionTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Repartitioning and sorting completed in $repartitionTime%.2f seconds")

    // Verify partitioning
    val finalPartitions = sortedDF.rdd.getNumPartitions
    logger.info(s"Final number of partitions: $finalPartitions")

    // Log bucket (cluster) size distribution
    val bucketStats = sortedDF
      .groupBy("cluster_id")
      .agg(count("*").as("count"))
      .orderBy("cluster_id")
      .collect()

    logger.info("Bucket entries distribution:")
    bucketStats.foreach { row =>
      val clusterId = row.getAs[Int]("cluster_id")
      val count = row.getAs[Long]("count")
      logger.info(s"  Bucket $clusterId: $count entries")
    }

    sortedDF
  }

  /**
   * Step 3: Build Threshold Graph, Find Connected Components, and Deduplicate (all in one pass)
   *
   * This method combines graph building, Union-Find, and deduplication into a single mapPartitions:
   * 1. Vectors are already sorted by distance to center (by previous step)
   * 2. Compare each vector with others WITHIN SAME BUCKET
   * 3. If distance < threshold: union the two vectors in Union-Find
   * 4. Compute min distance per component locally
   * 5. Mark representatives (vectors with min distance in their component)
   *
   * Since buckets are independent (no cross-bucket edges), everything is done locally
   * within each partition - NO SHUFFLE NEEDED!
   *
   * Input:  | id: Long | features: Array[Float] | cluster_id: Int | distance: Float |
   *         (partitioned by cluster_id, sorted by distance within partition)
   *
   * Output: | id: Long | features: Array[Float] | cluster_id: Int | distance: Float | component_id: Long | is_representative: Boolean |
   *         (complete deduplicated result with all fields)
   */
  private def buildGraphFindComponentsAndDeduplicate(df: DataFrame): DataFrame = {
    logger.info("Step 3: Building threshold graph, finding components, and deduplicating (all in one pass)...")
    val startTime = System.currentTimeMillis()

    val spark = df.sparkSession
    val threshold = distanceThreshold
    val metric = distanceMetric
    val featCol = featuresCol
    val idColumn = idCol

    // Get the schema for output
    val outputSchema = df.schema
      .add("component_id", LongType, nullable = false)
      .add("is_representative", BooleanType, nullable = false)

    // Build graph, find components, and deduplicate within each partition
    val resultRDD = df.rdd.mapPartitions { rowIterator =>
      // Collect all vectors in this partition first (needed for Union-Find)
      val vectors = rowIterator.toArray

      if (vectors.isEmpty) {
        Iterator.empty
      } else {
        val bucketId = vectors(0).getAs[Int]("cluster_id")
        val totalVectors = vectors.length

        // Union-Find data structure using arrays for efficiency
        val idToIndex = scala.collection.mutable.HashMap[Long, Int]()
        val indexToId = new Array[Long](totalVectors)
        val parent = new Array[Int](totalVectors)
        val rank = new Array[Int](totalVectors)

        // Initialize Union-Find: each vector is its own component
        var idx = 0
        while (idx < totalVectors) {
          val id = vectors(idx).getAs[Long](idColumn)
          idToIndex(id) = idx
          indexToId(idx) = id
          parent(idx) = idx
          rank(idx) = 0
          idx += 1
        }

        // Union-Find: find with path compression
        def find(x: Int): Int = {
          if (parent(x) != x) {
            parent(x) = find(parent(x))
          }
          parent(x)
        }

        // Union-Find: union by rank
        def union(x: Int, y: Int): Unit = {
          val rootX = find(x)
          val rootY = find(y)
          if (rootX != rootY) {
            if (rank(rootX) < rank(rootY)) {
              parent(rootX) = rootY
            } else if (rank(rootX) > rank(rootY)) {
              parent(rootY) = rootX
            } else {
              parent(rootY) = rootX
              rank(rootX) += 1
            }
          }
        }

        // Distance metric flags
        val useCosine = metric == "cosine"
        val thresholdSq = threshold * threshold

        // Pre-extract all feature vectors and distances to center for efficient access
        val featureVectors = new Array[Array[Float]](totalVectors)
        val distToCenter = new Array[Float](totalVectors)
        var k = 0
        while (k < totalVectors) {
          featureVectors(k) = vectors(k).getAs[scala.collection.mutable.WrappedArray[Float]](featCol).toArray
          distToCenter(k) = vectors(k).getAs[Float]("distance")
          k += 1
        }

        // Inline distance computation
        def computeL2DistanceSq(v1: Array[Float], v2: Array[Float]): Float = {
          var sumSq = 0.0f
          var idx = 0
          val len = v1.length
          while (idx < len) {
            val diff = v1(idx) - v2(idx)
            sumSq += diff * diff
            idx += 1
          }
          sumSq
        }

        def computeCosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
          var dotProduct = 0.0f
          var norm1Sq = 0.0f
          var norm2Sq = 0.0f
          var idx = 0
          val len = v1.length
          while (idx < len) {
            dotProduct += v1(idx) * v2(idx)
            norm1Sq += v1(idx) * v1(idx)
            norm2Sq += v2(idx) * v2(idx)
            idx += 1
          }
          val norm1 = math.sqrt(norm1Sq).toFloat
          val norm2 = math.sqrt(norm2Sq).toFloat
          if (norm1 > 0 && norm2 > 0) {
            1.0f - (dotProduct / (norm1 * norm2))
          } else {
            1.0f
          }
        }

        var edgeCount = 0
        var prunedByLowerBound = 0L
        var prunedByUpperBound = 0L

        // Full pairwise comparison with triangle inequality pruning
        var i = 0
        while (i < totalVectors) {
          val features_i = featureVectors(i)
          val dist_i = distToCenter(i)
          var j = i + 1
          while (j < totalVectors) {
            val dist_j = distToCenter(j)
            val distDiff = math.abs(dist_i - dist_j)

            if (useCosine) {
              // Cosine: lower bound pruning
              if (distDiff > threshold) {
                prunedByLowerBound += 1
              } else {
                val dist = computeCosineDistance(features_i, featureVectors(j))
                if (dist < threshold) {
                  union(i, j)
                  edgeCount += 1
                }
              }
            } else {
              // L2: both upper and lower bound pruning
              if (distDiff > threshold) {
                prunedByLowerBound += 1
              } else {
                val distSum = dist_i + dist_j
                if (distSum < threshold) {
                  union(i, j)
                  edgeCount += 1
                  prunedByUpperBound += 1
                } else {
                  val distSq = computeL2DistanceSq(features_i, featureVectors(j))
                  if (distSq < thresholdSq) {
                    union(i, j)
                    edgeCount += 1
                  }
                }
              }
            }
            j += 1
          }
          i += 1
        }

        // Compute min distance per component (locally, no shuffle!)
        val componentMinDist = scala.collection.mutable.HashMap[Long, Float]()
        var m = 0
        while (m < totalVectors) {
          val rootIdx = find(m)
          val componentId = indexToId(rootIdx)
          val dist = distToCenter(m)
          componentMinDist.get(componentId) match {
            case Some(minDist) =>
              if (dist < minDist) componentMinDist(componentId) = dist
            case None =>
              componentMinDist(componentId) = dist
          }
          m += 1
        }

        val totalPairs = totalVectors.toLong * (totalVectors - 1) / 2
        val computed = totalPairs - prunedByLowerBound - prunedByUpperBound
        val componentCount = componentMinDist.size
        logger.info(f"  Bucket $bucketId: $totalVectors vectors, $edgeCount edges, $componentCount components, " +
          f"pruned: ${prunedByLowerBound + prunedByUpperBound} (lower=$prunedByLowerBound, upper=$prunedByUpperBound), " +
          f"computed: $computed/${totalPairs} (${computed * 100.0 / totalPairs}%.1f%%)")

        // Output complete rows with component_id and is_representative
        vectors.indices.iterator.map { idx =>
          val row = vectors(idx)
          val rootIdx = find(idx)
          val componentId = indexToId(rootIdx)
          val dist = distToCenter(idx)
          val isRepresentative = dist == componentMinDist(componentId)

          // Build output row: original fields + component_id + is_representative
          org.apache.spark.sql.Row.fromSeq(row.toSeq :+ componentId :+ isRepresentative)
        }
      }
    }

    val resultDF = spark.createDataFrame(resultRDD, outputSchema)

    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Graph + Union-Find + Dedup completed in $totalTime%.2f seconds")

    resultDF
  }

  /**
   * Old Step 3: Build Threshold Graph and Find Connected Components within each Bucket
   * (Kept for backward compatibility, but not used in optimized pipeline)
   */
  private def buildThresholdGraphAndFindComponents(df: DataFrame): DataFrame = {
    logger.info("Step 3: Building threshold graph and finding components (bucket-local Union-Find)...")
    val startTime = System.currentTimeMillis()

    val spark = df.sparkSession
    val threshold = distanceThreshold
    val metric = distanceMetric
    val featCol = featuresCol
    val idColumn = idCol

    // Build graph and find components within each partition using Union-Find
    val componentsRDD = df.rdd.mapPartitions { rowIterator =>
      // Collect all vectors in this partition first (needed for Union-Find)
      val vectors = rowIterator.toArray

      if (vectors.isEmpty) {
        Iterator.empty
      } else {
        val bucketId = vectors(0).getAs[Int]("cluster_id")
        val totalVectors = vectors.length

        // Union-Find data structure using arrays for efficiency
        // Map vector id -> index for Union-Find operations
        val idToIndex = scala.collection.mutable.HashMap[Long, Int]()
        val indexToId = new Array[Long](totalVectors)
        val parent = new Array[Int](totalVectors)
        val rank = new Array[Int](totalVectors)

        // Initialize Union-Find: each vector is its own component
        var idx = 0
        while (idx < totalVectors) {
          val id = vectors(idx).getAs[Long](idColumn)
          idToIndex(id) = idx
          indexToId(idx) = id
          parent(idx) = idx
          rank(idx) = 0
          idx += 1
        }

        // Union-Find: find with path compression
        def find(x: Int): Int = {
          if (parent(x) != x) {
            parent(x) = find(parent(x))
          }
          parent(x)
        }

        // Union-Find: union by rank
        def union(x: Int, y: Int): Unit = {
          val rootX = find(x)
          val rootY = find(y)
          if (rootX != rootY) {
            if (rank(rootX) < rank(rootY)) {
              parent(rootX) = rootY
            } else if (rank(rootX) > rank(rootY)) {
              parent(rootY) = rootX
            } else {
              parent(rootY) = rootX
              rank(rootX) += 1
            }
          }
        }

        // Distance metric flags
        val useCosine = metric == "cosine"
        val thresholdSq = threshold * threshold

        // Pre-extract all feature vectors and distances to center for efficient access
        val featureVectors = new Array[Array[Float]](totalVectors)
        val distToCenter = new Array[Float](totalVectors)
        var k = 0
        while (k < totalVectors) {
          featureVectors(k) = vectors(k).getAs[scala.collection.mutable.WrappedArray[Float]](featCol).toArray
          distToCenter(k) = vectors(k).getAs[Float]("distance")
          k += 1
        }

        // Inline distance computation
        def computeL2DistanceSq(v1: Array[Float], v2: Array[Float]): Float = {
          var sumSq = 0.0f
          var idx = 0
          val len = v1.length
          while (idx < len) {
            val diff = v1(idx) - v2(idx)
            sumSq += diff * diff
            idx += 1
          }
          sumSq
        }

        def computeCosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
          var dotProduct = 0.0f
          var norm1Sq = 0.0f
          var norm2Sq = 0.0f
          var idx = 0
          val len = v1.length
          while (idx < len) {
            dotProduct += v1(idx) * v2(idx)
            norm1Sq += v1(idx) * v1(idx)
            norm2Sq += v2(idx) * v2(idx)
            idx += 1
          }
          val norm1 = math.sqrt(norm1Sq).toFloat
          val norm2 = math.sqrt(norm2Sq).toFloat
          if (norm1 > 0 && norm2 > 0) {
            1.0f - (dotProduct / (norm1 * norm2))
          } else {
            1.0f
          }
        }

        var edgeCount = 0
        var prunedByLowerBound = 0L
        var prunedByUpperBound = 0L

        // Full pairwise comparison with triangle inequality pruning
        var i = 0
        while (i < totalVectors) {
          val features_i = featureVectors(i)
          val dist_i = distToCenter(i)
          var j = i + 1
          while (j < totalVectors) {
            val dist_j = distToCenter(j)
            val distDiff = math.abs(dist_i - dist_j)

            if (useCosine) {
              // Cosine: lower bound pruning
              // If |d_i - d_j| > threshold, then dist(v_i, v_j) > threshold → not duplicate
              if (distDiff > threshold) {
                prunedByLowerBound += 1
              } else {
                val dist = computeCosineDistance(features_i, featureVectors(j))
                if (dist < threshold) {
                  union(i, j)
                  edgeCount += 1
                }
              }
            } else {
              // L2: both upper and lower bound pruning
              // Lower bound: |d_i - d_j| > threshold → not duplicate
              if (distDiff > threshold) {
                prunedByLowerBound += 1
              } else {
                val distSum = dist_i + dist_j
                // Upper bound: d_i + d_j < threshold → must be duplicate
                if (distSum < threshold) {
                  union(i, j)
                  edgeCount += 1
                  prunedByUpperBound += 1
                } else {
                  // Need actual distance computation
                  val distSq = computeL2DistanceSq(features_i, featureVectors(j))
                  if (distSq < thresholdSq) {
                    union(i, j)
                    edgeCount += 1
                  }
                }
              }
            }
            j += 1
          }
          i += 1
        }

        val totalPairs = totalVectors.toLong * (totalVectors - 1) / 2
        val computed = totalPairs - prunedByLowerBound - prunedByUpperBound
        logger.info(f"  Bucket $bucketId: $totalVectors vectors, $edgeCount edges, " +
          f"pruned: ${prunedByLowerBound + prunedByUpperBound} (lower=$prunedByLowerBound, upper=$prunedByUpperBound), " +
          f"computed: $computed/${totalPairs} (${computed * 100.0 / totalPairs}%.1f%%)")

        // Output: (id, component_id) where component_id is the id of the root
        vectors.indices.iterator.map { idx =>
          val id = indexToId(idx)
          val rootIdx = find(idx)
          val componentId = indexToId(rootIdx)
          (id, componentId)
        }
      }
    }

    import spark.implicits._
    val componentsDF = componentsRDD.toDF("id", "component_id")

    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Threshold graph + Union-Find completed in $totalTime%.2f seconds")

    componentsDF
  }

  /**
   * Compute distance between two Float arrays based on distanceMetric
   * Uses ND4J for optimized SIMD operations
   */
  private def computeDistance(v1: Array[Float], v2: Array[Float]): Float = {
    distanceMetric match {
      case "l2" => VectorOps.euclideanDistance(v1, v2)
      case "cosine" => VectorOps.cosineDistance(v1, v2)
      case _ => throw new IllegalArgumentException(s"Unknown distance metric: $distanceMetric")
    }
  }

  /**
   * Step 4: Find Connected Components
   *
   * Choose algorithm based on local parameter:
   * - local=true: Block-level Union-Find (good for dense graphs, local computation)
   * - local=false: LargeStar algorithm (good for sparse graphs, distributed computation)
   *
   * Input:  | src_id: Long | dst_id: Long | edge_distance: Float |
   *
   * Output: | id: Long | component_id: Long |
   *         (mapping from vector ID to component ID)
   */
  private def findConnectedComponents(graphDF: DataFrame): DataFrame = {
    logger.info("Step 4: Finding connected components...")

    // Create ConnectedComponentsOperation with local parameter
    // local=true: Custom block-level Union-Find (iterative propagation)
    // local=false: GraphFrames LargeStar algorithm (distributed)
    val ccOperation = new ConnectedComponentsOperation(
      local = local,
      maxIterations = 20,
      checkpointInterval = 5
    )

    // Find connected components (automatically chooses algorithm based on local)
    val components = ccOperation.findComponents(graphDF)

    components
  }

  /**
   * Step 5: Deduplicate by Components
   *
   * For each connected component, keep only one representative vector.
   * Strategy: keep the vector closest to cluster center (first in sorted order).
   *
   * Input:  | id: Long | features: Array[Float] | cluster_id: Int | distance: Float | component_id: Long |
   *
   * Output: | id: Long | features: Array[Float] | cluster_id: Int | is_representative: Boolean | component_id: Long |
   *         (deduplicated vectors with representative flag)
   */
  private def deduplicateByComponents(df: DataFrame): DataFrame = {
    logger.info("Step 5: Deduplicating by connected components...")

    val spark = df.sparkSession
    import spark.implicits._

    // For each component, find the representative (the one with minimum distance to cluster center)
    // Since the data is already sorted by distance within each partition/cluster,
    // we need to find the global minimum distance for each component_id
    val componentRepresentatives = df
      .groupBy("component_id")
      .agg(
        min("distance").as("min_distance")
      )

    // Join back to original data and mark representatives
    // A vector is a representative if its distance equals the minimum distance for its component
    val deduplicatedDF = df
      .join(componentRepresentatives, "component_id")
      .withColumn(
        "is_representative",
        when($"distance" === $"min_distance", true).otherwise(false)
      )
      .drop("min_distance")

    // Log deduplication statistics
    val totalVectors = df.count()
    val representatives = deduplicatedDF.filter($"is_representative").count()
    val duplicates = totalVectors - representatives
    val deduplicationRate = (duplicates.toDouble / totalVectors * 100)

    logger.info(s"Deduplication complete:")
    logger.info(s"  Total vectors: $totalVectors")
    logger.info(s"  Representatives: $representatives")
    logger.info(s"  Duplicates removed: $duplicates")
    logger.info(f"  Deduplication rate: $deduplicationRate%.2f%%")

    // Log component size distribution
    val componentSizes = deduplicatedDF
      .groupBy("component_id")
      .agg(count("*").as("component_size"))
      .groupBy("component_size")
      .agg(count("*").as("num_components"))
      .orderBy("component_size")

    logger.info("Component size distribution:")
    componentSizes.show(20, false)

    deduplicatedDF
  }

  /**
   * Validate input DataFrame schema
   */
  private def validateInput(df: DataFrame): Unit = {
    require(df.columns.contains(idCol), s"Input DataFrame must contain '$idCol' column")
    require(df.columns.contains(featuresCol), s"Input DataFrame must contain '$featuresCol' column")

    val featuresType = df.schema(featuresCol).dataType
    require(
      featuresType == ArrayType(FloatType, false) || featuresType == ArrayType(FloatType, true),
      s"'$featuresCol' column must be Array[Float], but got $featuresType"
    )

    logger.info("✓ Input validation passed")
  }

  /**
   * Log cluster statistics
   */
  private def logClusterStatistics(df: DataFrame): Unit = {
    logger.info("Cluster statistics:")

    val stats = df.groupBy("cluster_id")
      .agg(
        count("*").alias("count"),
        avg("distance").alias("avg_distance"),
        min("distance").alias("min_distance"),
        max("distance").alias("max_distance")
      )
      .orderBy("cluster_id")
      .collect()

    stats.take(10).foreach { row =>
      val clusterId = row.getAs[Int]("cluster_id")
      val count = row.getAs[Long]("count")
      val avgDist = row.getAs[Double]("avg_distance")
      val minDist = row.getAs[Double]("min_distance")
      val maxDist = row.getAs[Double]("max_distance")

      logger.info(f"  Cluster $clusterId%3d: count=$count%6d, avg_dist=$avgDist%8.2f, min_dist=$minDist%8.2f, max_dist=$maxDist%8.2f")
    }

    if (stats.length > 10) {
      logger.info(s"  ... (showing first 10 of ${stats.length} clusters)")
    }
  }

  // ==================== Save Methods ====================
  // TODO: Implement load methods to reuse cached intermediate results

  /**
   * Save KMeans clustering results
   * Called internally after Step 1 if cacheDir is provided
   */
  private def saveKMeansResults(clusteredDF: DataFrame): Unit = {
    cacheDir.foreach { dir =>
      val path = s"$dir/kmeans_results/assign_by_nearest_center"
      logger.info(s"Saving KMeans cluster assignments to: $path")
      // Save partitioned by cluster_id to create subdirectories cluster_id={0..k-1}
      clusteredDF
        .write
        .mode("overwrite")
        .partitionBy("cluster_id")
        .parquet(path)
      logger.info(s"✓ KMeans results saved")
    }
  }

  /**
   * Save threshold graph edges
   * Called internally after Step 3 if cacheDir is provided
   */
  private def saveGraphEdges(edgesDF: DataFrame): Unit = {
    cacheDir.foreach { dir =>
      val path = s"$dir/pairwise_results"
      logger.info(s"Saving threshold graph edges to: $path")
      edgesDF.write.mode("overwrite").parquet(path)
      logger.info(s"✓ Graph edges saved")
    }
  }

  /**
   * Save final results (duplicates and deduplicated vectors)
   * Called internally at the end of pipeline if outputDir is provided
   */
  private def saveFinalResults(deduplicatedDF: DataFrame): Unit = {
    val spark = deduplicatedDF.sparkSession
    import spark.implicits._

    outputDir.foreach { dir =>
      // Save duplicates (is_representative = false)
      // NOTE: For S3 data > disk capacity:
      //   - Only save essential columns to minimize output size
      //   - Use coalesce() to reduce number of output files
      val duplicatesPath = s"$dir/duplicates"
      logger.info(s"Saving duplicates to: $duplicatesPath")
      deduplicatedDF
        .filter(!$"is_representative")
        .select(idCol, "component_id")
        .coalesce(100)  // Limit output files (adjust based on cluster size)
        .write
        .mode("overwrite")
        .parquet(duplicatesPath)
      logger.info(s"✓ Duplicates saved")

      // Save deduplicated vectors (is_representative = true)
      // Select only essential columns to reduce output size by ~90%
      val dedupedPath = s"$dir/deduplicated"
      logger.info(s"Saving deduplicated vectors to: $dedupedPath")
      deduplicatedDF
        .filter($"is_representative")
        .select(idCol, "features", "cluster_id", "component_id")
        .coalesce(100)  // Limit output files
        .write
        .mode("overwrite")
        .parquet(dedupedPath)
      logger.info(s"✓ Deduplicated vectors saved")
    }
  }
}
