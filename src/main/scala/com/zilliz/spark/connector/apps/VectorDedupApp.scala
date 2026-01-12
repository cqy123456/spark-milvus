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
 * Vector Deduplication Application
 *
 * Pipeline:
 * 1. KMeans Clustering: Cluster vectors into k groups
 * 2. Repartition by Cluster: Move vectors to executors by cluster_id
 * 3. Sort by Distance: Within each cluster, sort by distance to center
 * 4. Build Threshold Graph: Connect similar vectors (distance < threshold)
 * 5. Find Connected Components: Identify duplicate groups
 * 6. Deduplicate: Keep one representative from each group
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
  def deduplicate(df: DataFrame): DataFrame = {
    logger.info("=" * 80)
    logger.info("Starting Vector Deduplication Pipeline")
    logger.info("=" * 80)
    logger.info(s"Configuration: k=$k, distanceThreshold=$distanceThreshold, maxIterKMeans=$maxIterKMeans")
    logger.info(s"Distance Metric: $distanceMetric (for threshold graph)")
    logger.info(s"Connected Components Algorithm: ${if (local) "Block-level Union-Find (local)" else "LargeStar (distributed)"}")
    cacheDir.foreach(dir => logger.info(s"Cache directory: $dir"))
    outputDir.foreach(dir => logger.info(s"Output directory: $dir"))

    val spark = df.sparkSession

    // Validate input
    validateInput(df)

    val totalVectors = df.count()
    logger.info(s"Total input vectors: $totalVectors")

    // Step 1: KMeans Clustering
    val clusteredDF = performKMeansClustering(df)
    saveKMeansResults(clusteredDF)

    // Step 2: Repartition by Cluster and Sort by Distance
    val sortedDF = repartitionAndSortByClusters(clusteredDF)

    // Step 3: Build Threshold Graph
    val graphDF = buildThresholdGraph(sortedDF)
    saveGraphEdges(graphDF)

    // Step 4: Find Connected Components
    val componentAssignments = findConnectedComponents(graphDF)

    // Step 5: Deduplicate by Components
    // Join component assignments back to the sorted vectors
    val sortedWithComponents = sortedDF.join(componentAssignments, Seq("id"), "left")
      .na.fill(Map("component_id" -> -1L))  // Vectors with no edges get their own component

    val dedupedDF = deduplicateByComponents(sortedWithComponents)

    // Save final results
    saveFinalResults(dedupedDF)

    logger.info("=" * 80)
    logger.info("Vector Deduplication Pipeline Completed")
    logger.info("=" * 80)

    dedupedDF
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

    // Transform: add cluster_id and distance columns
    logger.info("Assigning cluster IDs and computing distances...")
    val clusteredDF = model.transform(df)

    val clusteringTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ KMeans clustering completed in $clusteringTime%.2f seconds")

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

    // Repartition by cluster_id using RDD API for precise control
    logger.info(s"Repartitioning into $clusterCount partitions by cluster_id...")

    import spark.implicits._

    // Get the column indices for cluster_id and distance
    val clusterIdIdx = df.schema.fieldIndex("cluster_id")
    val distanceIdx = df.schema.fieldIndex("distance")

    // Convert to RDD for repartitioning and sorting
    val repartitionedRDD = df.rdd
      .map { row =>
        val clusterId = row.getInt(clusterIdIdx)
        (clusterId, row)
      }
      .partitionBy(new HashPartitioner(clusterCount))
      .mapPartitions { partition =>
        // Sort within each partition by distance (ascending - closest to center first)
        partition.toSeq.sortBy { case (clusterId, row) =>
          row.getFloat(distanceIdx)
        }.map(_._2).iterator
      }

    // Convert back to DataFrame
    val sortedDF = spark.createDataFrame(repartitionedRDD, df.schema)

    val repartitionTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Repartitioning and sorting completed in $repartitionTime%.2f seconds")

    // Verify partitioning
    val finalPartitions = sortedDF.rdd.getNumPartitions
    logger.info(s"Final number of partitions: $finalPartitions")

    sortedDF
  }

  /**
   * Step 3: Build Threshold Graph within each Executor
   *
   * For each cluster partition (on each executor):
   * 1. Vectors are already sorted by distance to center
   * 2. Compare each vector with subsequent vectors
   * 3. If euclidean distance < threshold: add edge
   * 4. Optimize: early stop when distance to center diff > threshold
   *
   * Input:  | id: Long | features: Array[Float] | cluster_id: Int | distance: Float |
   *
   * Output: | src_id: Long | dst_id: Long | edge_distance: Float |
   *         (edges between similar vectors within same cluster)
   */
  private def buildThresholdGraph(df: DataFrame): DataFrame = {
    logger.info("Step 3: Building threshold graph within each executor...")
    val startTime = System.currentTimeMillis()

    val spark = df.sparkSession
    val threshold = distanceThreshold

    // Build graph edges within each partition (cluster)
    // Each partition contains vectors from ONE cluster, sorted by distance to center
    val edgesRDD = df.rdd.mapPartitions { partition =>
      val vectors = partition.toArray
      val edges = scala.collection.mutable.ArrayBuffer[(Long, Long, Float)]()

      // Compare each vector with subsequent vectors in the same partition
      for (i <- vectors.indices) {
        val vi = vectors(i)
        val id_i = vi.getAs[Long](idCol)
        val features_i = vi.getAs[scala.collection.mutable.WrappedArray[Float]](featuresCol).toArray
        val dist_i = vi.getAs[Float]("distance") // distance to cluster center

        // Compare with subsequent vectors
        var j = i + 1
        var continue = true

        while (j < vectors.length && continue) {
          val vj = vectors(j)
          val id_j = vj.getAs[Long](idCol)
          val features_j = vj.getAs[scala.collection.mutable.WrappedArray[Float]](featuresCol).toArray
          val dist_j = vj.getAs[Float]("distance")

          // Triangle inequality optimization for early pruning:
          // Upper bound: definitely duplicate (can add edge without computing distance)
          // Lower bound: definitely not duplicate (can skip computation)

          val (definitelyDuplicate, definitelyNotDuplicate) = distanceMetric match {
            case "l2" =>
              // Upper bound: ||v_i - v_j|| <= d_i + d_j
              // If d_i + d_j < threshold, then definitely duplicate
              val upperBound = dist_i + dist_j < threshold

              // Lower bound: ||v_i - v_j|| >= |d_i - d_j|
              // If |d_i - d_j| > threshold, then definitely not duplicate
              val lowerBound = math.abs(dist_j - dist_i) > threshold

              (upperBound, lowerBound)

            case "cosine" =>
              // For cosine distance, we only have reliable lower bound
              // Upper bound optimization is less straightforward for cosine
              val upperBound = false  // Conservative: don't use upper bound for cosine

              // Lower bound: dist(v_i, v_j) >= |d_i - d_j|
              // If |d_i - d_j| > (1 - threshold), then similarity < threshold → not duplicate
              val lowerBound = math.abs(dist_j - dist_i) > (1.0f - threshold)

              (upperBound, lowerBound)

            case _ => (false, false)
          }

          if (definitelyDuplicate) {
            // Triangle inequality guarantees they are duplicates
            // Add edge without computing actual distance (use upper bound as estimate)
            val estimatedDist = dist_i + dist_j
            edges.append((id_i, id_j, estimatedDist))
            edges.append((id_j, id_i, estimatedDist))
          } else if (definitelyNotDuplicate) {
            // Triangle inequality guarantees they are not duplicates
            // Skip computation and stop checking further vectors (early stopping)
            continue = false
          } else {
            // Need to compute actual distance
            val dist = computeDistance(features_i, features_j)

            if (dist < threshold) {
              // Add bidirectional edge (for undirected graph)
              edges.append((id_i, id_j, dist))
              edges.append((id_j, id_i, dist))
            }
          }

          j += 1
        }
      }

      edges.iterator
    }

    // Convert to DataFrame
    import spark.implicits._
    val edgesDF = edgesRDD.toDF("src_id", "dst_id", "edge_distance")

    val graphTime = (System.currentTimeMillis() - startTime) / 1000.0
    val edgeCount = edgesDF.count()

    logger.info(f"✓ Threshold graph built in $graphTime%.2f seconds")
    logger.info(s"Total edges: $edgeCount")

    edgesDF
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
      val duplicatesPath = s"$dir/duplicates"
      logger.info(s"Saving duplicates to: $duplicatesPath")
      deduplicatedDF
        .filter(!$"is_representative")
        .select(idCol, "component_id")
        .write
        .mode("overwrite")
        .parquet(duplicatesPath)
      logger.info(s"✓ Duplicates saved")

      // Save deduplicated vectors (is_representative = true)
      val dedupedPath = s"$dir/deduplicated"
      logger.info(s"Saving deduplicated vectors to: $dedupedPath")
      deduplicatedDF
        .filter($"is_representative")
        .write
        .mode("overwrite")
        .parquet(dedupedPath)
      logger.info(s"✓ Deduplicated vectors saved")
    }
  }
}
