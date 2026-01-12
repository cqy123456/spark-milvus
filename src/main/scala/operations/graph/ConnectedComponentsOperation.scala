package com.zilliz.spark.connector.operations.graph

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.graphframes.GraphFrame
import org.slf4j.{Logger, LoggerFactory}

/**
 * Connected Components Operation
 *
 * Provides two algorithms for finding connected components:
 * 1. Block-level Union-Find (local=true): Custom implementation using partition-local Union-Find
 * 2. LargeStar algorithm (local=false): GraphFrames distributed algorithm
 *
 * @param local If true, use custom block-level Union-Find. If false, use GraphFrames LargeStar.
 * @param maxIterations Maximum iterations for the custom algorithm
 * @param checkpointInterval Checkpoint interval to break lineage
 */
class ConnectedComponentsOperation(
  val local: Boolean = true,
  val maxIterations: Int = 20,
  val checkpointInterval: Int = 5
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[ConnectedComponentsOperation])

  require(maxIterations > 0, "Maximum iterations must be positive")
  require(checkpointInterval >= 0, "Checkpoint interval must be non-negative")

  /**
   * Main entry point - chooses algorithm based on local parameter
   */
  def findComponents(edgesDF: DataFrame): DataFrame = {
    if (local) {
      findComponentsBlockUnionFind(edgesDF)
    } else {
      findComponentsLargeStar(edgesDF)
    }
  }

  /**
   * Block-level Union-Find Algorithm (Custom Implementation)
   *
   * Good for: Dense graphs, local computation
   * Uses iterative label propagation with min component ID
   */
  def findComponentsBlockUnionFind(edgesDF: DataFrame): DataFrame = {
    logger.info("=" * 80)
    logger.info("Connected Components (Block-level Union-Find)")
    logger.info("=" * 80)

    val startTime = System.currentTimeMillis()
    val spark = edgesDF.sparkSession
    import spark.implicits._

    // Cache edges for multiple passes
    val edges = edgesDF.cache()
    val edgeCount = edges.count()
    logger.info(s"Processing $edgeCount edges...")

    // Initialize: each vertex is its own component
    var components = edges.select($"src_id".as("id"))
      .union(edges.select($"dst_id".as("id")))
      .distinct()
      .withColumn("component_id", $"id")
      .cache()

    val vertexCount = components.count()
    logger.info(s"Total vertices: $vertexCount")

    var iteration = 0
    var converged = false

    while (!converged && iteration < maxIterations) {
      iteration += 1
      logger.info(s"Iteration $iteration...")

      // Join edges with current component assignments
      val srcComponents = components
        .withColumnRenamed("id", "src_id")
        .withColumnRenamed("component_id", "src_comp")

      val dstComponents = components
        .withColumnRenamed("id", "dst_id")
        .withColumnRenamed("component_id", "dst_comp")

      // Propagate minimum component ID
      val newComponents = edges
        .join(srcComponents, "src_id")
        .join(dstComponents, "dst_id")
        .select(
          $"src_id".as("id"),
          least($"src_comp", $"dst_comp").as("new_component_id")
        )
        .union(
          edges
            .join(srcComponents, "src_id")
            .join(dstComponents, "dst_id")
            .select(
              $"dst_id".as("id"),
              least($"src_comp", $"dst_comp").as("new_component_id")
            )
        )
        .groupBy("id")
        .agg(min("new_component_id").as("component_id"))
        .cache()

      // Check for convergence
      val changes = components.alias("old")
        .join(newComponents.alias("new"), "id")
        .filter($"old.component_id" =!= $"new.component_id")
        .count()

      logger.info(s"  Changes: $changes")

      if (changes == 0) {
        converged = true
        logger.info(s"✓ Converged after $iteration iterations")
      } else {
        components.unpersist()
        components = newComponents
      }

      // Checkpoint to break lineage
      if (checkpointInterval > 0 && iteration % checkpointInterval == 0) {
        components = components.checkpoint()
        logger.info(s"  Checkpoint at iteration $iteration")
      }
    }

    if (!converged) {
      logger.warn(s"Did not converge after $maxIterations iterations")
    }

    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Connected components computed in $totalTime%.2f seconds")

    // Cleanup
    edges.unpersist()

    // Log statistics
    logComponentStatistics(components)

    components
  }

  /**
   * LargeStar Algorithm using GraphFrames
   *
   * Good for: Sparse graphs, distributed computation, large datasets
   * Uses GraphFrames library which implements "Connected Components in MapReduce and Beyond"
   */
  def findComponentsLargeStar(edgesDF: DataFrame): DataFrame = {
    logger.info("=" * 80)
    logger.info("Connected Components (LargeStar - GraphFrames)")
    logger.info("=" * 80)

    val startTime = System.currentTimeMillis()
    val spark = edgesDF.sparkSession
    import spark.implicits._

    // Prepare edges: rename columns to match GraphFrame requirements
    val edges = edgesDF
      .select($"src_id".as("src"), $"dst_id".as("dst"))
      .cache()

    val edgeCount = edges.count()
    logger.info(s"Processing $edgeCount edges...")

    // Extract vertices from edges
    val vertices = edges.select($"src".as("id"))
      .union(edges.select($"dst".as("id")))
      .distinct()
      .cache()

    val vertexCount = vertices.count()
    logger.info(s"Total vertices: $vertexCount")

    // Create GraphFrame
    val graph = GraphFrame(vertices, edges)

    // Use GraphFrames algorithm (distributed LargeStar)
    logger.info("Using GraphFrames algorithm (distributed LargeStar)...")

    // Set checkpoint directory if not already set
    val checkpointDir = spark.sparkContext.getCheckpointDir
    if (checkpointDir.isEmpty) {
      val tmpDir = System.getProperty("java.io.tmpdir")
      val dir = s"$tmpDir/spark-checkpoints-${System.currentTimeMillis()}"
      logger.info(s"Setting checkpoint directory: $dir")
      spark.sparkContext.setCheckpointDir(dir)
    }

    val result = graph.connectedComponents
      .setAlgorithm("graphframes")
      .setCheckpointInterval(checkpointInterval)
      .run()

    // Rename component column to match expected output
    val components = result
      .select($"id", $"component".as("component_id"))
      .cache()

    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Connected components computed in $totalTime%.2f seconds")

    // Cleanup
    edges.unpersist()
    vertices.unpersist()

    // Log statistics
    logComponentStatistics(components)

    components
  }

  /**
   * Log component statistics
   */
  private def logComponentStatistics(components: DataFrame): Unit = {
    import components.sparkSession.implicits._

    val numComponents = components.select("component_id").distinct().count()
    logger.info(s"Total connected components: $numComponents")

    // Component size distribution
    val sizeDistribution = components
      .groupBy("component_id")
      .agg(count("*").as("size"))
      .groupBy("size")
      .agg(count("*").as("count"))
      .orderBy("size")
      .collect()

    logger.info("Component size distribution:")
    sizeDistribution.take(10).foreach { row =>
      val size = row.getAs[Long]("size")
      val count = row.getAs[Long]("count")
      logger.info(f"  Size $size%5d: $count%6d components")
    }

    if (sizeDistribution.length > 10) {
      logger.info(s"  ... (showing first 10 of ${sizeDistribution.length} size groups)")
    }
  }
}

/**
 * Factory object for creating ConnectedComponentsOperation instances
 */
object ConnectedComponentsOperation {

  /**
   * Create a ConnectedComponentsOperation with default settings (local Union-Find)
   */
  def apply(): ConnectedComponentsOperation = {
    new ConnectedComponentsOperation(local = true)
  }

  /**
   * Create a ConnectedComponentsOperation with custom settings
   *
   * @param local If true, use custom block-level Union-Find. If false, use GraphFrames LargeStar.
   * @param maxIterations Maximum iterations for custom algorithm
   * @param checkpointInterval Checkpoint interval for breaking lineage
   */
  def apply(local: Boolean, maxIterations: Int = 20, checkpointInterval: Int = 5): ConnectedComponentsOperation = {
    new ConnectedComponentsOperation(local, maxIterations, checkpointInterval)
  }
}
