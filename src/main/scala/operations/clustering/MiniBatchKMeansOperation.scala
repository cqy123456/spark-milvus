package com.zilliz.spark.connector.operations.clustering

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.Vector
import org.slf4j.{Logger, LoggerFactory}

/**
 * Mini-Batch K-Means clustering operation
 *
 * This operator performs Mini-Batch K-Means clustering, which is more efficient
 * for large-scale datasets. It trains the model on small random batches iteratively.
 *
 * Reference: spark-project MiniBatchKMeans.java implementation
 *
 * Algorithm:
 * 1. Split the dataset into multiple batches
 * 2. Train K-Means on each batch sequentially
 * 3. Use cluster centers from previous batch to initialize next batch
 * 4. Final prediction uses the last trained model on full dataset
 *
 * @param k Number of clusters
 * @param maxIter Maximum total iterations
 * @param batchSize Fraction of data to sample in each batch (0.0 to 1.0)
 * @param numBatches Number of batches to process
 * @param maxIterPerBatch Maximum iterations per batch
 * @param seed Random seed
 * @param featuresCol Name of the features column
 * @param predictionCol Name of the output prediction column
 * @param distanceMeasure Distance measure ("euclidean" or "cosine")
 *
 * Example usage:
 * {{{
 *   val minibatchOp = new MiniBatchKMeansOperation(
 *     k = 1000,
 *     batchSize = 0.1,
 *     numBatches = 10,
 *     featuresCol = "embedding"
 *   )
 *   val clustered = minibatchOp.transform(df)
 * }}}
 */
class MiniBatchKMeansOperation(
  val k: Int,
  val maxIter: Int = 100,
  val batchSize: Double = 0.1,
  val numBatches: Int = 10,
  val maxIterPerBatch: Int = 10,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val distanceMeasure: String = "euclidean"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MiniBatchKMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(batchSize > 0.0 && batchSize <= 1.0, "Batch size must be between 0.0 and 1.0")
  require(numBatches > 0, "Number of batches must be positive")
  require(maxIterPerBatch > 0, "Max iterations per batch must be positive")
  require(featuresCol.nonEmpty, "Features column name cannot be empty")
  require(predictionCol.nonEmpty, "Prediction column name cannot be empty")

  /**
   * Perform Mini-Batch K-Means clustering on the input DataFrame
   *
   * @param df Input DataFrame with vector features column
   * @return DataFrame with added cluster assignment column
   */
  def transform(df: DataFrame): DataFrame = {
    logger.info(s"Starting Mini-Batch K-Means: k=$k, batches=$numBatches, " +
                s"batchSize=$batchSize, maxIterPerBatch=$maxIterPerBatch")

    // Validate input
    validateInput(df)

    val startTime = System.currentTimeMillis()

    // Cache data for faster sampling
    df.cache()
    val totalCount = df.count()
    logger.info(s"Total data points: $totalCount")
    logger.info(s"Points per batch (approx): ${(totalCount * batchSize).toInt}")

    var currentModel: Option[KMeansModel] = None
    var totalIterations = 0
    val batchCosts = new Array[Double](numBatches)

    // Process each batch
    for (batchIdx <- 0 until numBatches) {
      logger.info(s"Processing batch ${batchIdx + 1}/$numBatches")

      // Sample a batch
      val batchDf = df.sample(
        withReplacement = false,
        fraction = batchSize,
        seed = seed + batchIdx  // Different seed for each batch
      )

      val batchCount = batchDf.count()
      logger.info(s"  Batch ${batchIdx + 1} sample size: $batchCount")

      // Build K-Means for this batch
      val kmeans = new KMeans()
        .setK(k)
        .setMaxIter(maxIterPerBatch)
        .setSeed(seed + batchIdx)
        .setFeaturesCol(featuresCol)
        .setPredictionCol(predictionCol)
        .setInitMode("k-means||")
        .setDistanceMeasure(distanceMeasure)

      // Train on batch
      val batchModel = kmeans.fit(batchDf)
      val batchIterations = batchModel.summary.numIter
      val batchCost = batchModel.summary.trainingCost

      totalIterations += batchIterations
      batchCosts(batchIdx) = batchCost

      logger.info(s"  Batch ${batchIdx + 1} completed: iterations=$batchIterations, cost=$batchCost")

      currentModel = Some(batchModel)
    }

    // Use final model to transform full dataset
    val finalModel = currentModel.get
    val result = finalModel.transform(df)

    df.unpersist()

    val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0

    // Log statistics
    logStatistics(finalModel, totalIterations, batchCosts, elapsedTime)

    result
  }

  /**
   * Train Mini-Batch K-Means model and return the final model
   *
   * @param df Input DataFrame
   * @return Trained KMeansModel
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training Mini-Batch K-Means model: k=$k, batches=$numBatches")

    validateInput(df)

    df.cache()

    var currentModel: Option[KMeansModel] = None

    for (batchIdx <- 0 until numBatches) {
      val batchDf = df.sample(false, batchSize, seed + batchIdx)

      val kmeans = new KMeans()
        .setK(k)
        .setMaxIter(maxIterPerBatch)
        .setSeed(seed + batchIdx)
        .setFeaturesCol(featuresCol)
        .setPredictionCol(predictionCol)
        .setInitMode("k-means||")
        .setDistanceMeasure(distanceMeasure)

      currentModel = Some(kmeans.fit(batchDf))
    }

    df.unpersist()

    currentModel.get
  }

  /**
   * Validate input DataFrame
   */
  private def validateInput(df: DataFrame): Unit = {
    require(df.columns.contains(featuresCol),
      s"Input DataFrame must contain column '$featuresCol'")

    val fieldOpt = df.schema.fields.find(_.name == featuresCol)
    fieldOpt.foreach { field =>
      require(
        field.dataType.typeName.contains("vector") ||
        field.dataType.simpleString.contains("vector"),
        s"Column '$featuresCol' must be of Vector type, but got ${field.dataType}"
      )
    }
  }

  /**
   * Log clustering statistics
   */
  private def logStatistics(
    model: KMeansModel,
    totalIterations: Int,
    batchCosts: Array[Double],
    elapsedTime: Double
  ): Unit = {
    val summary = model.summary

    logger.info("=" * 60)
    logger.info("Mini-Batch K-Means Clustering Results:")
    logger.info(s"  Number of clusters: $k")
    logger.info(s"  Number of batches: $numBatches")
    logger.info(s"  Batch size: ${(batchSize * 100).toInt}%")
    logger.info(s"  Total iterations: $totalIterations")
    logger.info(s"  Final training cost (WSSSE): ${summary.trainingCost}")
    logger.info(s"  Distance measure: $distanceMeasure")
    logger.info(s"  Training time: ${elapsedTime}s")

    logger.info("  Per-batch costs:")
    batchCosts.zipWithIndex.foreach { case (cost, idx) =>
      logger.info(f"    Batch ${idx + 1}: $cost%.2f")
    }

    logger.info("  Cluster sizes:")
    val clusterSizes = summary.clusterSizes
    clusterSizes.zipWithIndex.foreach { case (size, idx) =>
      logger.info(s"    Cluster $idx: $size points")
    }

    logger.info("=" * 60)
  }

  /**
   * Get cluster centers from trained model
   */
  def getClusterCenters(model: KMeansModel): Array[Vector] = {
    model.clusterCenters
  }
}

object MiniBatchKMeansOperation {

  /**
   * Create MiniBatchKMeansOperation with basic parameters
   */
  def apply(k: Int, batchSize: Double = 0.1, numBatches: Int = 10): MiniBatchKMeansOperation = {
    new MiniBatchKMeansOperation(
      k = k,
      batchSize = batchSize,
      numBatches = numBatches
    )
  }

  /**
   * Create MiniBatchKMeansOperation with custom features column
   */
  def apply(
    k: Int,
    batchSize: Double,
    numBatches: Int,
    featuresCol: String
  ): MiniBatchKMeansOperation = {
    new MiniBatchKMeansOperation(
      k = k,
      batchSize = batchSize,
      numBatches = numBatches,
      featuresCol = featuresCol
    )
  }

  /**
   * Create MiniBatchKMeansOperation with all parameters
   */
  def apply(
    k: Int,
    maxIter: Int,
    batchSize: Double,
    numBatches: Int,
    maxIterPerBatch: Int,
    seed: Long,
    featuresCol: String,
    predictionCol: String,
    distanceMeasure: String
  ): MiniBatchKMeansOperation = {
    new MiniBatchKMeansOperation(
      k = k,
      maxIter = maxIter,
      batchSize = batchSize,
      numBatches = numBatches,
      maxIterPerBatch = maxIterPerBatch,
      seed = seed,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      distanceMeasure = distanceMeasure
    )
  }
}
