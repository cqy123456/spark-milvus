package com.zilliz.spark.connector.operations.clustering

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.Vector
import org.slf4j.{Logger, LoggerFactory}

/**
 * Standard K-Means clustering operation
 *
 * This operator performs K-Means clustering on a DataFrame with vector features.
 * It's a standalone operator that can be used independently or combined with other operators.
 *
 * Reference: spark-project KMeans.java implementation
 *
 * @param k Number of clusters
 * @param maxIter Maximum number of iterations
 * @param seed Random seed for initialization
 * @param featuresCol Name of the features column (must contain Vector type)
 * @param predictionCol Name of the output prediction column (cluster_id)
 * @param distanceMeasure Distance measure to use ("euclidean" or "cosine")
 *
 * Example usage:
 * {{{
 *   val kmeansOp = new KMeansOperation(
 *     k = 100,
 *     maxIter = 50,
 *     featuresCol = "embedding"
 *   )
 *   val clustered = kmeansOp.transform(df)
 * }}}
 */
class KMeansOperation(
  val k: Int,
  val maxIter: Int = 100,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val distanceMeasure: String = "euclidean"
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[KMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(maxIter > 0, "Maximum iterations must be positive")
  require(featuresCol.nonEmpty, "Features column name cannot be empty")
  require(predictionCol.nonEmpty, "Prediction column name cannot be empty")

  /**
   * Perform K-Means clustering on the input DataFrame
   *
   * @param df Input DataFrame with vector features column
   * @return DataFrame with added cluster assignment column
   */
  def transform(df: DataFrame): DataFrame = {
    logger.info(s"Starting K-Means clustering: k=$k, maxIter=$maxIter, distanceMeasure=$distanceMeasure")

    // Validate input
    validateInput(df)

    val startTime = System.currentTimeMillis()

    // Build K-Means model
    val kmeans = new KMeans()
      .setK(k)
      .setMaxIter(maxIter)
      .setSeed(seed)
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)
      .setInitMode("k-means||")  // k-means++ initialization
      .setDistanceMeasure(distanceMeasure)

    // Train model
    val model = kmeans.fit(df)

    // Transform data
    val result = model.transform(df)

    val elapsedTime = (System.currentTimeMillis() - startTime) / 1000.0

    // Log statistics
    logStatistics(model, elapsedTime)

    result
  }

  /**
   * Train K-Means model and return the model
   *
   * @param df Input DataFrame
   * @return Trained KMeansModel
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training K-Means model: k=$k, maxIter=$maxIter")

    validateInput(df)

    val kmeans = new KMeans()
      .setK(k)
      .setMaxIter(maxIter)
      .setSeed(seed)
      .setFeaturesCol(featuresCol)
      .setPredictionCol(predictionCol)
      .setInitMode("k-means||")
      .setDistanceMeasure(distanceMeasure)

    val model = kmeans.fit(df)

    logStatistics(model, 0.0)

    model
  }

  /**
   * Validate input DataFrame
   */
  private def validateInput(df: DataFrame): Unit = {
    require(df.columns.contains(featuresCol),
      s"Input DataFrame must contain column '$featuresCol'")

    // Check if features column is Vector type
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
  private def logStatistics(model: KMeansModel, elapsedTime: Double): Unit = {
    val summary = model.summary

    logger.info("=" * 60)
    logger.info("K-Means Clustering Results:")
    logger.info(s"  Number of clusters: $k")
    logger.info(s"  Iterations completed: ${summary.numIter}")
    logger.info(s"  Training cost (WSSSE): ${summary.trainingCost}")
    logger.info(s"  Distance measure: $distanceMeasure")
    if (elapsedTime > 0) {
      logger.info(s"  Training time: ${elapsedTime}s")
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

object KMeansOperation {

  /**
   * Create KMeansOperation with required parameters
   */
  def apply(k: Int, maxIter: Int = 100): KMeansOperation = {
    new KMeansOperation(k = k, maxIter = maxIter)
  }

  /**
   * Create KMeansOperation with full configuration
   */
  def apply(
    k: Int,
    maxIter: Int,
    featuresCol: String
  ): KMeansOperation = {
    new KMeansOperation(
      k = k,
      maxIter = maxIter,
      featuresCol = featuresCol
    )
  }

  /**
   * Create KMeansOperation with all parameters
   */
  def apply(
    k: Int,
    maxIter: Int,
    seed: Long,
    featuresCol: String,
    predictionCol: String,
    distanceMeasure: String
  ): KMeansOperation = {
    new KMeansOperation(
      k = k,
      maxIter = maxIter,
      seed = seed,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      distanceMeasure = distanceMeasure
    )
  }
}
