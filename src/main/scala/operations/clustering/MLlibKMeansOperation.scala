package com.zilliz.spark.connector.operations.clustering

import org.apache.spark.ml.clustering.{KMeans => MLlibKMeans}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{ArrayType, FloatType}
import org.slf4j.{Logger, LoggerFactory}

/**
 * MLlib K-Means clustering operation wrapper
 *
 * This operation wraps Apache Spark MLlib's KMeans implementation to work with
 * Milvus Array[Float] format. It performs type conversions internally:
 * - Input: Array[Float] (Milvus format)
 * - Internal: Vector[Double] (MLlib format)
 * - Output: KMeansModel with Array[Float] centers
 *
 * Use this when you want MLlib's proven implementation with Double precision.
 * For native Float precision without conversions, use KMeansOperation instead.
 *
 * @param k Number of clusters
 * @param maxIter Maximum number of iterations
 * @param seed Random seed for initialization
 * @param featuresCol Name of the features column (must contain Array[Float])
 * @param predictionCol Name of the output prediction column (cluster_id)
 * @param initMode Initialization mode: "k-means||" (k-means++) or "random"
 * @param initSteps Number of steps for k-means|| initialization
 * @param tol Convergence tolerance
 */
class MLlibKMeansOperation(
  val k: Int,
  val maxIter: Int = 100,
  val seed: Long = 1L,
  val featuresCol: String = "features",
  val predictionCol: String = "cluster_id",
  val initMode: String = "k-means||",
  val initSteps: Int = 2,
  val tol: Double = 1e-4
) extends Serializable {

  private val logger: Logger = LoggerFactory.getLogger(classOf[MLlibKMeansOperation])

  require(k > 0, "Number of clusters must be positive")
  require(maxIter > 0, "Maximum iterations must be positive")
  require(featuresCol.nonEmpty, "Features column name cannot be empty")
  require(predictionCol.nonEmpty, "Prediction column name cannot be empty")
  require(initMode == "k-means||" || initMode == "random",
    s"initMode must be 'k-means||' or 'random', got: $initMode")

  /**
   * Train K-Means model using MLlib and return KMeansModel with Float centers
   */
  def fit(df: DataFrame): KMeansModel = {
    logger.info(s"Training MLlib K-Means: k=$k, maxIter=$maxIter, initMode=$initMode, tol=$tol")

    val spark = df.sparkSession
    val startTime = System.currentTimeMillis()

    // Validate input format
    val featuresType = df.schema(featuresCol).dataType
    require(featuresType == ArrayType(FloatType, false) || featuresType == ArrayType(FloatType, true),
      s"Input DataFrame must have Array[Float] column, but got $featuresType")

    val totalCount = df.count()
    logger.info(s"Total data points: $totalCount")

    // Step 1: Convert Array[Float] to Vector[Double] for MLlib
    logger.info("Converting Array[Float] to Vector[Double] for MLlib...")
    val conversionStart = System.currentTimeMillis()

    val toVector = udf((arr: Seq[Float]) => {
      Vectors.dense(arr.map(_.toDouble).toArray)
    })

    val vectorDF = df.select(col(featuresCol))
      .withColumn("__mllib_features", toVector(col(featuresCol)))
      .drop(featuresCol)  // 删掉原始 Float 列
      .cache()
    val conversionTime = System.currentTimeMillis() - conversionStart
    logger.info(s"Conversion completed in $conversionTime ms")

    // Step 2: Train MLlib KMeans
    logger.info("Training MLlib KMeans model...")
    val trainStart = System.currentTimeMillis()

    val mllibKMeans = new MLlibKMeans()
      .setK(k)
      .setMaxIter(maxIter)
      .setSeed(seed)
      .setInitMode(initMode)
      .setInitSteps(initSteps)
      .setTol(tol)
      .setFeaturesCol("__mllib_features")
      .setPredictionCol(predictionCol)

    val mllibModel = mllibKMeans.fit(vectorDF)
    val trainTime = System.currentTimeMillis() - trainStart
    logger.info(s"Training completed in $trainTime ms")

    // Step 3: Convert Vector[Double] centers back to Array[Float]
    logger.info("Converting cluster centers from Vector[Double] to Array[Float]...")
    val centers = mllibModel.clusterCenters.map { vector =>
      vector.toArray.map(_.toFloat)
    }

    vectorDF.unpersist()

    val totalTime = System.currentTimeMillis() - startTime

    // Log statistics
    logger.info("=" * 60)
    logger.info("MLlib K-Means Training Completed")
    logger.info("=" * 60)
    logger.info(s"Data conversion time: $conversionTime ms")
    logger.info(s"MLlib training time:  $trainTime ms")
    logger.info(s"Total training time:  $totalTime ms")
    logger.info(s"Number of clusters:   $k")
    logger.info(s"Cluster centers shape: ${centers.length} x ${centers(0).length}")
    logger.info("=" * 60)

    // Return custom KMeansModel with Float centers (unified API)
    new KMeansModel(centers, featuresCol, predictionCol)
  }
}

object MLlibKMeansOperation {

  def apply(k: Int, maxIter: Int = 100): MLlibKMeansOperation = {
    new MLlibKMeansOperation(k = k, maxIter = maxIter)
  }

  def apply(
    k: Int,
    maxIter: Int,
    seed: Long,
    featuresCol: String,
    predictionCol: String,
    initMode: String
  ): MLlibKMeansOperation = {
    new MLlibKMeansOperation(
      k = k,
      maxIter = maxIter,
      seed = seed,
      featuresCol = featuresCol,
      predictionCol = predictionCol,
      initMode = initMode
    )
  }
}
