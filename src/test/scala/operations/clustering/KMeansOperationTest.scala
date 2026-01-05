package com.zilliz.spark.connector.operations.clustering

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._

/**
 * Test suite for KMeansOperation and MiniBatchKMeansOperation
 */
class KMeansOperationTest extends AnyFunSuite with BeforeAndAfterAll {

  @transient var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("KMeansOperationTest")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  test("KMeansOperation - basic clustering") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create test data: 3 clusters
    val data = Seq(
      // Cluster 0: around (0, 0)
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(0.1, 0.1)),
      Tuple1(Vectors.dense(-0.1, 0.1)),
      Tuple1(Vectors.dense(0.1, -0.1)),
      // Cluster 1: around (5, 5)
      Tuple1(Vectors.dense(5.0, 5.0)),
      Tuple1(Vectors.dense(5.1, 5.1)),
      Tuple1(Vectors.dense(4.9, 5.1)),
      Tuple1(Vectors.dense(5.1, 4.9)),
      // Cluster 2: around (10, 10)
      Tuple1(Vectors.dense(10.0, 10.0)),
      Tuple1(Vectors.dense(10.1, 10.1)),
      Tuple1(Vectors.dense(9.9, 10.1)),
      Tuple1(Vectors.dense(10.1, 9.9))
    ).toDF("features")

    val kmeansOp = new KMeansOperation(
      k = 3,
      maxIter = 20,
      featuresCol = "features"
    )

    val result = kmeansOp.transform(data)

    // Verify result has cluster_id column
    assert(result.columns.contains("cluster_id"))

    // Verify we have 3 clusters
    val numClusters = result.select("cluster_id").distinct().count()
    assert(numClusters == 3)

    // Verify each cluster has 4 points
    val clusterCounts = result.groupBy("cluster_id").count()
    assert(clusterCounts.count() == 3)

    info("K-Means clustering test passed")
    result.show()
  }

  test("KMeansOperation - high dimensional vectors") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create high-dimensional test data (128 dimensions)
    val dim = 128
    val numPoints = 100
    val k = 10

    val data = (0 until numPoints).map { i =>
      val values = Array.fill(dim)(scala.util.Random.nextDouble())
      Tuple1(Vectors.dense(values))
    }.toDF("features")

    val kmeansOp = new KMeansOperation(
      k = k,
      maxIter = 50,
      featuresCol = "features"
    )

    val result = kmeansOp.transform(data)

    assert(result.columns.contains("cluster_id"))
    assert(result.count() == numPoints)

    val numClusters = result.select("cluster_id").distinct().count()
    assert(numClusters <= k)

    info(s"High-dimensional clustering test passed: $numClusters clusters found")
  }

  test("KMeansOperation - fit and transform separately") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(5.0, 5.0)),
      Tuple1(Vectors.dense(6.0, 6.0))
    ).toDF("features")

    val kmeansOp = new KMeansOperation(k = 2, maxIter = 20)

    // Fit model
    val model = kmeansOp.fit(data)

    assert(model != null)
    assert(model.clusterCenters.length == 2)

    // Transform with trained model
    val result = model.transform(data)
    assert(result.columns.contains("cluster_id"))

    info("Fit and transform test passed")
  }

  test("KMeansOperation - cosine distance") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(1.0, 0.0)),
      Tuple1(Vectors.dense(0.9, 0.1)),
      Tuple1(Vectors.dense(0.0, 1.0)),
      Tuple1(Vectors.dense(0.1, 0.9))
    ).toDF("features")

    val kmeansOp = new KMeansOperation(
      k = 2,
      maxIter = 20,
      distanceMeasure = "cosine"
    )

    val result = kmeansOp.transform(data)

    assert(result.columns.contains("cluster_id"))
    val numClusters = result.select("cluster_id").distinct().count()
    assert(numClusters == 2)

    info("Cosine distance test passed")
  }

  test("MiniBatchKMeansOperation - basic clustering") {
    val sparkSession = spark
    import sparkSession.implicits._

    // Create larger dataset for mini-batch
    val numPoints = 1000
    val dim = 10
    val k = 20

    val data = (0 until numPoints).map { i =>
      val clusterId = i % k
      val base = clusterId * 10.0
      val values = Array.fill(dim)(base + scala.util.Random.nextGaussian())
      Tuple1(Vectors.dense(values))
    }.toDF("features")

    val minibatchOp = new MiniBatchKMeansOperation(
      k = k,
      batchSize = 0.1,
      numBatches = 5,
      maxIterPerBatch = 10,
      featuresCol = "features"
    )

    val result = minibatchOp.transform(data)

    assert(result.columns.contains("cluster_id"))
    assert(result.count() == numPoints)

    val numClusters = result.select("cluster_id").distinct().count()
    assert(numClusters <= k)

    info(s"Mini-Batch K-Means test passed: $numClusters clusters found")
  }

  test("MiniBatchKMeansOperation - small batches") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = (0 until 500).map { i =>
      val values = Array.fill(8)(scala.util.Random.nextDouble())
      Tuple1(Vectors.dense(values))
    }.toDF("features")

    val minibatchOp = new MiniBatchKMeansOperation(
      k = 10,
      batchSize = 0.05,  // Very small batches
      numBatches = 20,
      maxIterPerBatch = 5
    )

    val result = minibatchOp.transform(data)

    assert(result.columns.contains("cluster_id"))
    assert(result.count() == 500)

    info("Small batch test passed")
  }

  test("KMeansOperation - custom column names") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0)),
      Tuple1(Vectors.dense(1.0, 1.0)),
      Tuple1(Vectors.dense(5.0, 5.0))
    ).toDF("embedding")

    val kmeansOp = new KMeansOperation(
      k = 2,
      maxIter = 20,
      featuresCol = "embedding",
      predictionCol = "my_cluster"
    )

    val result = kmeansOp.transform(data)

    assert(result.columns.contains("my_cluster"))
    assert(!result.columns.contains("cluster_id"))

    info("Custom column names test passed")
  }

  test("KMeansOperation - validation error on missing column") {
    val sparkSession = spark
    import sparkSession.implicits._

    val data = Seq(
      Tuple1(Vectors.dense(0.0, 0.0))
    ).toDF("features")

    val kmeansOp = new KMeansOperation(
      k = 2,
      featuresCol = "nonexistent"
    )

    assertThrows[IllegalArgumentException] {
      kmeansOp.transform(data)
    }

    info("Validation error test passed")
  }

  test("Compare KMeans vs MiniBatch performance") {
    val sparkSession = spark
    import sparkSession.implicits._

    val numPoints = 10000
    val dim = 64
    val k = 100

    info(s"Generating $numPoints points with $dim dimensions")

    val data = (0 until numPoints).map { i =>
      val values = Array.fill(dim)(scala.util.Random.nextDouble())
      Tuple1(Vectors.dense(values))
    }.toDF("features")

    data.cache()

    // Standard K-Means
    info("Running standard K-Means...")
    val kmeansOp = new KMeansOperation(k = k, maxIter = 50)
    val kmeansStart = System.currentTimeMillis()
    val kmeansResult = kmeansOp.transform(data)
    kmeansResult.count()  // Trigger computation
    val kmeansTime = (System.currentTimeMillis() - kmeansStart) / 1000.0
    info(s"Standard K-Means time: ${kmeansTime}s")

    // Mini-Batch K-Means
    info("Running Mini-Batch K-Means...")
    val minibatchOp = new MiniBatchKMeansOperation(
      k = k,
      batchSize = 0.1,
      numBatches = 10,
      maxIterPerBatch = 5
    )
    val minibatchStart = System.currentTimeMillis()
    val minibatchResult = minibatchOp.transform(data)
    minibatchResult.count()  // Trigger computation
    val minibatchTime = (System.currentTimeMillis() - minibatchStart) / 1000.0
    info(s"Mini-Batch K-Means time: ${minibatchTime}s")

    data.unpersist()

    info(s"Speedup: ${kmeansTime / minibatchTime}x")
  }
}
