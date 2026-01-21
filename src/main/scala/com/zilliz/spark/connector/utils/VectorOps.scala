package com.zilliz.spark.connector.utils

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.api.buffer.DataType

/**
 * Optimized vector operations using ND4J with Float32 precision
 *
 * ND4J provides high-performance linear algebra operations with:
 * - SIMD optimizations (AVX, AVX2, AVX-512)
 * - Native BLAS backend
 * - Batch operations for better cache utilization
 * - Float32 (single precision) for memory efficiency and speed
 */
object VectorOps {

  /**
   * Compute Euclidean (L2) distance between two vectors
   *
   * Formula: sqrt(sum((v1[i] - v2[i])^2))
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return L2 distance as Float
   */
  def euclideanDistance(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, s"Vector dimensions must match: ${v1.length} != ${v2.length}")

    // Convert to ND4J arrays with FLOAT (Float32) data type
    val nd1 = Nd4j.create(v1).castTo(DataType.FLOAT)
    val nd2 = Nd4j.create(v2).castTo(DataType.FLOAT)

    // Compute distance using ND4J's optimized implementation
    // Note: Transforms.euclideanDistance returns Double
    Transforms.euclideanDistance(nd1, nd2).toFloat
  }

  /**
   * Compute distance between two vectors using specified metric
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @param metric Distance metric: "l2" or "cosine"
   * @return Distance as Float
   */
  def computeDistance(v1: Array[Float], v2: Array[Float], metric: String): Float = {
    metric match {
      case "l2" => euclideanDistance(v1, v2)
      case "cosine" => cosineDistance(v1, v2)
      case _ => throw new IllegalArgumentException(s"Unknown distance metric: $metric")
    }
  }

  /**
   * Compute Cosine distance between two vectors
   *
   * Formula: 1 - (dot(v1, v2) / (||v1|| * ||v2||))
   * Range: [0, 2], where 0 = identical, 1 = orthogonal, 2 = opposite
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return Cosine distance as Float
   */
  def cosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, s"Vector dimensions must match: ${v1.length} != ${v2.length}")

    // Convert to ND4J arrays with FLOAT (Float32) data type
    val nd1 = Nd4j.create(v1).castTo(DataType.FLOAT)
    val nd2 = Nd4j.create(v2).castTo(DataType.FLOAT)

    // Compute cosine similarity using ND4J
    // Note: Transforms.cosineSim returns Double
    val cosineSimilarity = Transforms.cosineSim(nd1, nd2).toFloat

    // Convert to cosine distance
    1.0f - cosineSimilarity
  }

  /**
   * Compute dot product between two vectors
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return Dot product as Float
   */
  def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, s"Vector dimensions must match: ${v1.length} != ${v2.length}")

    val nd1 = Nd4j.create(v1).castTo(DataType.FLOAT)
    val nd2 = Nd4j.create(v2).castTo(DataType.FLOAT)

    nd1.mmul(nd2.transpose()).getFloat(0L)
  }

  /**
   * Compute L2 norm of a vector
   *
   * @param v Vector
   * @return L2 norm as Float
   */
  def norm2(v: Array[Float]): Float = {
    val nd = Nd4j.create(v).castTo(DataType.FLOAT)
    nd.norm2Number().floatValue()
  }

  /**
   * Compute pairwise squared Euclidean distances between queries and targets using ND4J
   * Returns distance matrix where result(i)(j) = ||queries(i) - targets(j)||²
   *
   * OPTIMIZED: Converts targets to ND4J matrix ONCE and reuses it for all query batches.
   * This avoids repeated conversion overhead when processing many queries against fixed targets (e.g., cluster centers).
   *
   * Uses formula: ||a - b||² = ||a||² + ||b||² - 2·a·b
   * Leverages BLAS matrix multiplication with SIMD acceleration (AVX, AVX2, AVX-512)
   *
   * @param queries Query vectors (n × dim)
   * @param targets Target vectors (m × dim) - converted to ND4J once and cached
   * @param queryBatchSize Process queries in batches of this size to limit memory (default 1000)
   * @return Distance matrix (n × m) of squared distances
   */
  def pairwiseSquaredDistances(
    queries: Array[Array[Float]],
    targets: Array[Array[Float]],
    queryBatchSize: Int = 1000
  ): Array[Array[Float]] = {
    if (queries.isEmpty || targets.isEmpty) return Array.empty[Array[Float]]

    val nQueries = queries.length
    val nTargets = targets.length
    val dim = queries(0).length

    require(queries.forall(_.length == dim), "All query vectors must have same dimension")
    require(targets.forall(_.length == dim), "All target vectors must have same dimension")

    // KEY OPTIMIZATION: Convert targets to ND4J ONCE and reuse for all query batches
    val targetsFlat = targets.flatten
    val targetsND = Nd4j.create(targetsFlat, Array(nTargets.toLong, dim.toLong), DataType.FLOAT)
    val targetsNormSq = targetsND.mul(targetsND).sum(true, 1)  // Compute once
    val targetsNDT = targetsND.transpose()  // Transpose once

    // Allocate result matrix
    val result = Array.ofDim[Float](nQueries, nTargets)

    // Process queries in batches to limit memory usage
    var batchStart = 0
    while (batchStart < nQueries) {
      val batchEnd = Math.min(batchStart + queryBatchSize, nQueries)
      val batchQueries = queries.slice(batchStart, batchEnd)
      val batchSize = batchQueries.length

      // Convert current query batch to ND4J
      val queriesFlat = batchQueries.flatten
      val queriesND = Nd4j.create(queriesFlat, Array(batchSize.toLong, dim.toLong), DataType.FLOAT)

      // Compute ||queries||² for this batch
      val queriesNormSq = queriesND.mul(queriesND).sum(true, 1)

      // Compute 2 * queries · targets^T using cached targetsNDT
      val dotProducts = queriesND.mmul(targetsNDT).mul(2.0f)

      // Compute ||a||² + ||b||² - 2·a·b
      val distancesSq = queriesNormSq.broadcast(batchSize, nTargets)
        .add(targetsNormSq.transpose().broadcast(batchSize, nTargets))
        .sub(dotProducts)

      // Copy to result array and clamp negative values (numerical errors)
      var i = 0
      while (i < batchSize) {
        var j = 0
        while (j < nTargets) {
          result(batchStart + i)(j) = Math.max(0.0f, distancesSq.getFloat(i.toLong, j.toLong))
          j += 1
        }
        i += 1
      }

      batchStart = batchEnd
    }

    result
  }

  /**
   * Find nearest target for each query vector using squared Euclidean distance
   * Returns only the nearest target index and distance for each query, not the full distance matrix.
   * This is memory-efficient and optimized for KMeans clustering where we only need the closest center.
   *
   * OPTIMIZED: Converts targets to ND4J matrix ONCE and reuses it for all query batches.
   * Uses formula: ||a - b||² = ||a||² + ||b||² - 2·a·b with BLAS matrix multiplication
   *
   * MEMORY OPTIMIZATION:
   * - Processes queries in batches to limit peak memory
   * - Computes distances row-by-row to avoid full distance matrix allocation
   * - Uses adaptive batch size based on number of targets to prevent OOM
   *
   * @param queries Query vectors (n × dim)
   * @param targets Target vectors (m × dim) - typically cluster centers
   * @param queryBatchSize Process queries in batches of this size to limit memory (default: auto-calculated)
   *                       Set to -1 for automatic batch size based on targets count
   * @return Array of (nearestIndex, squaredDistance) for each query
   */
  def findNearestTargets(
    queries: Array[Array[Float]],
    targets: Array[Array[Float]],
    queryBatchSize: Int = -1
  ): Array[(Int, Float)] = {
    if (queries.isEmpty || targets.isEmpty) return Array.empty[(Int, Float)]

    val nQueries = queries.length
    val nTargets = targets.length
    val dim = queries(0).length

    require(queries.forall(_.length == dim), "All query vectors must have same dimension")
    require(targets.forall(_.length == dim), "All target vectors must have same dimension")

    // ADAPTIVE BATCH SIZE: Limit memory for dotProducts matrix (batchSize × nTargets)
    // Target: ~10 MB per batch for dotProducts matrix
    // Memory = batchSize × nTargets × 4 bytes (Float)
    val effectiveBatchSize = if (queryBatchSize <= 0) {
      val targetMemoryMB = 10.0  // 10 MB target for dotProducts matrix
      val targetMemoryBytes = targetMemoryMB * 1024 * 1024
      val elementsPerBatch = targetMemoryBytes / 4.0  // 4 bytes per Float
      val calculatedBatchSize = Math.max(1, (elementsPerBatch / nTargets).toInt)
      Math.min(calculatedBatchSize, nQueries)
    } else {
      queryBatchSize
    }

    // KEY OPTIMIZATION: Convert targets to ND4J ONCE and reuse for all query batches
    val targetsFlat = targets.flatten
    val targetsND = Nd4j.create(targetsFlat, Array(nTargets.toLong, dim.toLong), DataType.FLOAT)
    val targetsNormSq = targetsND.mul(targetsND).sum(true, 1)  // Compute once
    val targetsNDT = targetsND.transpose()  // Transpose once

    // Allocate result array
    val result = Array.ofDim[(Int, Float)](nQueries)

    // Process queries in batches to limit memory usage
    var batchStart = 0
    while (batchStart < nQueries) {
      val batchEnd = Math.min(batchStart + effectiveBatchSize, nQueries)
      val batchQueries = queries.slice(batchStart, batchEnd)
      val batchSize = batchQueries.length

      // Convert current query batch to ND4J
      val queriesFlat = batchQueries.flatten
      val queriesND = Nd4j.create(queriesFlat, Array(batchSize.toLong, dim.toLong), DataType.FLOAT)

      // Compute ||queries||² for this batch
      val queriesNormSq = queriesND.mul(queriesND).sum(true, 1)

      // Compute 2 * queries · targets^T using cached targetsNDT
      val dotProducts = queriesND.mmul(targetsNDT).mul(2.0f)

      // MEMORY OPTIMIZATION: Compute distances and find minimum row-by-row
      // to avoid creating full distance matrix in memory
      var i = 0
      while (i < batchSize) {
        val queryNormSq = queriesNormSq.getFloat(i.toLong, 0L)
        var minDist = Float.MaxValue
        var minIdx = 0

        var j = 0
        while (j < nTargets) {
          // Compute ||a||² + ||b||² - 2·a·b for single element
          val dotProduct = dotProducts.getFloat(i.toLong, j.toLong)
          val targetNormSq = targetsNormSq.getFloat(j.toLong, 0L)
          val dist = Math.max(0.0f, queryNormSq + targetNormSq - dotProduct)

          if (dist < minDist) {
            minDist = dist
            minIdx = j
          }
          j += 1
        }
        result(batchStart + i) = (minIdx, minDist)
        i += 1
      }

      batchStart = batchEnd
    }

    result
  }

  /**
   * Batch compute Euclidean distances between a query vector and multiple target vectors
   * This is more efficient than computing distances one by one
   *
   * Optimized for KMeans with memory-efficient implementation:
   * - For small batches (k < 100): uses manual loop to avoid ND4J overhead
   * - For large batches (k >= 100): uses ND4J with SIMD for maximum performance
   * - Uses direct buffer creation to minimize object allocation
   *
   * @param query Query vector
   * @param targets Array of target vectors
   * @return Array of distances
   */
  def batchEuclideanDistance(query: Array[Float], targets: Array[Array[Float]]): Array[Float] = {
    if (targets.isEmpty) return Array.empty[Float]

    val dim = query.length
    require(targets.forall(_.length == dim), "All vectors must have the same dimension")

    // For small batches (typical in KMeans with k <= 10000), use manual computation
    // This avoids ND4J object creation overhead which causes OOM in iterative algorithms
    // Benchmark shows manual loop is faster for k < 10000 due to zero allocation overhead
    if (targets.length <= 10000) {
      val distances = new Array[Float](targets.length)

      var i = 0
      while (i < targets.length) {
        val target = targets(i)
        var sumSq = 0.0f

        var j = 0
        while (j < dim) {
          val diff = query(j) - target(j)
          sumSq += diff * diff
          j += 1
        }

        distances(i) = Math.sqrt(sumSq).toFloat
        i += 1
      }

      distances
    } else {
      // For large batches, ND4J's SIMD optimization is worth the overhead
      // Create ND4J matrices with proper API
      val queryND = Nd4j.create(query).castTo(DataType.FLOAT).reshape(1, dim)

      // Create targets matrix row by row
      val targetsND = Nd4j.create(DataType.FLOAT, targets.length.toLong, dim.toLong)
      var i = 0
      while (i < targets.length) {
        targetsND.putRow(i, Nd4j.create(targets(i)).castTo(DataType.FLOAT))
        i += 1
      }

      // Compute pairwise distances efficiently with SIMD
      val diff = targetsND.subRowVector(queryND.getRow(0))
      val squared = Transforms.pow(diff, 2)
      val summed = squared.sum(true, 1)  // Sum along dimension axis
      val distances = Transforms.sqrt(summed)

      distances.toFloatVector
    }
  }

  /**
   * Batch compute Cosine distances between a query vector and multiple target vectors
   *
   * @param query Query vector
   * @param targets Array of target vectors
   * @return Array of distances
   */
  def batchCosineDistance(query: Array[Float], targets: Array[Array[Float]]): Array[Float] = {
    if (targets.isEmpty) return Array.empty[Float]

    val dim = query.length
    require(targets.forall(_.length == dim), "All vectors must have the same dimension")

    // Use manual computation to avoid ND4J overhead
    val distances = new Array[Float](targets.length)

    // Compute query norm once
    var queryNormSq = 0.0f
    var j = 0
    while (j < dim) {
      queryNormSq += query(j) * query(j)
      j += 1
    }
    val queryNorm = Math.sqrt(queryNormSq).toFloat

    var i = 0
    while (i < targets.length) {
      val target = targets(i)

      // Compute dot product and target norm
      var dotProduct = 0.0f
      var targetNormSq = 0.0f

      var k = 0
      while (k < dim) {
        dotProduct += query(k) * target(k)
        targetNormSq += target(k) * target(k)
        k += 1
      }

      val targetNorm = Math.sqrt(targetNormSq).toFloat

      // Cosine similarity = dot / (norm1 * norm2)
      val cosineSim = if (queryNorm > 0 && targetNorm > 0) {
        dotProduct / (queryNorm * targetNorm)
      } else {
        0.0f
      }

      // Cosine distance = 1 - similarity
      distances(i) = 1.0f - cosineSim
      i += 1
    }

    distances
  }
}
