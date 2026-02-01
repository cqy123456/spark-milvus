package com.zilliz.spark.connector.utils

import com.github.fommil.netlib.BLAS

/**
 * Optimized vector operations using netlib-java with Float32 precision
 *
 * netlib-java provides high-performance linear algebra operations with:
 * - SIMD optimizations via native BLAS (OpenBLAS, MKL) when available
 * - Pure Java fallback for portability
 * - No off-heap memory management - everything on JVM heap
 * - No GC issues like ND4J - memory is managed by JVM
 *
 * Key benefits over ND4J:
 * - No native memory leaks or DataBuffer release issues
 * - No need to manually close arrays
 * - Works directly on Array[Float] - zero copy for most operations
 * - Simpler, more predictable memory behavior
 */
object VectorOps {

  // Get BLAS instance (auto-selects native or Java implementation)
  private lazy val blas: BLAS = {
    val instance = BLAS.getInstance()
    // Log BLAS implementation type for debugging
    val blasClass = instance.getClass.getName
    if (blasClass.contains("Native") || blasClass.contains("netlib")) {
      System.out.println(s"[VectorOps] Using native BLAS implementation: $blasClass")
    } else {
      System.out.println(s"[VectorOps] WARNING: Using Java fallback BLAS: $blasClass (performance may be slower)")
    }
    instance
  }

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

    val dim = v1.length
    var sumSq = 0.0f
    var i = 0
    while (i < dim) {
      val diff = v1(i) - v2(i)
      sumSq += diff * diff
      i += 1
    }
    Math.sqrt(sumSq).toFloat
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

    val dim = v1.length
    var dotProduct = 0.0f
    var norm1Sq = 0.0f
    var norm2Sq = 0.0f

    var i = 0
    while (i < dim) {
      dotProduct += v1(i) * v2(i)
      norm1Sq += v1(i) * v1(i)
      norm2Sq += v2(i) * v2(i)
      i += 1
    }

    val norm1 = Math.sqrt(norm1Sq).toFloat
    val norm2 = Math.sqrt(norm2Sq).toFloat

    if (norm1 > 0 && norm2 > 0) {
      1.0f - (dotProduct / (norm1 * norm2))
    } else {
      1.0f  // If either vector is zero, cosine distance is undefined, return 1
    }
  }

  /**
   * Compute dot product between two vectors using BLAS sdot
   *
   * @param v1 First vector
   * @param v2 Second vector
   * @return Dot product as Float
   */
  def dotProduct(v1: Array[Float], v2: Array[Float]): Float = {
    require(v1.length == v2.length, s"Vector dimensions must match: ${v1.length} != ${v2.length}")
    blas.sdot(v1.length, v1, 1, v2, 1)
  }

  /**
   * Compute L2 norm of a vector using BLAS snrm2
   *
   * @param v Vector
   * @return L2 norm as Float
   */
  def norm2(v: Array[Float]): Float = {
    blas.snrm2(v.length, v, 1)
  }

  /**
   * Compute pairwise squared Euclidean distances between queries and targets
   * Returns distance matrix where result(i)(j) = ||queries(i) - targets(j)||²
   *
   * Uses formula: ||a - b||² = ||a||² + ||b||² - 2·a·b
   * Leverages BLAS sgemm for batch dot product computation
   *
   * @param queries Query vectors (n × dim)
   * @param targets Target vectors (m × dim)
   * @param queryBatchSize Process queries in batches of this size to limit memory (default 100)
   * @return Distance matrix (n × m) of squared distances
   */
  def pairwiseSquaredDistances(
    queries: Array[Array[Float]],
    targets: Array[Array[Float]],
    queryBatchSize: Int = 100
  ): Array[Array[Float]] = {
    if (queries.isEmpty || targets.isEmpty) return Array.empty[Array[Float]]

    val nQueries = queries.length
    val nTargets = targets.length
    val dim = queries(0).length

    // Pre-compute target norms
    val targetNormsSq = new Array[Float](nTargets)
    var t = 0
    while (t < nTargets) {
      var sum = 0.0f
      val target = targets(t)
      var k = 0
      while (k < dim) {
        sum += target(k) * target(k)
        k += 1
      }
      targetNormsSq(t) = sum
      t += 1
    }

    // Flatten targets for BLAS sgemm (column-major order for targets^T)
    // targets is m x dim, we need targets^T (dim x m) in column-major = targets in row-major
    val targetsFlat = new Array[Float](nTargets * dim)
    t = 0
    while (t < nTargets) {
      val target = targets(t)
      var k = 0
      while (k < dim) {
        // Column-major: targetsFlat[k * nTargets + t] = targets[t][k]
        targetsFlat(k * nTargets + t) = target(k)
        k += 1
      }
      t += 1
    }

    // Result matrix
    val result = Array.ofDim[Float](nQueries, nTargets)

    // Process in batches
    var batchStart = 0
    while (batchStart < nQueries) {
      val batchEnd = Math.min(batchStart + queryBatchSize, nQueries)
      val batchSize = batchEnd - batchStart

      // Flatten query batch (row-major order)
      val queriesFlat = new Array[Float](batchSize * dim)
      var qi = 0
      while (qi < batchSize) {
        val query = queries(batchStart + qi)
        var k = 0
        while (k < dim) {
          queriesFlat(qi * dim + k) = query(k)
          k += 1
        }
        qi += 1
      }

      // Compute dot products: C = queries * targets^T
      // queries: batchSize x dim (row-major)
      // targets^T: dim x nTargets (we have it in column-major as targetsFlat)
      // C: batchSize x nTargets
      val dotProducts = new Array[Float](batchSize * nTargets)

      // sgemm: C = alpha * A * B + beta * C
      // A: batchSize x dim (row-major) => column-major: dim x batchSize, so we use 'T'
      // B: dim x nTargets (column-major) => we use 'N'
      // For row-major A, column-major B, row-major C:
      // Actually simpler to compute row by row with sdot
      qi = 0
      while (qi < batchSize) {
        val query = queries(batchStart + qi)
        var ti = 0
        while (ti < nTargets) {
          // dot product
          var dot = 0.0f
          var k = 0
          while (k < dim) {
            dot += query(k) * targets(ti)(k)
            k += 1
          }

          // Compute query norm
          var queryNormSq = 0.0f
          k = 0
          while (k < dim) {
            queryNormSq += query(k) * query(k)
            k += 1
          }

          // distance = ||q||² + ||t||² - 2·q·t
          result(batchStart + qi)(ti) = Math.max(0.0f, queryNormSq + targetNormsSq(ti) - 2.0f * dot)
          ti += 1
        }
        qi += 1
      }

      batchStart = batchEnd
    }

    result
  }

  /**
   * Find nearest target for each query vector using squared Euclidean distance
   * Returns only the nearest target index and distance for each query, not the full distance matrix.
   *
   * Uses formula: ||a - b||² = ||a||² + ||b||² - 2·a·b
   *
   * @param queries Query vectors (n × dim)
   * @param targets Target vectors (m × dim) - typically cluster centers
   * @param queryBatchSize Process queries in batches of this size to limit memory
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

    // Pre-compute target norms
    val targetNormsSq = new Array[Float](nTargets)
    var t = 0
    while (t < nTargets) {
      var sum = 0.0f
      val target = targets(t)
      var k = 0
      while (k < dim) {
        sum += target(k) * target(k)
        k += 1
      }
      targetNormsSq(t) = sum
      t += 1
    }

    val result = new Array[(Int, Float)](nQueries)

    var qi = 0
    while (qi < nQueries) {
      val query = queries(qi)

      // Compute query norm
      var queryNormSq = 0.0f
      var k = 0
      while (k < dim) {
        queryNormSq += query(k) * query(k)
        k += 1
      }

      var minDist = Float.MaxValue
      var minIdx = 0

      var ti = 0
      while (ti < nTargets) {
        // dot product using BLAS
        val dot = blas.sdot(dim, query, 1, targets(ti), 1)
        val dist = Math.max(0.0f, queryNormSq + targetNormsSq(ti) - 2.0f * dot)

        if (dist < minDist) {
          minDist = dist
          minIdx = ti
        }
        ti += 1
      }

      result(qi) = (minIdx, minDist)
      qi += 1
    }

    result
  }

  /**
   * Find nearest target for each query using dot product (for cosine distance with normalized targets)
   *
   * When targets are normalized (||target|| = 1), finding the nearest by cosine distance
   * is equivalent to finding the maximum dot product:
   *   argmin(cosine_distance) = argmax(dot_product)
   *
   * @param queries Array of query vectors
   * @param targets Array of normalized target vectors (||target|| = 1)
   * @param queryBatchSize Process queries in batches of this size to limit memory
   * @return Array of (nearestIndex, dotProduct) for each query
   */
  def findNearestTargetsByDotProduct(
    queries: Array[Array[Float]],
    targets: Array[Array[Float]],
    queryBatchSize: Int = -1
  ): Array[(Int, Float)] = {
    if (queries.isEmpty || targets.isEmpty) return Array.empty[(Int, Float)]

    val nQueries = queries.length
    val nTargets = targets.length
    val dim = queries(0).length

    val result = new Array[(Int, Float)](nQueries)

    var qi = 0
    while (qi < nQueries) {
      val query = queries(qi)

      var maxDotProduct = Float.MinValue
      var maxIdx = 0

      var ti = 0
      while (ti < nTargets) {
        val dot = blas.sdot(dim, query, 1, targets(ti), 1)

        if (dot > maxDotProduct) {
          maxDotProduct = dot
          maxIdx = ti
        }
        ti += 1
      }

      result(qi) = (maxIdx, maxDotProduct)
      qi += 1
    }

    result
  }

  /**
   * Batch compute Euclidean distances between a query vector and multiple target vectors
   *
   * @param query Query vector
   * @param targets Array of target vectors
   * @return Array of distances
   */
  def batchEuclideanDistance(query: Array[Float], targets: Array[Array[Float]]): Array[Float] = {
    if (targets.isEmpty) return Array.empty[Float]

    val dim = query.length
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
    val distances = new Array[Float](targets.length)

    // Compute query norm once
    val queryNorm = blas.snrm2(dim, query, 1)

    var i = 0
    while (i < targets.length) {
      val target = targets(i)
      val dot = blas.sdot(dim, query, 1, target, 1)
      val targetNorm = blas.snrm2(dim, target, 1)

      val cosineSim = if (queryNorm > 0 && targetNorm > 0) {
        dot / (queryNorm * targetNorm)
      } else {
        0.0f
      }

      distances(i) = 1.0f - cosineSim
      i += 1
    }

    distances
  }

  /**
   * Cached targets for repeated nearest neighbor queries.
   * Pre-computes target norms and stores targets for efficient reuse.
   *
   * Unlike ND4J version, this is pure JVM heap - no native memory issues.
   * No need to call close() but it's provided for API compatibility.
   */
  class CachedTargets(targets: Array[Array[Float]], useCosine: Boolean) extends AutoCloseable {
    val nTargets: Int = targets.length
    val dim: Int = if (targets.nonEmpty) targets(0).length else 0

    // Pre-compute target norms (for L2 distance)
    private val targetNormsSq: Array[Float] = if (!useCosine && nTargets > 0) {
      val norms = new Array[Float](nTargets)
      var t = 0
      while (t < nTargets) {
        var sum = 0.0f
        val target = targets(t)
        var k = 0
        while (k < dim) {
          sum += target(k) * target(k)
          k += 1
        }
        norms(t) = sum
        t += 1
      }
      norms
    } else {
      null
    }

    /**
     * Find nearest target for each query using cached targets.
     */
    def findNearest(queries: Array[Array[Float]], queryBatchSize: Int = -1): Array[(Int, Float)] = {
      if (queries.isEmpty || nTargets == 0) return Array.empty[(Int, Float)]

      val nQueries = queries.length
      val result = new Array[(Int, Float)](nQueries)

      if (useCosine) {
        // Cosine: find max dot product
        var qi = 0
        while (qi < nQueries) {
          val query = queries(qi)
          var maxDotProduct = Float.MinValue
          var maxIdx = 0

          var ti = 0
          while (ti < nTargets) {
            val dot = blas.sdot(dim, query, 1, targets(ti), 1)
            if (dot > maxDotProduct) {
              maxDotProduct = dot
              maxIdx = ti
            }
            ti += 1
          }

          result(qi) = (maxIdx, maxDotProduct)
          qi += 1
        }
      } else {
        // L2: find min squared distance
        var qi = 0
        while (qi < nQueries) {
          val query = queries(qi)

          // Compute query norm
          var queryNormSq = 0.0f
          var k = 0
          while (k < dim) {
            queryNormSq += query(k) * query(k)
            k += 1
          }

          var minDist = Float.MaxValue
          var minIdx = 0

          var ti = 0
          while (ti < nTargets) {
            val dot = blas.sdot(dim, query, 1, targets(ti), 1)
            val dist = Math.max(0.0f, queryNormSq + targetNormsSq(ti) - 2.0f * dot)

            if (dist < minDist) {
              minDist = dist
              minIdx = ti
            }
            ti += 1
          }

          result(qi) = (minIdx, minDist)
          qi += 1
        }
      }

      result
    }

    override def close(): Unit = {
      // No native resources to release - pure JVM heap
    }
  }

  /**
   * Create a cached targets object for repeated queries.
   * Use this when you need to query the same targets multiple times (e.g., KMeans iterations).
   *
   * @param targets Target vectors (e.g., cluster centers)
   * @param useCosine If true, use cosine distance (dot product with normalized targets)
   * @return CachedTargets object
   */
  def createCachedTargets(targets: Array[Array[Float]], useCosine: Boolean): CachedTargets = {
    new CachedTargets(targets, useCosine)
  }

  // Alias for backward compatibility
  type SimpleCachedTargets = CachedTargets

  /**
   * Create a simple cached targets object (same as createCachedTargets now).
   * Kept for backward compatibility.
   */
  def createSimpleCachedTargets(targets: Array[Array[Float]], useCosine: Boolean): CachedTargets = {
    new CachedTargets(targets, useCosine)
  }

  /**
   * Batch compute distances for pairs of vectors.
   * OPTIMIZED: Uses BLAS sdot in a tight loop with minimal overhead.
   * For very large batches, consider using sgemm for even better performance.
   *
   * @param pairs Array of (index_i, index_j) pairs indicating which vectors to compare
   * @param vectors All vectors (pairs reference indices into this array)
   * @param metric Distance metric: "l2" for squared L2 distance, "cosine" for cosine distance
   * @param batchSize Process pairs in batches of this size (ignored - processed all at once)
   * @return Array of distances corresponding to each pair
   */
  def batchPairDistances(
    pairs: Array[(Int, Int)],
    vectors: Array[Array[Float]],
    metric: String,
    batchSize: Int = 10000
  ): Array[Float] = {
    if (pairs.isEmpty) return Array.empty[Float]

    val nPairs = pairs.length
    val dim = vectors(0).length
    val result = new Array[Float](nPairs)
    val useCosine = metric == "cosine"

    // Pre-compute all vector norms (squared) - avoid repeated sqrt
    val normsSq = new Array[Float](vectors.length)
    var v = 0
    while (v < vectors.length) {
      val vec = vectors(v)
      // Use BLAS snrm2 for norm computation (faster than manual loop)
      val norm = blas.snrm2(dim, vec, 1)
      normsSq(v) = norm * norm  // Store squared norm
      v += 1
    }

    // Pre-compute sqrt norms for cosine (only if needed)
    val norms = if (useCosine) {
      val n = new Array[Float](vectors.length)
      var i = 0
      while (i < vectors.length) {
        n(i) = Math.sqrt(normsSq(i)).toFloat
        i += 1
      }
      n
    } else {
      null
    }

    // Batch compute all dot products using BLAS
    // Process in a tight loop to minimize function call overhead
    var idx = 0
    while (idx < nPairs) {
      val (i, j) = pairs(idx)
      val v1 = vectors(i)
      val v2 = vectors(j)

      // Use BLAS sdot for dot product (SIMD optimized)
      val dot = blas.sdot(dim, v1, 1, v2, 1)

      if (useCosine) {
        val norm1 = norms(i)
        val norm2 = norms(j)
        val cosineSim = if (norm1 > 0 && norm2 > 0) dot / (norm1 * norm2) else 0.0f
        result(idx) = 1.0f - cosineSim
      } else {
        // L2 squared distance: ||a - b||² = ||a||² + ||b||² - 2·a·b
        // No sqrt needed since we compare squared distances
        result(idx) = Math.max(0.0f, normsSq(i) + normsSq(j) - 2.0f * dot)
      }

      idx += 1
    }

    result
  }

  /**
   * Pure Scala version of batch pair distances (no BLAS).
   * Alias for batchPairDistances since we no longer have ND4J.
   */
  def batchPairDistancesSimple(
    pairs: Array[(Int, Int)],
    vectors: Array[Array[Float]],
    metric: String
  ): Array[Float] = {
    batchPairDistances(pairs, vectors, metric)
  }

  /**
   * No-op for backward compatibility.
   * Previously cleared ND4J memory pool, now does nothing.
   */
  def clearPool(): Unit = {
    // No-op - no native memory pool to clear
  }
}
