package com.zilliz.spark.connector.utils

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
 * Optimized vector operations using ND4J
 *
 * ND4J provides high-performance linear algebra operations with:
 * - SIMD optimizations (AVX, AVX2, AVX-512)
 * - Native BLAS backend
 * - Batch operations for better cache utilization
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

    // Convert to ND4J arrays
    val nd1 = Nd4j.create(v1)
    val nd2 = Nd4j.create(v2)

    // Compute distance using ND4J's optimized implementation
    Transforms.euclideanDistance(nd1, nd2).getFloat(0)
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

    // Convert to ND4J arrays
    val nd1 = Nd4j.create(v1)
    val nd2 = Nd4j.create(v2)

    // Compute cosine similarity using ND4J
    val cosineSimilarity = Transforms.cosineSim(nd1, nd2).getFloat(0)

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

    val nd1 = Nd4j.create(v1)
    val nd2 = Nd4j.create(v2)

    nd1.mmul(nd2.transpose()).getFloat(0)
  }

  /**
   * Compute L2 norm of a vector
   *
   * @param v Vector
   * @return L2 norm as Float
   */
  def norm2(v: Array[Float]): Float = {
    val nd = Nd4j.create(v)
    nd.norm2Number().floatValue()
  }

  /**
   * Batch compute Euclidean distances between a query vector and multiple target vectors
   * This is more efficient than computing distances one by one
   *
   * @param query Query vector
   * @param targets Array of target vectors
   * @return Array of distances
   */
  def batchEuclideanDistance(query: Array[Float], targets: Array[Array[Float]]): Array[Float] = {
    if (targets.isEmpty) return Array.empty[Float]

    val dim = query.length
    require(targets.forall(_.length == dim), "All vectors must have the same dimension")

    // Convert query to ND4J array
    val queryND = Nd4j.create(query).reshape(1, dim)

    // Convert targets to ND4J matrix (one row per target)
    val targetsND = Nd4j.create(targets.length, dim)
    targets.zipWithIndex.foreach { case (target, i) =>
      targetsND.putRow(i, Nd4j.create(target))
    }

    // Compute pairwise distances efficiently
    // Distance = sqrt(sum((query - target)^2))
    val diff = targetsND.subRowVector(queryND.getRow(0))
    val squared = Transforms.pow(diff, 2)
    val summed = squared.sum(1)  // Sum along dimension axis
    val distances = Transforms.sqrt(summed)

    distances.toFloatVector
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

    // Convert to ND4J arrays
    val queryND = Nd4j.create(query).reshape(1, dim)
    val targetsND = Nd4j.create(targets.length, dim)
    targets.zipWithIndex.foreach { case (target, i) =>
      targetsND.putRow(i, Nd4j.create(target))
    }

    // Compute cosine similarities
    val queryNorm = queryND.norm2(1)
    val targetNorms = targetsND.norm2(1)
    val dotProducts = targetsND.mmul(queryND.transpose()).getColumn(0)

    // Cosine similarity = dot / (norm1 * norm2)
    val similarities = dotProducts.div(queryNorm).div(targetNorms)

    // Convert to distances: 1 - similarity
    val ones = Nd4j.ones(similarities.shape(): _*)
    val distances = ones.sub(similarities)

    distances.toFloatVector
  }
}
