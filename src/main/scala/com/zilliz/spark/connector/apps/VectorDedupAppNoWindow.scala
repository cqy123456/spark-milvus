package com.zilliz.spark.connector.apps

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.slf4j.{Logger, LoggerFactory}

/**
 * Vector Deduplication without Sliding Window
 *
 * 三种模式:
 * 1. BRUTE_FORCE: 簇内全量两两比较，O(n²)，零漏检
 * 2. GRID_BASED: 基于网格的空间划分，O(n × 网格邻居)，低漏检
 * 3. SORTED_BLOCKS: 分块 + 块内全量 + 块间边界比较
 *
 * 适用场景:
 * - 簇较小 (< 5000 向量) 时用 BRUTE_FORCE
 * - 簇较大时用 GRID_BASED 或 SORTED_BLOCKS
 */
object DedupMode extends Enumeration {
  type DedupMode = Value
  val BRUTE_FORCE, GRID_BASED, SORTED_BLOCKS = Value
}

class VectorDedupAppNoWindow(
  val k: Int,
  val distanceThreshold: Float = 0.1f,
  val maxIterKMeans: Int = 20,
  val distanceMetric: String = "l2",
  val dedupMode: DedupMode.DedupMode = DedupMode.BRUTE_FORCE,
  val blockSize: Int = 500,  // for SORTED_BLOCKS mode
  val gridResolution: Int = 10,  // for GRID_BASED mode
  val featuresCol: String = "features",
  val idCol: String = "id",
  val outputDir: Option[String] = None
) extends Serializable {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(classOf[VectorDedupAppNoWindow])

  require(k > 0, "Number of clusters must be positive")
  require(distanceThreshold > 0, "Distance threshold must be positive")
  require(distanceMetric == "l2" || distanceMetric == "cosine",
    s"distanceMetric must be 'l2' or 'cosine', got: $distanceMetric")

  /**
   * Step 3 替代实现: 不使用滑动窗口的连通分量发现
   */
  def buildThresholdGraphAndFindComponentsNoWindow(df: DataFrame): DataFrame = {
    logger.info(s"Step 3: Building threshold graph (mode=$dedupMode, no sliding window)...")
    val startTime = System.currentTimeMillis()

    val spark = df.sparkSession
    val threshold = distanceThreshold
    val metric = distanceMetric
    val featCol = featuresCol
    val idColumn = idCol
    val mode = dedupMode
    val blkSize = blockSize
    val gridRes = gridResolution

    val componentsRDD = df.rdd.mapPartitions { rowIterator =>
      val vectors = rowIterator.toArray

      if (vectors.isEmpty) {
        Iterator.empty
      } else {
        val bucketId = vectors(0).getAs[Int]("cluster_id")
        val totalVectors = vectors.length

        // 提取特征向量
        val features = vectors.map { row =>
          row.getAs[scala.collection.mutable.WrappedArray[Float]](featCol).toArray
        }
        val ids = vectors.map(_.getAs[Long](idColumn))

        // Union-Find 数据结构
        val parent = Array.tabulate(totalVectors)(identity)
        val rank = new Array[Int](totalVectors)

        def find(x: Int): Int = {
          if (parent(x) != x) {
            parent(x) = find(parent(x))
          }
          parent(x)
        }

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

        // 距离计算
        val thresholdSq = threshold * threshold
        val useCosine = metric == "cosine"

        def computeL2DistanceSq(v1: Array[Float], v2: Array[Float]): Float = {
          var sumSq = 0.0f
          var i = 0
          val len = v1.length
          while (i < len) {
            val diff = v1(i) - v2(i)
            sumSq += diff * diff
            i += 1
          }
          sumSq
        }

        def computeCosineDistance(v1: Array[Float], v2: Array[Float]): Float = {
          var dotProduct = 0.0f
          var norm1Sq = 0.0f
          var norm2Sq = 0.0f
          var i = 0
          val len = v1.length
          while (i < len) {
            dotProduct += v1(i) * v2(i)
            norm1Sq += v1(i) * v1(i)
            norm2Sq += v2(i) * v2(i)
            i += 1
          }
          val norm1 = math.sqrt(norm1Sq).toFloat
          val norm2 = math.sqrt(norm2Sq).toFloat
          if (norm1 > 0 && norm2 > 0) {
            1.0f - (dotProduct / (norm1 * norm2))
          } else {
            1.0f
          }
        }

        @inline def isWithinThreshold(i: Int, j: Int): Boolean = {
          if (useCosine) {
            computeCosineDistance(features(i), features(j)) < threshold
          } else {
            computeL2DistanceSq(features(i), features(j)) < thresholdSq
          }
        }

        var edgeCount = 0L

        // 根据模式选择算法
        mode match {
          // ============================================================
          // 模式 1: BRUTE_FORCE - 全量两两比较
          // ============================================================
          case DedupMode.BRUTE_FORCE =>
            var i = 0
            while (i < totalVectors) {
              var j = i + 1
              while (j < totalVectors) {
                // 早停优化: 如果已经在同一组，跳过
                if (find(i) != find(j) && isWithinThreshold(i, j)) {
                  union(i, j)
                  edgeCount += 1
                }
                j += 1
              }
              i += 1
            }

          // ============================================================
          // 模式 2: SORTED_BLOCKS - 分块比较
          // ============================================================
          case DedupMode.SORTED_BLOCKS =>
            // 按到簇中心的距离排序的索引
            val sortedIndices = vectors.indices.sortBy { i =>
              vectors(i).getAs[Float]("distance")
            }.toArray

            val numBlocks = (totalVectors + blkSize - 1) / blkSize

            // 块内全量比较
            var blockIdx = 0
            while (blockIdx < numBlocks) {
              val blockStart = blockIdx * blkSize
              val blockEnd = math.min(blockStart + blkSize, totalVectors)

              var i = blockStart
              while (i < blockEnd) {
                var j = i + 1
                while (j < blockEnd) {
                  val ii = sortedIndices(i)
                  val jj = sortedIndices(j)
                  if (find(ii) != find(jj) && isWithinThreshold(ii, jj)) {
                    union(ii, jj)
                    edgeCount += 1
                  }
                  j += 1
                }
                i += 1
              }
              blockIdx += 1
            }

            // 相邻块边界比较 (前块的后半部分 vs 后块的前半部分)
            val boundarySize = math.min(blkSize / 2, 100)
            blockIdx = 0
            while (blockIdx < numBlocks - 1) {
              val block1End = math.min((blockIdx + 1) * blkSize, totalVectors)
              val block1BoundaryStart = math.max(block1End - boundarySize, blockIdx * blkSize)

              val block2Start = (blockIdx + 1) * blkSize
              val block2BoundaryEnd = math.min(block2Start + boundarySize, totalVectors)

              var i = block1BoundaryStart
              while (i < block1End) {
                var j = block2Start
                while (j < block2BoundaryEnd) {
                  val ii = sortedIndices(i)
                  val jj = sortedIndices(j)
                  if (find(ii) != find(jj) && isWithinThreshold(ii, jj)) {
                    union(ii, jj)
                    edgeCount += 1
                  }
                  j += 1
                }
                i += 1
              }
              blockIdx += 1
            }

          // ============================================================
          // 模式 3: GRID_BASED - 基于网格的空间划分
          // ============================================================
          case DedupMode.GRID_BASED =>
            val dim = features(0).length

            // 计算每个维度的范围
            val mins = new Array[Float](dim)
            val maxs = new Array[Float](dim)
            java.util.Arrays.fill(mins, Float.MaxValue)
            java.util.Arrays.fill(maxs, Float.MinValue)

            var i = 0
            while (i < totalVectors) {
              val f = features(i)
              var d = 0
              while (d < dim) {
                if (f(d) < mins(d)) mins(d) = f(d)
                if (f(d) > maxs(d)) maxs(d) = f(d)
                d += 1
              }
              i += 1
            }

            // 使用前几个主要维度进行网格划分 (高维时只用前几维)
            val gridDims = math.min(dim, 8)
            val cellSize = new Array[Float](gridDims)
            for (d <- 0 until gridDims) {
              cellSize(d) = (maxs(d) - mins(d)) / gridRes
              if (cellSize(d) <= 0) cellSize(d) = 1.0f
            }

            // 计算每个向量的网格坐标
            def getGridCell(idx: Int): Seq[Int] = {
              val f = features(idx)
              (0 until gridDims).map { d =>
                val cell = ((f(d) - mins(d)) / cellSize(d)).toInt
                math.min(cell, gridRes - 1)
              }
            }

            // 构建网格 -> 向量索引的映射
            val grid = scala.collection.mutable.HashMap[Seq[Int], scala.collection.mutable.ArrayBuffer[Int]]()
            i = 0
            while (i < totalVectors) {
              val cell = getGridCell(i)
              grid.getOrElseUpdate(cell, scala.collection.mutable.ArrayBuffer[Int]()) += i
              i += 1
            }

            // 生成邻居偏移 (-1, 0, 1) 的组合
            def generateNeighborOffsets(dims: Int): Seq[Seq[Int]] = {
              if (dims == 0) Seq(Seq.empty)
              else {
                for {
                  rest <- generateNeighborOffsets(dims - 1)
                  offset <- Seq(-1, 0, 1)
                } yield offset +: rest
              }
            }
            val neighborOffsets = generateNeighborOffsets(gridDims)

            // 对每个网格单元，比较本单元和邻居单元的向量
            for ((cell, indices) <- grid) {
              // 本单元内部比较
              val n = indices.length
              var ii = 0
              while (ii < n) {
                var jj = ii + 1
                while (jj < n) {
                  val i = indices(ii)
                  val j = indices(jj)
                  if (find(i) != find(j) && isWithinThreshold(i, j)) {
                    union(i, j)
                    edgeCount += 1
                  }
                  jj += 1
                }
                ii += 1
              }

              // 辅助函数: 字典序比较两个 Seq[Int]
              def seqGreaterThan(a: Seq[Int], b: Seq[Int]): Boolean = {
                a.zip(b).find { case (x, y) => x != y } match {
                  case Some((x, y)) => x > y
                  case None => false
                }
              }

              // 与邻居单元比较 (只比较一半邻居，避免重复)
              for (offset <- neighborOffsets) {
                val neighborCell = cell.zip(offset).map { case (c, o) => c + o }
                // 只处理"更大"的邻居，避免重复比较 (字典序比较)
                if (seqGreaterThan(neighborCell, cell)) {
                  grid.get(neighborCell).foreach { neighborIndices =>
                    for (i <- indices; j <- neighborIndices) {
                      if (find(i) != find(j) && isWithinThreshold(i, j)) {
                        union(i, j)
                        edgeCount += 1
                      }
                    }
                  }
                }
              }
            }
        }

        // 输出日志
        val modeStr = mode.toString
        println(s"  Bucket $bucketId: $totalVectors vectors, $edgeCount edges (mode=$modeStr)")

        // 返回 (id, component_id)
        (0 until totalVectors).iterator.map { idx =>
          val id = ids(idx)
          val rootIdx = find(idx)
          val componentId = ids(rootIdx)
          (id, componentId)
        }
      }
    }

    import spark.implicits._
    val componentsDF = componentsRDD.toDF("id", "component_id")

    val totalTime = (System.currentTimeMillis() - startTime) / 1000.0
    logger.info(f"✓ Threshold graph (no window) completed in $totalTime%.2f seconds")

    componentsDF
  }
}

/**
 * 独立的工具对象，可直接替换原有 Step 3
 */
object NoWindowDedup {

  /**
   * 簇内全量比较 - 最简单直接的实现
   *
   * @param vectors 向量数组 Array[Array[Float]]
   * @param ids 对应的 ID 数组
   * @param threshold 距离阈值
   * @param metric "l2" 或 "cosine"
   * @return (id, component_id) 的数组
   */
  def bruteForceDedup(
    vectors: Array[Array[Float]],
    ids: Array[Long],
    threshold: Float,
    metric: String = "l2"
  ): Array[(Long, Long)] = {

    val n = vectors.length
    val thresholdSq = threshold * threshold
    val useCosine = metric == "cosine"

    // Union-Find
    val parent = Array.tabulate(n)(identity)
    val rank = new Array[Int](n)

    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }

    def union(x: Int, y: Int): Unit = {
      val px = find(x)
      val py = find(y)
      if (px != py) {
        if (rank(px) < rank(py)) parent(px) = py
        else if (rank(px) > rank(py)) parent(py) = px
        else { parent(py) = px; rank(px) += 1 }
      }
    }

    // 距离计算
    def l2DistSq(v1: Array[Float], v2: Array[Float]): Float = {
      var sum = 0.0f
      var i = 0
      while (i < v1.length) {
        val d = v1(i) - v2(i)
        sum += d * d
        i += 1
      }
      sum
    }

    def cosineDist(v1: Array[Float], v2: Array[Float]): Float = {
      var dot = 0.0f
      var n1 = 0.0f
      var n2 = 0.0f
      var i = 0
      while (i < v1.length) {
        dot += v1(i) * v2(i)
        n1 += v1(i) * v1(i)
        n2 += v2(i) * v2(i)
        i += 1
      }
      if (n1 > 0 && n2 > 0) 1.0f - dot / math.sqrt(n1 * n2).toFloat
      else 1.0f
    }

    // 全量比较
    var i = 0
    while (i < n) {
      var j = i + 1
      while (j < n) {
        val similar = if (useCosine) {
          cosineDist(vectors(i), vectors(j)) < threshold
        } else {
          l2DistSq(vectors(i), vectors(j)) < thresholdSq
        }
        if (similar && find(i) != find(j)) {
          union(i, j)
        }
        j += 1
      }
      i += 1
    }

    // 返回结果
    Array.tabulate(n) { i =>
      (ids(i), ids(find(i)))
    }
  }

  /**
   * 自适应选择算法
   */
  def adaptiveDedup(
    vectors: Array[Array[Float]],
    ids: Array[Long],
    threshold: Float,
    metric: String = "l2"
  ): Array[(Long, Long)] = {

    val n = vectors.length

    if (n <= 2000) {
      // 小簇: 全量比较
      bruteForceDedup(vectors, ids, threshold, metric)
    } else if (n <= 10000) {
      // 中簇: 分块比较
      sortedBlocksDedup(vectors, ids, threshold, metric, blockSize = 500)
    } else {
      // 大簇: 网格方法
      gridBasedDedup(vectors, ids, threshold, metric, gridResolution = 20)
    }
  }

  /**
   * 分块比较
   */
  def sortedBlocksDedup(
    vectors: Array[Array[Float]],
    ids: Array[Long],
    threshold: Float,
    metric: String = "l2",
    blockSize: Int = 500
  ): Array[(Long, Long)] = {

    val n = vectors.length
    val thresholdSq = threshold * threshold
    val useCosine = metric == "cosine"

    val parent = Array.tabulate(n)(identity)
    val rank = new Array[Int](n)

    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }

    def union(x: Int, y: Int): Unit = {
      val px = find(x)
      val py = find(y)
      if (px != py) {
        if (rank(px) < rank(py)) parent(px) = py
        else if (rank(px) > rank(py)) parent(py) = px
        else { parent(py) = px; rank(px) += 1 }
      }
    }

    def isWithinThreshold(i: Int, j: Int): Boolean = {
      if (useCosine) {
        var dot = 0.0f; var n1 = 0.0f; var n2 = 0.0f
        val v1 = vectors(i); val v2 = vectors(j)
        var k = 0
        while (k < v1.length) {
          dot += v1(k) * v2(k)
          n1 += v1(k) * v1(k)
          n2 += v2(k) * v2(k)
          k += 1
        }
        (1.0f - dot / math.sqrt(n1 * n2).toFloat) < threshold
      } else {
        var sum = 0.0f
        val v1 = vectors(i); val v2 = vectors(j)
        var k = 0
        while (k < v1.length) {
          val d = v1(k) - v2(k)
          sum += d * d
          k += 1
        }
        sum < thresholdSq
      }
    }

    // 计算向量的 L2 范数作为排序依据
    val norms = vectors.map { v =>
      var sum = 0.0f
      var i = 0
      while (i < v.length) { sum += v(i) * v(i); i += 1 }
      math.sqrt(sum).toFloat
    }
    val sortedIndices = norms.indices.sortBy(norms).toArray

    val numBlocks = (n + blockSize - 1) / blockSize

    // 块内比较
    for (b <- 0 until numBlocks) {
      val start = b * blockSize
      val end = math.min(start + blockSize, n)
      for (i <- start until end; j <- i + 1 until end) {
        val ii = sortedIndices(i)
        val jj = sortedIndices(j)
        if (find(ii) != find(jj) && isWithinThreshold(ii, jj)) {
          union(ii, jj)
        }
      }
    }

    // 相邻块边界比较
    val boundarySize = math.min(blockSize / 4, 100)
    for (b <- 0 until numBlocks - 1) {
      val b1End = math.min((b + 1) * blockSize, n)
      val b1Start = math.max(b1End - boundarySize, b * blockSize)
      val b2Start = (b + 1) * blockSize
      val b2End = math.min(b2Start + boundarySize, n)

      for (i <- b1Start until b1End; j <- b2Start until b2End) {
        val ii = sortedIndices(i)
        val jj = sortedIndices(j)
        if (find(ii) != find(jj) && isWithinThreshold(ii, jj)) {
          union(ii, jj)
        }
      }
    }

    Array.tabulate(n)(i => (ids(i), ids(find(i))))
  }

  /**
   * 基于网格的空间划分
   */
  def gridBasedDedup(
    vectors: Array[Array[Float]],
    ids: Array[Long],
    threshold: Float,
    metric: String = "l2",
    gridResolution: Int = 10
  ): Array[(Long, Long)] = {

    val n = vectors.length
    val dim = vectors(0).length
    val thresholdSq = threshold * threshold
    val useCosine = metric == "cosine"

    val parent = Array.tabulate(n)(identity)
    val rank = new Array[Int](n)

    def find(x: Int): Int = {
      if (parent(x) != x) parent(x) = find(parent(x))
      parent(x)
    }

    def union(x: Int, y: Int): Unit = {
      val px = find(x)
      val py = find(y)
      if (px != py) {
        if (rank(px) < rank(py)) parent(px) = py
        else if (rank(px) > rank(py)) parent(py) = px
        else { parent(py) = px; rank(px) += 1 }
      }
    }

    def isWithinThreshold(i: Int, j: Int): Boolean = {
      val v1 = vectors(i)
      val v2 = vectors(j)
      if (useCosine) {
        var dot = 0.0f; var n1 = 0.0f; var n2 = 0.0f
        var k = 0
        while (k < v1.length) {
          dot += v1(k) * v2(k)
          n1 += v1(k) * v1(k)
          n2 += v2(k) * v2(k)
          k += 1
        }
        (1.0f - dot / math.sqrt(n1 * n2).toFloat) < threshold
      } else {
        var sum = 0.0f
        var k = 0
        while (k < v1.length) {
          val d = v1(k) - v2(k)
          sum += d * d
          k += 1
        }
        sum < thresholdSq
      }
    }

    // 使用前几个维度做网格 (高维时只用部分维度)
    val gridDims = math.min(dim, 6)

    // 计算边界
    val mins = Array.fill(gridDims)(Float.MaxValue)
    val maxs = Array.fill(gridDims)(Float.MinValue)
    for (i <- 0 until n; d <- 0 until gridDims) {
      if (vectors(i)(d) < mins(d)) mins(d) = vectors(i)(d)
      if (vectors(i)(d) > maxs(d)) maxs(d) = vectors(i)(d)
    }

    val cellSize = (0 until gridDims).map { d =>
      val size = (maxs(d) - mins(d)) / gridResolution
      if (size <= 0) 1.0f else size
    }.toArray

    // 计算网格坐标
    def getCell(idx: Int): IndexedSeq[Int] = {
      (0 until gridDims).map { d =>
        val c = ((vectors(idx)(d) - mins(d)) / cellSize(d)).toInt
        math.max(0, math.min(c, gridResolution - 1))
      }
    }

    // 构建网格索引
    val grid = scala.collection.mutable.HashMap[IndexedSeq[Int], scala.collection.mutable.ArrayBuffer[Int]]()
    for (i <- 0 until n) {
      val cell = getCell(i)
      grid.getOrElseUpdate(cell, scala.collection.mutable.ArrayBuffer()) += i
    }

    // 生成邻居偏移
    def neighborOffsets(dims: Int): Seq[IndexedSeq[Int]] = {
      if (dims == 0) Seq(IndexedSeq.empty)
      else for {
        rest <- neighborOffsets(dims - 1)
        o <- Seq(-1, 0, 1)
      } yield o +: rest
    }
    val offsets = neighborOffsets(gridDims)

    // 比较
    for ((cell, indices) <- grid) {
      // 单元内比较
      for (i <- indices.indices; j <- i + 1 until indices.length) {
        val ii = indices(i)
        val jj = indices(j)
        if (find(ii) != find(jj) && isWithinThreshold(ii, jj)) {
          union(ii, jj)
        }
      }

      // 辅助函数: 字典序比较
      def idxSeqGreaterThan(a: IndexedSeq[Int], b: IndexedSeq[Int]): Boolean = {
        a.zip(b).find { case (x, y) => x != y } match {
          case Some((x, y)) => x > y
          case None => false
        }
      }

      // 邻居单元比较
      for (offset <- offsets if offset.exists(_ != 0)) {
        val neighbor = cell.zip(offset).map { case (c, o) => c + o }
        // 字典序比较，只处理"更大"的邻居避免重复
        if (neighbor.forall(c => c >= 0 && c < gridResolution) &&
            idxSeqGreaterThan(neighbor, cell)) {
          grid.get(neighbor).foreach { neighborIndices =>
            for (i <- indices; j <- neighborIndices) {
              if (find(i) != find(j) && isWithinThreshold(i, j)) {
                union(i, j)
              }
            }
          }
        }
      }
    }

    Array.tabulate(n)(i => (ids(i), ids(find(i))))
  }
}
