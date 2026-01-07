# SIFT1M K-Means Benchmark

这个基准测试用于对比标准 K-Means 和 Mini-Batch K-Means 在 SIFT1M 数据集上的性能表现。

## ⚡ 快速开始

如果数据已经准备好，直接运行：

```bash
./test-sift1m-benchmark.sh
```

或使用 sbt：

```bash
sbt "testOnly com.zilliz.spark.connector.operations.clustering.Sift1MBenchMark"
```

详细使用文档请查看：[SIFT1M_BENCHMARK_USAGE.md](SIFT1M_BENCHMARK_USAGE.md)

## 📋 测试配置

- **数据集**: SIFT1M (1,000,000 个 128 维向量) - 二进制格式
- **nlist (k)**: 1024 个聚类中心
- **nprobe**: 32 个聚类用于召回率计算
- **维度**: 128

## 📊 评估指标

测试会对比以下四个关键指标：

1. **训练时间 (Training Time)**: 聚类模型训练所需时间
2. **分配时间 (Assignment Time)**: 将所有向量分配到聚类中心的时间
3. **K-Means 损失 (Loss/Inertia)**: 簇内平方和 (WCSS - Within-Cluster Sum of Squares)
4. **Recall@1**: 使用 nprobe=32 时的召回率

## 🚀 准备工作

### 1. 下载并转换 SIFT1M 数据集

#### 步骤 1: 下载 HDF5 格式数据集

```bash
# 创建数据目录
mkdir -p data

# 下载 SIFT1M 数据集 (HDF5 格式，约 500MB)
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5
```

**数据集来源**: http://ann-benchmarks.com/

#### 步骤 2: 转换为二进制格式

由于各种 HDF5 Java 库存在兼容性问题，我们提供了 Python 脚本将 HDF5 转换为简单的二进制格式：

```bash
# 安装依赖（如果需要）
pip install h5py numpy

# 运行转换脚本
python3 convert_sift1m_h5_to_binary.py
```

**转换脚本会生成以下文件：**
- `data/sift1m_train.fbin`: 训练向量 (1M x 128 floats, ~512MB)
- `data/sift1m_test.fbin`: 测试向量 (10K x 128 floats, ~5MB)
- `data/sift1m_neighbors.ibin`: Ground truth 最近邻 (10K x 100 ints, ~4MB)

**二进制格式说明：**
- 每个文件前 12 字节是头部：
  - 8 字节: num_vectors (long)
  - 4 字节: dimension (int)
- 其余数据按行优先顺序存储

### 2. 自定义数据集路径（可选）

默认路径:
- `./data/sift1m_train.fbin`
- `./data/sift1m_test.fbin`
- `./data/sift1m_neighbors.ibin`

或者设置环境变量：
```bash
export SIFT1M_TRAIN=/path/to/sift1m_train.fbin
export SIFT1M_TEST=/path/to/sift1m_test.fbin
export SIFT1M_NEIGHBORS=/path/to/sift1m_neighbors.ibin
```

## 🧪 运行基准测试

### 方法 1: 本地模式（standalone）

```bash
sbt "testOnly com.zilliz.spark.connector.operations.clustering.SIFT1MBenchmarkTest"
```

### 方法 2: 集群模式（分布式执行）

使用 spark-submit 提交到 Spark 集群：

```bash
# 1. 先编译打包
sbt assembly

# 2. 使用 spark-submit 提交到集群
spark-submit \
  --class com.zilliz.spark.connector.operations.clustering.SIFT1MBenchmarkApp \
  --master spark://your-master:7077 \
  --driver-memory 8g \
  --executor-memory 16g \
  --executor-cores 4 \
  --num-executors 4 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  target/scala-2.13/spark-connector-assembly-0.2.1-SNAPSHOT.jar \
  /path/to/sift-128-euclidean.hdf5
```

**参数说明：**
- `--class`: 主类名 `SIFT1MBenchmarkApp`（standalone应用，非测试类）
- `--master`: Spark master 地址，如 `spark://hostname:7077`
- `--driver-memory`: Driver 内存，推荐 8g+（需要加载数据集到内存）
- `--executor-memory`: Executor 内存，推荐 16g+（处理百万级向量聚类）
- `--executor-cores`: 每个 Executor 的核心数（建议 4-8）
- `--num-executors`: Executor 数量（根据集群资源配置）
- 最后一个参数：HDF5 文件路径（可选，默认 `./data/sift-128-euclidean.hdf5`）

**示例（使用您的集群）：**
```bash
spark-submit \
  --class com.zilliz.spark.connector.operations.clustering.SIFT1MBenchmarkApp \
  --master spark://cqy:7077 \
  --driver-memory 8g \
  --executor-memory 16g \
  --executor-cores 4 \
  --num-executors 4 \
  --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
  target/scala-2.13/spark-connector-assembly-0.2.1-SNAPSHOT.jar \
  /home/cqy/workspace/cqy/data/sift-128-euclidean.hdf5
```

### 方法 3: 使用自定义数据集路径

```bash
export SIFT1M_PATH=/path/to/sift-128-euclidean.hdf5
sbt "testOnly com.zilliz.spark.connector.operations.clustering.SIFT1MBenchmarkTest"
```

### 方法 4: 运行所有聚类测试

```bash
sbt "testOnly com.zilliz.spark.connector.operations.clustering.*"
```

## 📈 预期输出

测试运行后会输出详细的对比报告：

```
================================================================================
SIFT1M Benchmark: K-Means vs Mini-Batch K-Means
================================================================================
Loading SIFT1M dataset from HDF5: ./data/sift-128-euclidean.hdf5
Loaded 1000000 training vectors
Loaded 10000 test vectors
Loaded 10000 ground truth entries
Converting to Spark DataFrame...
Created DataFrame with 1000000 vectors
--------------------------------------------------------------------------------
Configuration: nlist=1024, nprobe=32, dim=128
--------------------------------------------------------------------------------

Testing Standard K-Means...
--------------------------------------------------------------------------------
✓ Training time: 45.23 seconds
✓ Assignment time: 12.34 seconds
✓ K-Means loss (inertia): 1234567.89
✓ Recall@1 (nprobe=32): 0.8945 (89.45%)

Testing Mini-Batch K-Means...
--------------------------------------------------------------------------------
✓ Training time: 18.92 seconds
✓ Assignment time: 11.87 seconds
✓ K-Means loss (inertia): 1245678.90
✓ Recall@1 (nprobe=32): 0.8823 (88.23%)

================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================
Dataset: SIFT1M (1000000 vectors, 128 dimensions)
Configuration: nlist=1024, nprobe=32
--------------------------------------------------------------------------------
Metric                                   K-Means         Mini-Batch      Speedup
--------------------------------------------------------------------------------
Training Time (s)                        45.23           18.92           2.39x
Assignment Time (s)                      12.34           11.87           1.04x
K-Means Loss (inertia)                   1234567.89      1245678.90      1.01x
Recall@1 (nprobe=32)                     89.45%          88.23%          0.99x
--------------------------------------------------------------------------------
Overall Training Speedup: 2.39x
Loss Difference: 0.90% worse
Recall Difference: 1.36% worse
================================================================================
```

## 🔍 结果解读

### 训练时间 (Training Time)
- **预期**: Mini-Batch K-Means 通常快 2-3 倍
- **原因**: 每次迭代只使用部分数据（默认 10%）

### 分配时间 (Assignment Time)
- **预期**: 两者相近
- **原因**: 分配阶段都需要计算所有向量到所有中心的距离

### K-Means 损失 (Loss)
- **预期**: Mini-Batch 的损失略高 (1-5%)
- **原因**: 使用采样数据训练，精度略有下降

### Recall@1 (nprobe=32)
- **预期**: Mini-Batch 的召回率略低 (0.5-2%)
- **原因**: 聚类中心略微不够精确
- **实际意义**: 在向量检索场景中，使用 32 个最近的聚类进行搜索的准确率

## ⚙️ 调优建议

### 如果需要更高精度
```scala
// 增加批次数量和每批次的迭代次数
val minibatchOp = new MiniBatchKMeansOperation(
  k = 1024,
  batchSize = 0.2,      // 增加批次大小到 20%
  numBatches = 20,      // 增加批次数量
  maxIterPerBatch = 10  // 增加每批次迭代
)
```

### 如果需要更快速度
```scala
// 使用更小的批次
val minibatchOp = new MiniBatchKMeansOperation(
  k = 1024,
  batchSize = 0.05,     // 减小批次大小到 5%
  numBatches = 5,       // 减少批次数量
  maxIterPerBatch = 3   // 减少每批次迭代
)
```

### 不同 nprobe 值的影响

| nprobe | Recall 预期 | 搜索时间 | 使用场景 |
|--------|------------|---------|----------|
| 1      | 40-50%     | 最快    | 快速粗略搜索 |
| 8      | 70-80%     | 快      | 平衡性能 |
| 32     | 85-92%     | 中等    | 推荐配置 |
| 64     | 90-95%     | 慢      | 高精度需求 |
| 128    | 95-98%     | 很慢    | 最高精度 |

## 📝 注意事项

1. **内存要求**: 测试需要至少 8GB 内存
   - SIFT1M 数据集加载到内存约需 500MB
   - Spark 处理需要额外内存
   - 如果内存不足，可以修改 SparkSession 配置

2. **数据集大小**: HDF5 文件约 500MB
   - 下载时间取决于网络速度
   - 确保有足够的磁盘空间

3. **运行时间**: 完整测试可能需要 5-10 分钟
   - K-Means 训练: ~2-5 分钟
   - Mini-Batch 训练: ~1-2 分钟
   - Recall 计算: ~2-3 分钟

4. **Java 版本**:
   - 需要 Java 17+ (Spark 4.0.0)
   - 或使用 Java 11 并降级到 Spark 3.5.x

## 🐛 故障排除

### 问题: 找不到数据集文件

**症状**:
```
SIFT1M dataset (binary format) not found.
```

**解决方案**:
```bash
# 1. 检查二进制文件是否存在
ls -lh ./data/sift1m_*.bin

# 2. 如果没有，先下载 HDF5 文件
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O ./data/sift-128-euclidean.hdf5

# 3. 运行转换脚本
python3 convert_sift1m_h5_to_binary.py

# 4. 或设置正确的路径
export SIFT1M_TRAIN=/path/to/sift1m_train.fbin
export SIFT1M_TEST=/path/to/sift1m_test.fbin
export SIFT1M_NEIGHBORS=/path/to/sift1m_neighbors.ibin
```

### 问题: Python h5py 库缺失

**症状**:
```
ModuleNotFoundError: No module named 'h5py'
```

**解决方案**:
```bash
# 使用 pip 安装
pip install h5py numpy

# 或使用 conda
conda install h5py numpy
```

### 问题: 内存不足 (OutOfMemoryError)

**症状**:
```
java.lang.OutOfMemoryError: Java heap space
```

**解决方案**:
修改测试中的 Spark 配置（已在代码中设置为 8g）或增加内存：
```scala
spark = SparkSession.builder()
  .config("spark.driver.memory", "16g")  // 增加到 16GB
  .config("spark.executor.memory", "16g")
  .getOrCreate()
```

或在运行时指定：
```bash
sbt -J-Xmx16g "testOnly com.zilliz.spark.connector.operations.clustering.SIFT1MBenchmarkTest"
```

### 问题: Java 版本不兼容

**症状**:
```
UnsupportedClassVersionError: ... has been compiled by a more recent version of the Java Runtime
```

**解决方案**:
```bash
# 使用 SDKMAN 安装 Java 17
sdk install java 17.0.14.crac-zulu
sdk use java 17.0.14.crac-zulu

# 验证版本
java -version
```

## 🔗 数据集来源

- **ANN Benchmarks**: http://ann-benchmarks.com/
- **SIFT1M 原始数据**: http://corpus-texmex.irisa.fr/
- **HDF5 格式说明**: https://github.com/erikbern/ann-benchmarks

## 📚 相关文档

- [K-Means Quickstart](KMEANS_QUICKSTART.md) - K-Means 快速入门
- [Clustering Documentation](docs/CLUSTERING.md) - 聚类操作详细文档
- [ANN Benchmarks](http://ann-benchmarks.com/) - 向量检索基准测试平台

## 🎯 基准测试目标

这个基准测试帮助你了解：

1. **性能权衡**: Mini-Batch K-Means 的速度提升与精度损失的平衡
2. **大规模场景**: 在百万级向量数据上的实际表现
3. **配置选择**: 如何根据实际需求选择合适的算法和参数
4. **向量检索**: 聚类结果在向量检索中的实际效果
5. **IVF 索引**: 理解 Milvus/Faiss 中 IVF (Inverted File) 索引的工作原理

## 💡 扩展实验

你可以尝试修改参数进行更多实验：

### 实验 1: 不同的 nlist 值
```scala
// 测试不同数量的聚类中心
val nlistValues = Seq(128, 256, 512, 1024, 2048, 4096)
```

### 实验 2: 不同的 batch size
```scala
// 测试不同的批次大小
val batchSizes = Seq(0.01, 0.05, 0.1, 0.2, 0.5)
```

### 实验 3: Recall vs nprobe 曲线
```scala
// 绘制 recall-nprobe 曲线
val nprobeValues = Seq(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
```

## 📞 问题反馈

如果遇到问题或有改进建议，请提交 Issue 或 Pull Request。

## 🎓 学习资源

- [Faiss IVF 索引原理](https://github.com/facebookresearch/faiss/wiki/Faster-search)
- [K-Means 算法详解](https://en.wikipedia.org/wiki/K-means_clustering)
- [Mini-Batch K-Means](https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf)
- [向量检索基准测试](http://ann-benchmarks.com/)
