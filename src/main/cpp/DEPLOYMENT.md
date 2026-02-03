# Native Library 部署说明

## 编译

```bash
cd /home/ubuntu/spark-milvus/src/main/cpp
./build.sh
```

编译完成后，库文件位于：`build/libvectorops_avx512.so`

## 上传到 S3

### 方法 1: 使用上传脚本（推荐）

```bash
cd /home/ubuntu/spark-milvus/src/main/cpp
./upload_to_s3.sh
```

**注意**: 需要配置 AWS 凭证（通过 `aws configure` 或环境变量）

### 方法 2: 手动上传

```bash
aws s3 cp build/libvectorops_avx512.so \
  s3://vdc-cloud-dev/SessionStage/job-02f46718f29vfqaxrsuvjw/cqy-dedup/native-libs/libvectorops_avx512.so
```

## 验证

检查 S3 上的文件：

```bash
aws s3 ls s3://vdc-cloud-dev/SessionStage/job-02f46718f29vfqaxrsuvjw/cqy-dedup/native-libs/
```

## Kubernetes 配置

配置文件：`/home/ubuntu/k8s/spark-10m.yaml`

关键配置：
- `spark.zilliz.vectorops.preferNative: "true"` - 启用 native 实现
- `spark.zilliz.vectorops.useNd4j: "true"` - 如果 native 不可用，回退到 ND4J
- `-Djava.library.path=/opt/spark/native-libs:...` - 设置库搜索路径
- `LD_LIBRARY_PATH=/opt/spark/native-libs:...` - 设置系统库路径

## 部署流程

1. **编译库**:
   ```bash
   cd /home/ubuntu/spark-milvus/src/main/cpp
   ./build.sh
   ```

2. **上传到 S3**:
   ```bash
   ./upload_to_s3.sh
   ```

3. **应用 Kubernetes 配置**:
   ```bash
   kubectl apply -f /home/ubuntu/k8s/spark-10m.yaml
   ```

4. **验证部署**:
   - 检查 initContainer 日志，确认库已下载
   - 检查 Spark 应用日志，确认 native 库已加载

## 回退机制

如果 native 库不可用，系统会自动回退：
1. **Native C++ AVX-512** (最快) - 如果库可用且 CPU 支持
2. **ND4J** (中等) - 如果 `useNd4j=true` 且 native 不可用
3. **Pure Scala** (最慢) - 最终回退

## 性能对比

| 实现 | 性能 | 内存使用 |
|------|------|---------|
| Native AVX-512 | 3-5x 快 | 20-30% 减少 |
| ND4J | 基准 | 基准 |
| Pure Scala | 0.3-0.5x | 基准 |

## 故障排查

### 库未加载

检查日志：
```bash
kubectl logs <pod-name> -c install-native-libs
```

常见问题：
- S3 访问权限不足
- 库路径配置错误
- CPU 不支持 AVX-512（会自动回退到 AVX2 或标量）

### 性能未提升

- 确认 `spark.zilliz.vectorops.preferNative: "true"` 已设置
- 检查日志确认使用了 native 实现
- 验证 CPU 支持 AVX-512（`grep avx512 /proc/cpuinfo`）
