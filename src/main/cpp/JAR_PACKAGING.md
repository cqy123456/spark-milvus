# Native 库打包到 JAR 说明

## 实现方式

Native 库（`.so` 文件）可以打包到 JAR 中，但需要在运行时提取到临时文件系统才能加载。

## 实现细节

### 1. 库文件位置

Native 库文件已放置在：
```
src/main/resources/native/libvectorops_avx512.so
```

### 2. Java 代码实现

`VectorOpsNative.java` 已更新，支持：
- 从 JAR 中提取 native 库到临时文件
- 自动设置执行权限
- 加载临时文件中的库
- 如果 JAR 中不存在，回退到系统库路径

### 3. 加载流程

```
1. 尝试从 JAR 中提取并加载 (loadLibraryFromJar)
   └─ 如果成功 → 使用 native 实现
   └─ 如果失败 → 继续步骤 2

2. 尝试从系统库路径加载 (System.loadLibrary)
   └─ 如果成功 → 使用 native 实现
   └─ 如果失败 → 使用 ND4J/Scala 回退
```

### 4. Build 配置

`build.sbt` 已配置：
- `Compile / resourceDirectories` 包含 `src/main/resources`
- `assembly / assemblyMergeStrategy` 处理 `native/` 目录下的文件

## 验证

### 编译并打包

```bash
cd /home/ubuntu/spark-milvus
sbt clean assembly
```

### 验证 JAR 中包含库

```bash
# 检查 JAR 中是否包含 native 库
jar tf target/scala-2.13/spark-connector-assembly-*.jar | grep native/libvectorops_avx512.so
```

### 运行时验证

运行时日志会显示：
- `WARN: vectorops_avx512 native library not available` - 如果加载失败
- 无警告 - 如果加载成功

## 优势

1. **简化部署**：不需要单独上传库文件到 S3
2. **版本一致**：库版本与 JAR 版本绑定
3. **自动回退**：如果 native 库不可用，自动使用 ND4J/Scala

## 注意事项

1. **临时文件**：库会被提取到系统临时目录（`/tmp`），程序退出后自动删除
2. **权限**：代码会自动设置执行权限
3. **平台兼容**：当前只包含 Linux x86_64 版本，如需其他平台需要编译对应版本

## 多平台支持（可选）

如果需要支持多个平台，可以：

1. 编译不同平台的库：
   ```bash
   # Linux x86_64
   ./build.sh
   cp build/libvectorops_avx512.so src/main/resources/native/linux-x86_64/libvectorops_avx512.so
   
   # macOS (需要 macOS 环境)
   # macOS arm64
   # Windows x86_64
   ```

2. 修改 `VectorOpsNative.java` 检测平台并加载对应库

## 当前状态

✅ Native 库已复制到 `src/main/resources/native/`
✅ Java 代码已更新支持从 JAR 加载
✅ Build 配置已正确设置

下一步：重新编译 JAR 包即可使用。
