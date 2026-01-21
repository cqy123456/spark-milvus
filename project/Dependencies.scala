import sbt._

object Dependencies {
  // Version constants
  lazy val scalapbVersion = "1.0.0-alpha.1"  // Use 1.0.x for Scala 2.13 compatibility
  // Spark 4.0.1 requires Java 17+ and Scala 2.13
  lazy val sparkVersion = "4.0.0"  // Use Spark 4.0.0 for K8s cluster (4.0.1 not yet released to Maven)
  lazy val grpcJavaVersion = "1.37.0"
  lazy val parquetVersion = "1.14.1"  // Updated for Spark 4.0
  lazy val hadoopVersion = "3.4.1"  // Spark 4.0 uses Hadoop 3.4.x
  lazy val jacksonVersion = "2.17.3"

  lazy val munit = "org.scalameta" %% "munit" % "0.7.29"
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.2.15"
  lazy val grpcNetty =
    "io.grpc" % "grpc-netty-shaded" % grpcJavaVersion excludeAll ExclusionRule(
      organization = "org.slf4j"
    )
  lazy val scalapbRuntime =
    "com.thesamet.scalapb" %% "scalapb-runtime" % scalapbVersion
  lazy val scalapbRuntimeGrpc =
    "com.thesamet.scalapb" %% "scalapb-runtime-grpc" % scalapbVersion
  lazy val scalapbCompilerPlugin =
    "com.thesamet.scalapb" %% "compilerplugin" % scalapbVersion
  lazy val sparkCore =
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided,test" excludeAll(
      ExclusionRule(organization = "org.apache.arrow")
    )
  lazy val sparkSql =
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided,test" excludeAll(
      ExclusionRule(organization = "org.apache.arrow")
    )
  lazy val sparkCatalyst =
    "org.apache.spark" %% "spark-catalyst" % sparkVersion % "provided,test" excludeAll(
      ExclusionRule(organization = "org.apache.arrow")
    )
  lazy val sparkMLlib =
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided,test" excludeAll(
      ExclusionRule(organization = "org.apache.arrow")
    )
  lazy val parquetHadoop =
    "org.apache.parquet" % "parquet-hadoop" % parquetVersion
  lazy val hadoopCommon =
    "org.apache.hadoop" % "hadoop-common" % hadoopVersion exclude ("javax.activation", "activation")
  lazy val hadoopAws =
    "org.apache.hadoop" % "hadoop-aws" % hadoopVersion exclude("software.amazon.awssdk", "bundle")
  lazy val awsSdkS3 =
    "software.amazon.awssdk" % "s3" % "2.30.38" // doc: https://javadoc.io/doc/software.amazon.awssdk/s3/2.30.38/index.html
  lazy val awsSdkS3Transfer = 
    "software.amazon.awssdk" % "s3-transfer-manager" % "2.30.38"
  lazy val awsSdkCore =
    "com.amazonaws" % "aws-java-sdk-core" % "1.12.780"
  lazy val jacksonScala =
    "com.fasterxml.jackson.module" %% "jackson-module-scala" % jacksonVersion
  lazy val jacksonDatabind =
    "com.fasterxml.jackson.core" % "jackson-databind" % jacksonVersion

  // Arrow dependencies for milvus-storage JNI
  lazy val arrowVersion = "17.0.0"
  lazy val arrowFormat = "org.apache.arrow" % "arrow-format" % arrowVersion
  lazy val arrowVector = "org.apache.arrow" % "arrow-vector" % arrowVersion
  lazy val arrowMemoryCore = "org.apache.arrow" % "arrow-memory-core" % arrowVersion
  lazy val arrowMemoryNetty = "org.apache.arrow" % "arrow-memory-netty" % arrowVersion
  lazy val arrowCData = "org.apache.arrow" % "arrow-c-data" % arrowVersion

  // HDF5 for reading ANN benchmark datasets
  // Using CISD JHDF5 from SciJava repository
  lazy val hdf5 = "cisd" % "jhdf5" % "19.04.1"

  // GraphFrames for graph processing
  // Note: GraphFrames not yet available for Spark 4.0, disabled for now
  // lazy val graphframes = "graphframes" % "graphframes" % "0.8.4-spark3.5-s_2.12"

  // ND4J for optimized linear algebra and vector operations
  // nd4j-native-platform: CPU backend with AVX/AVX2/AVX-512 optimizations
  // For GPU support, replace with nd4j-cuda-11.8-platform (requires CUDA 11.8+)
  lazy val nd4jVersion = "1.0.0-M2.1"
  lazy val nd4jNative = "org.nd4j" % "nd4j-native-platform" % nd4jVersion
  // lazy val nd4jCuda = "org.nd4j" % "nd4j-cuda-11.8-platform" % nd4jVersion  // Uncomment for GPU support
}
