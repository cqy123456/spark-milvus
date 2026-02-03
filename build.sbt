import scala.sys.process.Process
import scala.io.Source

import xerial.sbt.Sonatype._
import Dependencies._

// Load Sonatype Central credentials
credentials += {
  val credFile = Path.userHome / ".sbt" / "sonatype_central_credentials"
  if (credFile.exists) {
    val lines = Source.fromFile(credFile).getLines().toList
    val props = lines.map { line =>
      val parts = line.split("=", 2)
      if (parts.length == 2) Some(parts(0).trim -> parts(1).trim) else None
    }.flatten.toMap

    Credentials(
      "Sonatype Nexus Repository Manager",
      props.getOrElse("host", "central.sonatype.com"),
      props.getOrElse("user", ""),
      props.getOrElse("password", "")
    )
  } else {
    Credentials(Path.userHome / ".sbt" / "sonatype.credentials")
  }
}

ThisBuild / organizationName := "zilliz"
ThisBuild / organizationHomepage := Some(url("https://zilliz.com/"))
// For cross-compiling (if applicable)
// crossScalaVersions := Seq("2.12.x", "2.13.x")
ThisBuild / scalaVersion := "2.13.14"  // Spark 4.0 requires Scala 2.13
ThisBuild / description := "Milvus Spark Connector to use in Spark ETLs to populate a Milvus vector database."
ThisBuild / versionScheme := Some("early-semver")

// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }
ThisBuild / publishMavenStyle := true

ThisBuild / publishTo := {
  val centralSnapshots =
    "https://central.sonatype.com/repository/maven-snapshots/"
  if (isSnapshot.value) Some("central-snapshots" at centralSnapshots)
  else localStaging.value
}

ThisBuild / licenses := List(
  "Server Side Public License v1" -> new URL(
    "https://raw.githubusercontent.com/mongodb/mongo/refs/heads/master/LICENSE-Community.txt"
  ),
  "GNU Affero General Public License v3 (AGPLv3)" -> new URL(
    "https://www.gnu.org/licenses/agpl-3.0.txt"
  )
)
ThisBuild / homepage := Some(
  url("https://github.com/zilliztech/milvus-spark-connector")
)
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/zilliztech/milvus-spark-connector"),
    "scm:git@github.com:zilliztech/milvus-spark-connector.git"
  )
)
ThisBuild / developers := List(
  Developer(
    id = "simfg",
    name = "SimFG",
    email = "bang.fu@zilliz.com",
    url = url("https://github.com/SimFG")
  )
)

// Task to build native C++ library (AVX-512 vector operations)
lazy val buildNativeLib = taskKey[Unit]("Build native AVX-512 vector operations library")

buildNativeLib := {
  val log = streams.value.log
  val cppDir = baseDirectory.value / "src" / "main" / "cpp"
  val targetDir = baseDirectory.value / "src" / "main" / "resources" / "native"
  val libName = "libvectorops_avx512.so"

  if (!(targetDir / libName).exists()) {
    log.info("Building native AVX-512 vector operations library...")
    val buildScript = cppDir / "build.sh"
    if (buildScript.exists()) {
      val exitCode = Process(Seq("bash", buildScript.getAbsolutePath), cppDir).!
      if (exitCode != 0) {
        log.warn(s"Native library build failed with exit code $exitCode. Will use pure Scala fallback.")
      } else {
        // Copy built library to resources
        val builtLib = cppDir / "build" / libName
        if (builtLib.exists()) {
          IO.copyFile(builtLib, targetDir / libName)
          log.info(s"Native library built and copied to ${targetDir / libName}")
        }
      }
    } else {
      log.warn(s"Build script not found at $buildScript. Will use pure Scala fallback.")
    }
  } else {
    log.info("Native library already exists, skipping build.")
  }
}

lazy val root = (project in file("."))
  .settings(
    name := "spark-connector",
    assembly / parallelExecution := true,
    Test / parallelExecution := true,
    Compile / compile / parallelExecution := true,
    version := "0.2.1-SNAPSHOT",
    organization := "com.zilliz",

    // Fork JVM for run and tests to properly load native libraries
    run / fork := true,
    Test / fork := true,

    // Show test logs immediately (don't buffer)
    Test / logBuffered := false,

    // JVM options for run
    run / javaOptions ++= Seq(
      "-Xss2m",
      "-Djava.library.path=.",
      "--add-opens=java.base/java.nio=ALL-UNNAMED"
    ),

    run / envVars := Map(
      "LD_PRELOAD" -> (baseDirectory.value / s"src/main/resources/native/libmilvus-storage.so").getAbsolutePath
    ),

    // Include test dependencies in run classpath for example applications
    Compile / run / fullClasspath := (Compile / run / fullClasspath).value ++ (Test / fullClasspath).value,

    // JVM options for tests
    Test / javaOptions ++= Seq(
      "-Xss2m",
      "-Xmx4g",
      "-Djava.library.path=.",
      "-Dlog4j2.configurationFile=log4j2.properties",
      "-Dlog4j2.debug=true",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED"
    ),

    Test / envVars := Map(
      "LD_PRELOAD" -> (baseDirectory.value / s"src/main/resources/native/libmilvus-storage.so").getAbsolutePath
    ),

    // Add milvus-storage JNI library as unmanaged dependency
    Compile / unmanagedJars += baseDirectory.value / "milvus-storage" / "java" / "target" / "scala-2.13" / "milvus-storage-jni-test_2.13-0.1.0-SNAPSHOT.jar",
    Test / unmanagedJars += baseDirectory.value / "milvus-storage" / "java" / "target" / "scala-2.13" / "milvus-storage-jni-test_2.13-0.1.0-SNAPSHOT.jar",

    libraryDependencies ++= Seq(
      munit % Test,
      scalaTest % Test,
      grpcNetty,
      scalapbRuntime % "protobuf",
      scalapbRuntimeGrpc,
      scalapbCompilerPlugin,
      sparkCore,
      sparkSql,
      sparkCatalyst,
      sparkMLlib,
      parquetHadoop,
      hadoopCommon,
      hadoopAws,
      awsSdkS3,
      awsSdkS3Transfer,
      awsSdkCore,
      jacksonScala,
      jacksonDatabind,
      arrowFormat,
      arrowVector,
      arrowMemoryCore,
      arrowMemoryNetty,
      arrowCData,
      // graphframes removed - not available for Spark 4.0
      // netlib-java for optimized BLAS (replaces ND4J - no off-heap memory issues)
      netlibJava,
      netlibJavaNativeRef,  // NativeRefBLAS 需要的原生库
      netlibJavaNativeSystem,  // NativeSystemBLAS 需要的原生库（备用）
      breeze
    ),

    Compile / PB.protoSources += baseDirectory.value / "milvus-proto/proto",
    Compile / PB.targets := Seq(
      scalapb.gen(grpc = true) -> (Compile / sourceManaged).value / "scalapb"
    ),
    Compile / unmanagedSourceDirectories += (
      Compile / PB.targets
    ).value.head.outputPath,
    Compile / packageBin / mappings ++= {
      val base = (Compile / PB.targets).value.head.outputPath
      (base ** "*.scala").get.map { file =>
        file -> s"generated_protobuf/${file.relativeTo(base).getOrElse(file)}"
      }
    },
    Compile / resourceDirectories += baseDirectory.value / "src" / "main" / "resources",
    // Build native library before compile and assembly
    Compile / compile := (Compile / compile).dependsOn(buildNativeLib).value,
    assembly := assembly.dependsOn(buildNativeLib).value,
    // 发布 assembly JAR 作为单独的 artifact，带 classifier
    assembly / artifact := {
      val art = (assembly / artifact).value
      art.withClassifier(Some("assembly"))
    },
    addArtifact(assembly / artifact, assembly)
  )

// Filter out problematic files during assembly to avoid shading errors
assembly / assemblyExcludedJars := {
  val cp = (assembly / fullClasspath).value
  // Don't exclude entire JARs, just filter specific files in merge strategy
  cp.filter(_ => false)
}

// Disable shading to avoid Java 21 class file issues during assembly
// The shading process uses ASM bytecode library which cannot read Java 21 class files
// These Java 21 files come from multi-release JARs (Jackson 2.17.x, BouncyCastle, etc.)
// Protobuf conflicts can be resolved via spark.driver.userClassPathFirst=true
assembly / assemblyShadeRules := Seq()

// If shading is needed in the future, upgrade sbt-assembly to version that supports Java 21
// Original shading rules (disabled for now):
// assembly / assemblyShadeRules := Seq(
//   ShadeRule.rename("com.google.protobuf.**" -> "shade_proto.@1").inAll,
//   ShadeRule.rename("com.google.common.**" -> "shade_googlecommon.@1").inAll
//   // Note: Arrow cannot be shaded due to JNI bindings with hardcoded class names
//   // Use spark.driver.userClassPathFirst=true to prioritize our Arrow version
// )

assembly / assemblyMergeStrategy := {
  case PathList("native", xs @ _*) => MergeStrategy.first
  // Handle all Netty native-image files
  case PathList("META-INF", "native-image", "io.netty", _*) =>
    MergeStrategy.discard
  // Handle Netty version properties
  case PathList("META-INF", "io.netty.versions.properties") =>
    MergeStrategy.discard
  // Handle mime.types
  case PathList("mime.types") =>
    MergeStrategy.filterDistinctLines
  // Handle FastDoubleParser notice
  case PathList("META-INF", "FastDoubleParser-NOTICE") =>
    MergeStrategy.discard
  // Handle Arrow git properties
  case PathList("arrow-git.properties") =>
    MergeStrategy.first
  // Handle module-info.class files
  case x if x.endsWith("module-info.class") =>
    MergeStrategy.discard
  // Handle hadoop package-info conflicts
  case PathList("org", "apache", "hadoop", xs @ _*) if xs.last == "package-info.class" =>
    MergeStrategy.first
  // Handle AWS SDK VersionInfo conflicts
  case PathList("software", "amazon", "awssdk", xs @ _*) if xs.last == "VersionInfo.class" =>
    MergeStrategy.first
  // Handle @nowarn annotation conflicts between scala-library and scala-collection-compat
  case PathList("scala", "annotation", "nowarn.class") => MergeStrategy.first
  case PathList("scala", "annotation", "nowarn$.class") => MergeStrategy.first
  // Handle javax.annotation conflicts between jsr305 and other libs
  case PathList("javax", "annotation", xs @ _*) => MergeStrategy.first
  // Discard Java multi-release jar entries (versions 11, 17, 21) to avoid conflicts
  case PathList("META-INF", "versions", _, xs @ _*) => MergeStrategy.first
  // Handle protobuf conflicts
  case PathList("google", "protobuf", xs @ _*) if xs.last.endsWith(".proto") => MergeStrategy.first
  // Handle guava conflicts (publicsuffix classes)
  case PathList("com", "google", "thirdparty", "publicsuffix", xs @ _*) => MergeStrategy.first
  // Handle META-INF native-image conflicts
  case PathList("META-INF", "native-image", _*) => MergeStrategy.first
  // Default case
  case x =>
    val oldStrategy = (ThisBuild / assemblyMergeStrategy).value
    oldStrategy(x)
}

lazy val benchmarks = (project in file("benchmarks"))
  .dependsOn(root % "compile->compile;test->test")
  .settings(
    name := "spark-connector-benchmarks",
    version := "0.2.1-SNAPSHOT",
    organization := "com.zilliz",
    scalaVersion := "2.13.14",  // Match root project for Spark 4.0

    // Fork JVM for run
    run / fork := true,

    // JVM options for benchmarks
    run / javaOptions ++= Seq(
      "-Xss2m",
      "-Djava.library.path=.",
      "--add-opens=java.base/java.nio=ALL-UNNAMED"
    ),

    run / envVars := Map(
      "LD_PRELOAD" -> (baseDirectory.value / ".." / "src" / "main" / "resources" / "native" / "libmilvus-storage.so").getAbsolutePath
    ),

    // Include main project dependencies
    libraryDependencies ++= Seq(
      sparkCore,
      sparkSql,
      sparkMLlib
    ),

    // milvus-storage JNI disabled for Scala 2.13
    // Compile / unmanagedJars += baseDirectory.value / ".." / "milvus-storage" / "java" / "target" / "scala-2.13" / "milvus-storage-jni-test_2.13-0.1.0-SNAPSHOT.jar",

    // Inherit root project's assembly JAR for running benchmarks
    Compile / run / fullClasspath := (Compile / run / fullClasspath).value,

    // Enable assembly for benchmarks subproject
    assembly / assemblyJarName := s"spark-connector-benchmarks-assembly-${version.value}.jar",

    // Use the same merge strategy as root project
    assembly / assemblyMergeStrategy := {
      case PathList("native", xs @ _*) => MergeStrategy.first
      case PathList("META-INF", "native-image", "io.netty", _*) => MergeStrategy.discard
      case PathList("META-INF", "io.netty.versions.properties") => MergeStrategy.discard
      case PathList("mime.types") => MergeStrategy.filterDistinctLines
      case PathList("META-INF", "FastDoubleParser-NOTICE") => MergeStrategy.discard
      case PathList("arrow-git.properties") => MergeStrategy.first
      case x if x.endsWith("module-info.class") => MergeStrategy.discard
      case PathList("org", "apache", "hadoop", xs @ _*) if xs.last == "package-info.class" => MergeStrategy.first
      case PathList("software", "amazon", "awssdk", xs @ _*) if xs.last == "VersionInfo.class" => MergeStrategy.first
      case PathList("scala", "annotation", "nowarn.class") => MergeStrategy.first
      case PathList("scala", "annotation", "nowarn$.class") => MergeStrategy.first
      case PathList("javax", "annotation", xs @ _*) => MergeStrategy.first
      case PathList("META-INF", "versions", _, xs @ _*) => MergeStrategy.first
      case PathList("google", "protobuf", xs @ _*) if xs.last.endsWith(".proto") => MergeStrategy.first
      case PathList("com", "google", "thirdparty", "publicsuffix", xs @ _*) => MergeStrategy.first
      case PathList("META-INF", "native-image", _*) => MergeStrategy.first
      case x =>
        val oldStrategy = (assembly / assemblyMergeStrategy).value
        oldStrategy(x)
    },

    // Disable shading for benchmarks too
    assembly / assemblyShadeRules := Seq()
  )

// import scalapb.compiler.Version
// val grpcJavaVersion =
//   SettingKey[String]("grpcJavaVersion", "ScalaPB gRPC Java version")
// grpcJavaVersion := Version.grpcJavaVersion

// See https://www.scala-sbt.org/1.x/docs/Using-Sonatype.html for instructions on how to publish to Sonatype.
