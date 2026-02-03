package com.zilliz.spark.connector.utils

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
 * Configuration for VectorOps operations
 * 
 * Supports reading from multiple sources in priority order:
 * 1. Spark configuration (spark.conf)
 * 2. System properties
 * 3. Environment variables
 * 4. Default values
 */
/**
 * VectorOps implementation strategy
 */
object ImplementationStrategy extends Enumeration {
  val Native, Scala = Value  // ND4J removed, only Native and Scala fallback
}

case class VectorOpsConfig(
  preferNative: Boolean = true  // Prefer native AVX-512 implementation if available
) {
  /**
   * Get configuration value with fallback chain
   */
  private def getConfigValue(
    sparkConf: Option[String],
    systemProp: String,
    envVar: String,
    defaultValue: String
  ): String = {
    sparkConf
      .orElse(Option(System.getProperty(systemProp)))
      .orElse(Option(System.getenv(envVar)))
      .getOrElse(defaultValue)
  }

  /**
   * Get boolean configuration value
   */
  private def getBooleanConfig(
    sparkConf: Option[String],
    systemProp: String,
    envVar: String,
    defaultValue: Boolean
  ): Boolean = {
    val value = getConfigValue(sparkConf, systemProp, envVar, defaultValue.toString)
    java.lang.Boolean.parseBoolean(value)
  }
}

object VectorOpsConfig {
  
  /**
   * Configuration keys
   */
  object Keys {
    val PREFER_NATIVE = "spark.zilliz.vectorops.preferNative"
    
    // System property keys (without spark. prefix)
    val SYS_PREFER_NATIVE = "com.zilliz.spark.vectorops.preferNative"
    
    // Environment variable keys
    val ENV_PREFER_NATIVE = "ZILLIZ_VECTOROPS_PREFER_NATIVE"
  }

  /**
   * Get configuration from SparkSession
   * Priority: Spark conf > System property > Environment variable > Default
   */
  def fromSparkSession(spark: SparkSession): VectorOpsConfig = {
    val sparkConf = spark.sparkContext.getConf
    
    VectorOpsConfig(
      preferNative = getBooleanConfig(
        sparkConf.getOption(Keys.PREFER_NATIVE),
        Keys.SYS_PREFER_NATIVE,
        Keys.ENV_PREFER_NATIVE,
        defaultValue = true
      )
    )
  }

  /**
   * Get configuration from SparkContext
   */
  def fromSparkContext(sc: SparkContext): VectorOpsConfig = {
    val sparkConf = sc.getConf
    
    VectorOpsConfig(
      preferNative = getBooleanConfig(
        sparkConf.getOption(Keys.PREFER_NATIVE),
        Keys.SYS_PREFER_NATIVE,
        Keys.ENV_PREFER_NATIVE,
        defaultValue = true
      )
    )
  }

  /**
   * Get configuration from system properties and environment variables only
   * (fallback when Spark context is not available)
   */
  def fromSystem(): VectorOpsConfig = {
    VectorOpsConfig(
      preferNative = getBooleanConfig(
        None,
        Keys.SYS_PREFER_NATIVE,
        Keys.ENV_PREFER_NATIVE,
        defaultValue = true
      )
    )
  }

  /**
   * Helper method to get boolean config
   */
  private def getBooleanConfig(
    sparkConf: Option[String],
    systemProp: String,
    envVar: String,
    defaultValue: Boolean
  ): Boolean = {
    val value = getConfigValue(sparkConf, systemProp, envVar, defaultValue.toString)
    java.lang.Boolean.parseBoolean(value)
  }

  /**
   * Helper method to get config value with priority
   */
  private def getConfigValue(
    sparkConf: Option[String],
    systemProp: String,
    envVar: String,
    defaultValue: String
  ): String = {
    sparkConf
      .orElse(Option(System.getProperty(systemProp)))
      .orElse(Option(System.getenv(envVar)))
      .getOrElse(defaultValue)
  }
}
