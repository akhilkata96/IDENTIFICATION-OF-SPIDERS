package com.bdm.spark_scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import java.lang.String
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.elasticsearch.spark._
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
object sqls {

  def main(args: Array[String])
  {
    val spark = SparkSession.builder().appName("spark example").master("local").config("spark_option","value").getOrCreate()
    val df =spark.read.csv("/home/akhil/Downloads/dataseti.csv")
   df.show()
    df.select("_c4").show()
 // val res = df.select("_c4", "_c2", unix_timestamp("_c2","yyyy/MM/dd HH:mm:ss").cast(TimestampType).as("timestamp"))
 df.createOrReplaceTempView("data")
 val sqlDF = spark.sql("SELECT * FROM data WHERE _c4 LIKE '%bot%'")
 sqlDF.createOrReplaceTempView("bot")
  val sqlDF2 = spark.sql("SELECT * FROM data WHERE _c4 NOT LIKE '%bot%'")

 val d1 = spark.sql("select _c2 FROM bot GROUP BY _c2 having count(*) >1")

  d1.show()

}

}
