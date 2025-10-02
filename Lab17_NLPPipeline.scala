package com.harito.spark

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
import org.apache.spark.ml.linalg.Vector

object Lab17_NLPPipeline {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    Thread.sleep(3000)

    // --- NEW: dễ chỉnh số lượng document ---
    val limitDocuments = 1000   // chỉnh con số này nếu muốn xử lý nhiều/ít hơn
    val dataPath = "D:/Hoc_NLP/c4-train.00000-of-01024-30K.json.gz"

    val readStart = System.nanoTime()
    val initialDF = spark.read.json(dataPath).limit(limitDocuments)
    val readDuration = (System.nanoTime() - readStart) / 1e9d

    println(f"Successfully read ${initialDF.count()} records in $readDuration%.2f seconds.")
    initialDF.printSchema()
    initialDF.show(5, truncate = false)

    val dfWithLabel = initialDF.withColumn("label", length($"text") % 2)

    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(1000)

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("tfidf_features")

    // --- Normalizer ---
    val normalizer = new Normalizer()
      .setInputCol("tfidf_features")
      .setOutputCol("features")
      .setP(2.0)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer, lr))

    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(dfWithLabel)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(dfWithLabel).cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Tính vocab size
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after preprocessing: $actualVocabSize terms.")

    // --- Show results ---
    transformedDF.select("text", "label", "prediction", "probability").show(5, truncate = false)

    // --- Write Metrics ---
    val log_path = "results/lab17_metrics.log"   // đổi để ghi ngay trong thư mục dự án
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Read data duration: $readDuration%.2f seconds")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Vocabulary size: $actualVocabSize")
      logWriter.println(s"HashingTF numFeatures set to: 1000")
    } finally logWriter.close()

    // --- Write Output ---
    val result_path = "results/lab17_pipeline_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println("--- NLP Pipeline Output (Sample Results) ---")
      transformedDF.select("text", "label", "prediction", "probability", "features")
        .take(20)
        .foreach { row =>
          resultWriter.println("="*80)
          resultWriter.println(s"Text: ${row.getAs[String]("text").take(100)}...")
          resultWriter.println(s"Label: ${row.getAs[Double]("label")}")
          resultWriter.println(s"Prediction: ${row.getAs[Double]("prediction")}")
          resultWriter.println(s"Probability: ${row.getAs[Vector]("probability")}")
          resultWriter.println(s"Features: ${row.getAs[Vector]("features")}")
        }
    } finally resultWriter.close()

    // --- Cosine Similarity Demo ---
    println("\n--- Cosine Similarity Demo ---")

    val sampleRow = transformedDF.limit(1).collect()(0)
    val sampleText = sampleRow.getAs[String]("text")
    val sampleVec = sampleRow.getAs[Vector]("features")

    def cosineSim(v1: Vector, v2: Vector): Double = {
      val arr1 = v1.toArray
      val arr2 = v2.toArray
      val dot = arr1.zip(arr2).map { case (a, b) => a * b }.sum
      val norm1 = math.sqrt(arr1.map(x => x * x).sum)
      val norm2 = math.sqrt(arr2.map(x => x * x).sum)
      if (norm1 == 0.0 || norm2 == 0.0) 0.0 else dot / (norm1 * norm2)
    }

    val sims = transformedDF.rdd.map { row =>
      val txt = row.getAs[String]("text")
      val vec = row.getAs[Vector]("features")
      val sim = cosineSim(sampleVec, vec)
      (txt, sim)
    }

    val top5 = sims.top(5)(Ordering.by(_._2))

    println(s"\nSample text: ${sampleText.take(200)}...\n")
    println("Top 5 most similar documents:")
    top5.foreach { case (txt, sim) =>
      println(f"Sim=$sim%.4f | Text: ${txt.take(120)}...")
    }

    // --- Giữ Spark UI chạy vô thời hạn ---
    println("\nSpark job đã chạy xong.")
    println("Spark UI đang mở, truy cập tại http://localhost:4040 (hoặc 4041, 4042...)")
    println("Nhấn Ctrl+C trong terminal để dừng Spark.")

    Thread.sleep(Long.MaxValue)

    spark.stop()
    println("Spark Session stopped.")
  }
}
