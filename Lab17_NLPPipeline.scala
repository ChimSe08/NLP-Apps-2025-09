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

    val dataPath = "D:/Hoc_NLP/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000)
    println(s"Successfully read ${initialDF.count()} records.")
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

    // --- NEW: Normalizer ---
    val normalizer = new Normalizer()
      .setInputCol("tfidf_features")
      .setOutputCol("features") // Logistic Regression dùng cột này
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
    val log_path = "../log/lab17_metrics.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Vocabulary size: $actualVocabSize")
      logWriter.println(s"HashingTF numFeatures set to: 1000")
    } finally logWriter.close()

    // --- Write Output ---
    val result_path = "../results/lab17_pipeline_output.txt"
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
          resultWriter.println(s"Probability: ${row.getAs[org.apache.spark.ml.linalg.Vector]("probability")}")
          resultWriter.println(s"Features: ${row.getAs[org.apache.spark.ml.linalg.Vector]("features")}")
        }
    } finally resultWriter.close()

    // --- NEW: Cosine Similarity Demo ---
    println("\n--- Cosine Similarity Demo ---")

    // Lấy 1 văn bản bất kỳ (vd: document đầu tiên)
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

    val top10 = sims.top(10)(Ordering.by(_._2))

    println(s"\nSample text: ${sampleText.take(200)}...\n")
    println("Top 10 most similar documents:")
    top10.foreach { case (txt, sim) =>
      println(f"Sim=$sim%.4f | Text: ${txt.take(120)}...")
    }

    spark.stop()
    println("Spark Session stopped.")
  }
}
