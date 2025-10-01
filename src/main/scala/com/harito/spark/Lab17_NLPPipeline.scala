package com.harito.spark

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}

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

    // 1. --- Read Dataset ---
    val dataPath = "D:/Hoc_NLP/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000)
    println(s"Successfully read ${initialDF.count()} records.")
    initialDF.printSchema()
    initialDF.show(5, truncate = false)

    // Thêm label giả để huấn luyện Logistic Regression
    val dfWithLabel = initialDF.withColumn("label", length($"text") % 2)

    // 2. --- Tokenization ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // Nếu muốn thử Tokenizer thường thì comment RegexTokenizer và bật dòng này:
    // val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("tokens")

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. --- Feature Extraction ---
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(1000) // yêu cầu 2: giảm vector size

    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features")

    // Nếu muốn thử Word2Vec thì comment HashingTF + IDF và bật đoạn này:
    /*
    val word2Vec = new Word2Vec()
      .setInputCol("filtered_tokens")
      .setOutputCol("features")
      .setVectorSize(100)
      .setMinCount(0)
    */

    // 5. --- Logistic Regression (classification) ---
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)

    // 6. --- Assemble Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, lr))
      // Nếu dùng Word2Vec thay cho TF-IDF thì pipeline như sau:
      // .setStages(Array(tokenizer, stopWordsRemover, word2Vec, lr))

    // --- Fit & Transform ---
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

    spark.stop()
    println("Spark Session stopped.")
  }
}
