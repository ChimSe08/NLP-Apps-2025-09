package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}

object Lab17_CombinedNLP {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Combined with Timer + Normalizer")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    import spark.implicits._

    def time[R](stage: String)(block: => R): R = {
      val start = System.nanoTime()
      val result = block
      val end = System.nanoTime()
      val durationMs = (end - start) / 1e6
      println(f"⏱ Stage [$stage] completed in $durationMs%.2f ms")
      result
    }

    val limitDocuments = 1000
    val dataPath = "D:/Hoc_NLP/c4-train.00000-of-01024-30K.json.gz"

    val df = time("Read Data") {
      spark.read.json(dataPath).limit(limitDocuments)
    }
    println(s" Loaded ${df.count()} documents from $dataPath")
    df.show(3, truncate = 100)

    val dfWithId = df.withColumn("id", monotonically_increasing_id())

    val wordsData = time("Tokenization") {
      val tokenizer = new RegexTokenizer()
        .setInputCol("text")
        .setOutputCol("words")
        .setPattern("[\\s/\"';:<>,.?\\|\\[\\]{}`~\\\\]+")
        .setToLowercase(true)
      tokenizer.transform(dfWithId)
    }

    println("\n Tokenized sample:")
    wordsData.select("id", "words").show(3, truncate = 80)

    val filteredData = time("Stopword Removal") {
      val stopWordsRemover = new StopWordsRemover()
        .setInputCol("words")
        .setOutputCol("filtered")
      stopWordsRemover.transform(wordsData)
    }

    println("\n After stopword removal:")
    filteredData.select("id", "filtered").show(3, truncate = 80)

    val (tfidfData, cvModel, idfModel) = time("Vectorization + Training") {
      val cv = new CountVectorizer()
        .setInputCol("filtered")
        .setOutputCol("features")
        .setVocabSize(1000)
        .setMinDF(5)

      val cvModel = cv.fit(filteredData)
      val featurizedData = cvModel.transform(filteredData)

      val idf = new IDF()
        .setInputCol("features")
        .setOutputCol("tfidf")
      val idfModel = idf.fit(featurizedData)
      val tfidfData = idfModel.transform(featurizedData)

      val normalizedData = new Normalizer()
        .setInputCol("tfidf")
        .setOutputCol("normFeatures")
        .setP(2.0)
        .transform(tfidfData)
        .cache()

      (normalizedData, cvModel, idfModel)
    }

    val queryId = 0
    val queryRow = tfidfData.filter($"id" === queryId).head()
    val queryText = queryRow.getAs[String]("text")
    val queryVector = queryRow.getAs[Vector]("normFeatures")

    def cosineSim(v1: Vector, v2: Vector): Double = {
      val a1 = v1.toArray
      val a2 = v2.toArray
      val dot = a1.zip(a2).map { case (x, y) => x * y }.sum
      val n1 = math.sqrt(a1.map(x => x * x).sum)
      val n2 = math.sqrt(a2.map(x => x * x).sum)
      if (n1 == 0.0 || n2 == 0.0) 0.0 else dot / (n1 * n2)
    }

    val similarities = time("Cosine Similarity Computation") {
      tfidfData.select("id", "normFeatures", "text")
        .as[(Long, Vector, String)]
        .map { case (id, vec, txt) => (id, txt, cosineSim(queryVector, vec)) }
        .collect()
        .filter(_._1 != queryId)
    }

    val top5 = similarities.sortBy(-_._3).take(5)

    println("\n Query text sample:")
    println(queryText.take(200) + "...\n")
    println(" Top 5 most similar documents:")
    top5.foreach { case (id, txt, sim) =>
      println(f"[$id] Sim=$sim%.4f | ${txt.take(120)}...")
    }

    time("Save Results") {
      val resultPath = "results/lab17_combined_output.txt"
      new File(resultPath).getParentFile.mkdirs()
      val writer = new PrintWriter(new File(resultPath))
      try {
        writer.println("=== NLP Pipeline Combined Output ===")
        writer.println(s"Query text: ${queryText.take(200)}...\n")
        top5.foreach { case (id, txt, sim) =>
          writer.println("=" * 80)
          writer.println(s"ID=$id")
          writer.println(f"Similarity=$sim%.4f")
          writer.println(s"Text: ${txt.take(200)}...")
        }
      } finally writer.close()
      println(s"\n Kết quả đã ghi ra file: $resultPath")
    }

    spark.stop()
    println(" Spark Session stopped.")
  }
}
