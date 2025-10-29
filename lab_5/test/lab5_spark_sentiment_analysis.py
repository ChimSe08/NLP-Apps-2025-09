import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def load_data(spark):
    data = [
        ("I loved this movie, it was amazing!", 1),
        ("Fantastic acting and story!", 1),
        ("Very boring and too long.", 0),
        ("Terrible movie, waste of time.", 0),
    ]
    return spark.createDataFrame(data, ["text", "label"])

def main():
    spark = SparkSession.builder.master("local[*]").appName("Spark Sentiment").getOrCreate()

    df = load_data(spark)
    train, test = df.randomSplit([0.7, 0.3], seed=42)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    lr = LogisticRegression(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])
    model = pipeline.fit(train)
    preds = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )

    print("\n=== Spark Sentiment Result ===")
    print("✅ Accuracy:", evaluator.evaluate(preds))

    spark.stop()

if __name__ == "__main__":
    main()
