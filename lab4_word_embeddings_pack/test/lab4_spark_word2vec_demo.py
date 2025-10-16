from __future__ import annotations
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec

INPUT_JSON = 'data/c4-train.00000-of-01024-30K.json'

def main():
    spark = SparkSession.builder.appName("Lab4SparkWord2Vec").getOrCreate()
    try:
        df = spark.read.json(INPUT_JSON)
    except Exception as e:
        print(f"[ERROR] Cannot read {INPUT_JSON}: {e}")
        spark.stop()
        return

    df = (
        df.select(lower(col('text')).alias('text'))
          .na.drop(subset=['text'])
          .withColumn('text', regexp_replace('text', r"[^a-z0-9\s']", ' '))
          .withColumn('tokens', split(col('text'), r"\s+"))
    )

    w2v = Word2Vec(vectorSize=100, minCount=5, inputCol='tokens', outputCol='features')
    model = w2v.fit(df)

    try:
        synonyms = model.findSynonyms('computer', 5)
        synonyms.show(truncate=False)
    except Exception as e:
        print("Synonyms demo skipped:", e)

    spark.stop()

if __name__ == '__main__':
    main()
