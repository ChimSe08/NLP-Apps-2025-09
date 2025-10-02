```
Hoc_NLP/
│── build.sbt                     # SBT build file
│── project/                      
│── src/
│   └── main/
│       ├── resources/
│       │   └── log4j2.properties # Spark logging config
│       └── scala/
│           └── com/
│               └── harito/
│                   └── spark/
│                       └── Lab17_NLPPipeline.scala  # Main Scala source file
│── c4-train.00000-of-01024-30K.json.gz   # Dataset sample (30K docs)
│── README.md
```

## Sample Output

### Pipeline Logs
- Pipeline fitting took **5.76 seconds**
- Data transformation of **1000 records** took **0.83 seconds**
- Actual vocabulary size after preprocessing: **31355 terms**

### Predictions (Top 5)
```
| text                                                                                 | label | prediction | probability                               |
|-------------------------------------------------------------------------------------------------|-------|------------|-------------------------------------------|
| Beginners BBQ Class Taking Place in Missoula!...                                                |   1   |    1.0     | [0.08089365040047715, 0.9191063495995229] |
| Discussion in 'Mac OS X Lion (10.7)' started by axboi87...                                      |   0   |    0.0     | [0.9318268942860257, 0.06817310571397428] |
| Foil plaid lycra and spandex shortall with metallic slinky insets...                            |   1   |    1.0     | [0.3848292760480332, 0.6151707239519668]  |
| How many backlinks per day for new site?...                                                     |   1   |    1.0     | [0.20512775087427473, 0.7948722491257253] |
| The Denver Board of Education opened the 2017-18 school year with an update on projects...  
```
 Implementation Steps
1. Dataset
Dataset: C4 – Colossal Clean Crawled Corpus (subset).

File sử dụng: c4-train.00000-of-01024-30K.json.gz.

Load 1000 mẫu để xử lý nhanh hơn.

2. NLP Pipeline
Pipeline bao gồm các stage:

RegexTokenizer – tách văn bản thành tokens.

StopWordsRemover – loại bỏ stop words.

HashingTF – vector hóa tokens thành feature vector (numFeatures = 1000).

IDF (Inverse Document Frequency) – tính trọng số TF-IDF.

LogisticRegression – mô hình phân loại (nhãn giả label = length(text) % 2).

3. Outputs
Kết quả tiền xử lý và phân loại được lưu trong results/lab17_pipeline_output.txt.

Log hiệu năng được lưu trong log/lab17_metrics.log.

 How to Run
Clone repo:

bash
Sao chép mã
git clone <repo_url>
cd Hoc_NLP
Cài đặt môi trường:

Java 17

SBT (Scala Build Tool)

Apache Spark 3.5+

Chạy chương trình:

bash
Sao chép mã
sbt "runMain com.harito.spark.Lab17_NLPPipeline"


 Results
Pipeline fitting: ~6.36 giây cho 1000 mẫu.

Data transformation: ~0.97 giây.

Vocabulary size sau preprocessing: 31,355 từ.

Vì numFeatures = 1000 nhỏ hơn số vocab → có hash collisions.

Ví dụ kết quả dự đoán:

vbnet
Sao chép mã
text: "Beginners BBQ Class Taking Place in Missoula!..."
label: 1
prediction: 1.0
probability: [0.08, 0.91]
 Difficulties & Solutions
Lỗi đường dẫn: ban đầu Spark báo Path does not exist.
→ Sửa thành đường dẫn tuyệt đối D:/Hoc_NLP/c4-train.00000-of-01024-30K.json.gz.

Lỗi chuỗi trong Scala: viết nhầm ""path"".
→ Sửa thành "path".

Cảnh báo BLAS: Spark không load được thư viện native.
→ Không ảnh hưởng đến kết quả, chỉ giảm tốc độ.

4. Normalization of Count Vectors

Mục đích

Sau khi tính TF-IDF, mỗi văn bản được biểu diễn thành một vector.

Các vector này có độ dài khác nhau, nên cần chuẩn hóa để dễ so sánh.

Normalization đưa mọi vector về cùng độ dài (norm = 1).

Cách thực hiện

Dùng Normalizer trong Spark ML.

Normalizer đảm bảo tất cả các vector sau khi xử lý đều có độ dài bằng 1 (L2 norm).

Code bổ sung
```
val normalizer = new Normalizer()
  .setInputCol("features")
  .setOutputCol("norm_features")
  .setP(2.0)   // L2 normalization
```

Stage này được thêm sau IDF trong pipeline.

Kết quả ví dụ
```
Features: (1000,[5,23,59],[0.32,0.45,0.18])
Norm_Features: (1000,[5,23,59],[0.56,0.78,0.31])
```
5. Cosine Similarity Demo

Mục đích

Lấy một văn bản bất kỳ làm “query”.

Tìm văn bản khác giống nó nhất (hoặc top 10 giống nhất).

Độ đo: Cosine Similarity.

Cách thực hiện

Lấy ngẫu nhiên một document trong DataFrame.

Tính tích vô hướng (dot product) giữa vector chuẩn hóa của document đó với toàn bộ văn bản khác.

Sắp xếp theo cosine similarity giảm dần.

Lấy top 1 hoặc top 10.

Code minh họa
```
// ví dụ chọn 1 văn bản làm query
val sample = transformedDF.limit(1).collect()(0)
val sampleVector = sample.getAs[Vector]("norm_features")

// UDF tính cosine similarity
val dot_udf = udf((v1: Vector, v2: Vector) =>
  v1.asBreeze.dot(v2.asBreeze)
)

val similarities = transformedDF
  .withColumn("cosine_sim", dot_udf(lit(sampleVector), $"norm_features"))
  .orderBy(desc("cosine_sim"))
  .limit(10)

similarities.select("text", "cosine_sim").show(false)

```
Kết quả ví dụ
```
**Sample text:**  
*"Beginners BBQ Class Taking Place in Missoula!"*

### Top 10 most similar documents

Sim=1.0000 | Text: Beginners BBQ Class Taking Place in Missoula!
Do you want to get better at making delicious BBQ? You will have the oppor...

Sim=0.2630 | Text: The latest in Jazz North East∩┐╜s series of Schmazz gigs at the Jazz Caf∩┐╜ featured British guitarists Mike Walker and Stua...

Sim=0.2072 | Text: Unlike many of his peers, Crane is quick to let down his hair [not literally, of course].
You might know Ben Crane best ...

Sim=0.2018 | Text: Sign up to Lineout to stay up to date with all of the latest RUPA news / Thanks for subscribing!
With Rugby World Cup (∩┐╜...

Sim=0.1979 | Text: The results of the NOMAD crowd-sourced data analytics competition with Kaggle are out!
The goal of this competition was ...

Sim=0.1839 | Text: MAMADOU SAKHO remains unavailable for selection for Liverpool's Europa League semi-final first leg clash with Villarreal...

Sim=0.1805 | Text: HOUGHTON ∩┐╜ Whether you∩┐╜re a survivor, family member, caregiver or neighbor, events like th...

Sim=0.1702 | Text: Have you ever been there? It also clarifies that you understand what they are saying.
Even though the language we use to...

Sim=0.1696 | Text: ArtikPix is an app designed to help children improve their speech production at the word and sentence levels.
This app i...

Sim=0.1659 | Text: It was time to pack up camp and head north. Our drive to New Bern/Beaufort area was only 2 hours; allowing ample time to...

```
Ý nghĩa

Cho thấy pipeline có thể dùng không chỉ để phân loại mà còn để tìm kiếm văn bản tương tự.

Đây là một bước mở rộng giúp ứng dụng vào hệ thống gợi ý và tìm kiếm thông tin
