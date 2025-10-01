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

