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
Mở Spark UI tại http://localhost:4040 để theo dõi.

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

