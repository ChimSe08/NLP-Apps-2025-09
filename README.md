```
Hoc_NLP/
│── build.sbt                     # SBT build file
│── project/
│── results/
│   └── lab17_pipeline_output.txt/                        
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

Tokenization

Sau khi tách từ bằng RegexTokenizer:
```
+---+--------------------------------------------------------------------------------+
| id|                                                                           words|
+---+--------------------------------------------------------------------------------+
|  0|[beginners, bbq, class, taking, place, in, missoula!, do, you, want, to, get,...|
|  1|[discussion, in, mac, os, x, lion, (10, 7), started, by, axboi87, jan, 20, 20...|
|  2|[foil, plaid, lycra, and, spandex, shortall, with, metallic, slinky, insets, ...|
+---+--------------------------------------------------------------------------------+
only showing top 3 rows
```

 Stage [Tokenization] completed in ~109 ms

Stopword Removal

Sau khi loại bỏ stop words:
```
+---+--------------------------------------------------------------------------------+
| id|                                                                        filtered|
+---+--------------------------------------------------------------------------------+
|  0|[beginners, bbq, class, taking, place, missoula!, want, get, better, making, ...|
|  1|[discussion, mac, os, x, lion, (10, 7), started, axboi87, jan, 20, 2012, ve, ...|
|  2|[foil, plaid, lycra, spandex, shortall, metallic, slinky, insets, attached, m...|
+---+--------------------------------------------------------------------------------+
only showing top 3 rows
```

 Stage [Stopword Removal] completed in ~91 ms

Vectorization + Training

Quá trình CountVectorizer + IDF + Normalizer hoàn tất.
 Stage [Vectorization + Training] completed in ~2622 ms
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
```
 # Lab 17: Spark NLP Pipeline with Cosine Similarity

## Implementation Steps
Pipeline được cài đặt theo các bước:
1. **Dataset**  
   - Nguồn: C4 – Colossal Clean Crawled Corpus (subset).  
   - File: `c4-train.00000-of-01024-30K.json.gz`.  
   - Load 1000 mẫu để giảm thời gian chạy (có biến `limitDocuments` để chỉnh).  

2. **NLP Pipeline Stages**
   - `RegexTokenizer`: tách text thành tokens.  
   - `StopWordsRemover`: loại bỏ stop words.  
   - `CountVectorizer`: biến tokens thành vector.  
   - `IDF`: tính trọng số TF-IDF.  
   - `Normalizer`: chuẩn hóa vector TF-IDF về norm = 1.  
   - (Mở rộng): LogisticRegression thử nghiệm phân loại với label giả.  

3. **Cosine Similarity**
   - Chọn một văn bản làm query.  
   - Tính cosine similarity với tất cả văn bản khác.  
   - In ra Top 5 văn bản tương tự nhất.  

---

## How to Run
1. Clone repo:
   ```bash
   git clone <repo_url>
   cd Hoc_NLP
```
2. Cài đặt môi trường:

Java 17

SBT 1.11.6+

Apache Spark 3.5+

3. Chạy chương trình:

bash
Sao chép mã
sbt "runMain com.harito.spark.Lab17_NLPPipeline"

4. Kết quả:

Console hiển thị log + thời gian từng stage (Read Data, Tokenization, Stopword Removal, Vectorization, Cosine Similarity, Save).

File kết quả: results/lab17_pipeline_output.txt.

Results
Pipeline fitting: ~6s cho 1000 mẫu.

Data transformation: ~1s.

Vocabulary size: ~31K terms → bị hash collisions khi numFeatures=1000.

Ví dụ kết quả cosine similarity:

csharp
Sao chép mã
Top 5 most similar documents:
[54] Sim=0.4666 | Know Buckeye Trail Class of 2001 graduates...
[311] Sim=0.2786 | UChicago chose a Class Day speaker...
[278] Sim=0.2593 | Class A Burn Prop - Stove Simulator...
...
Ý nghĩa: Document "BBQ Class" có similarity cao với các văn bản khác chứa từ “class” → pipeline hoạt động đúng.

Difficulties & Solutions
Path error: ban đầu Spark báo “Path does not exist” → sửa thành đường dẫn tuyệt đối D:/Hoc_NLP/....

Scala string error: nhầm dấu ""path"" → sửa thành "path".

BLAS warning: Spark không load thư viện native → chỉ ảnh hưởng tốc độ, không ảnh hưởng kết quả.

sbt server lock error (ServerAlreadyBootingException) → giải quyết bằng cách chọn y để tạo server mới khi sbt hỏi.

Normalization
Sau TF-IDF, các vector có độ dài khác nhau.

Normalizer đảm bảo tất cả vector có norm = 1 → dễ so sánh cosine similarity.

Ví dụ:
```
makefile
Sao chép mã
TF-IDF:  (1000,[5,23,59],[0.32,0.45,0.18])
NormVec: (1000,[5,23,59],[0.56,0.78,0.31])
```
References

Apache Spark MLlib Documentation

Spark NLP Lab Instructions

C4 Dataset (Colossal Clean Crawled Corpus)
