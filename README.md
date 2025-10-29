Mục tiêu

Xây dựng một pipeline phân loại văn bản hoàn chỉnh, từ dữ liệu thô đến mô hình học máy đã được huấn luyện, sử dụng các kỹ thuật tách từ (tokenization) và vector hóa (vectorization) đã học ở các bài lab trước.

Cơ sở lý thuyết
Phân loại văn bản (Text Classification) là quá trình gán nhãn hoặc danh mục cho các tài liệu văn bản.
Các ứng dụng phổ biến gồm có:
Phân tích cảm xúc (sentiment analysis)
Phát hiện thư rác (spam detection)

Gán chủ đề (topic labeling)
1) Cấu trúc thư mục dự án
```
lab5_project/
│
├── src/                    # (chứa class xử lý văn bản, vectorizer, classifier)
│   ├── text_classifier.py
│   ├── tfidf_vectorizer.py
│   ├── regex_tokenizer.py
│   └── __init__.py
│
├── test/
│   ├── lab5_baseline.py
│   ├── lab5_improvement_nb.py
│   ├── lab5_improvement_linearsvc.py
│   ├── lab5_improvement_cleaning.py
│   └── lab5_spark_sentiment_analysis.py
│
└── requirements.txt        # Liệt kê thư viện cần thiết
```
2) Dữ liệu sử dụng
| Label | Nội dung                                 |
| ----: | ---------------------------------------- |
|     1 | This movie is fantastic and I love it!   |
|     0 | I hate this film, it's terrible.         |
|     1 | The acting was superb…                   |
|     0 | What a waste of time, absolutely boring. |
|     1 | Highly recommend this, a masterpiece.    |
|     0 | Could not finish watching, so bad.       |

3) Các bước triển khai
```
| Bước | Mô tả                                           | File thực hiện                             |
| ---- | ----------------------------------------------- | ------------------------------------------ |
| 1️⃣  | Chuẩn bị dataset & xử lý chuỗi                  | trong tất cả script                        |
| 2️⃣  | Baseline: TF-IDF (1-gram) + Logistic Regression | `lab5_baseline.py`                         |
| 3️⃣  | Cải tiến mô hình: NB, LinearSVC                 | `lab5_improvement_nb.py` / `_linearsvc.py` |
| 4️⃣  | Cleaning nâng cao → TF-IDF 1-2gram + LogReg     | `lab5_improvement_cleaning.py`             |
| 5️⃣  | Spark ML Pipeline demo                          | `lab5_spark_sentiment_analysis.py`         |
```
4) Báo cáo các chỉ số hiệu suất của mô hình Logistic Regression cơ bản

Mô hình: Baseline
Phương pháp: TF-IDF (unigram) + Logistic Regression
Dữ liệu training nhỏ, cân bằng
| Metric   | Giá trị      |
| -------- | ------------ |
| Accuracy | **1.0000 ** |
| F1-score | **1.0000 ** |
| ROC-AUC  | 1.0000      |
Nhận xét:

Mô hình dự đoán đúng toàn bộ trên tập test
Lý do hiệu suất quá cao: dataset quá nhỏ → dễ thuộc lòng dữ liệu
Đây là một dạng overfitting tích cực (không có sample đa dạng)

5) Báo cáo kết quả của các mô hình cải tiến (Improvement Models)
```
| Mô hình                                       | Accuracy |      F1-score     | Nhận xét nhanh            |
| --------------------------------------------- | :------: | :---------------: | ------------------------- |
| MultinomialNB + TFIDF 1-2gram                 |   0.50   |       0.00       | NB dự đoán toàn 1 class   |
| LinearSVC + TFIDF 1-2gram                     |   0.50   | 0.00 → 0.22 CV ⚠ | Học chưa ổn định          |
| Cleaning + TFIDF 1-2gram + LogisticRegression |   0.50   |    **0.6667 **   | Hiệu quả cải thiện rõ rệt |
| Spark ML LogisticRegression                   |   0.50   |         —         | Chỉ mang tính demo        |
```
Highlight:
Cleaning + Logistic Regression là mô hình cải tiến tốt nhất
NB & SVC không cải thiện do dữ liệu nhỏ → mô hình khó phân tách rõ ràng

6) Khó khăn và hướng giải quyết
```
| Khó khăn gặp phải                                                 | Nguyên nhân kỹ thuật                                                 | Cách khắc phục đã áp dụng                                                                                          |
| ----------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Spark báo lỗi **“The system cannot find the path specified”**   | PySpark không tìm được biến môi trường Spark/Java trên Windows       | Cài **Java JDK 17**, đặt biến môi trường `JAVA_HOME`, thêm vào PATH, và set `PYSPARK_PYTHON` bằng Python đang dùng |
| Không import được module (ModuleNotFoundError: regex_tokenizer) | Bỏ thư mục `src/` và chạy trực tiếp từng file → đường dẫn import sai | Tích hợp lại code vào từng file test → **bỏ hoàn toàn import từ src**                                              |
| Cross-Validation bị lỗi “n_splits > samples per class”          | Dữ liệu chỉ có 6 mẫu (3 mẫu mỗi nhãn) → 5-fold không thể chia đều    | Giảm về **KFold = 3**, vẫn đảm bảo stratified                                                                      |
| F1-score = 0 với MultinomialNB / SVC                            | Bộ dữ liệu quá bé → mô hình dự đoán 1 lớp duy nhất                   | Ghi rõ phân tích và chuyển sang Logistic Regression + Cleaning để cải thiện                                        |
| Accuracy Spark thấp                                             | Tập test chỉ có **2 mẫu**                                            | Chỉ trình bày Spark ở mức **demo pipeline**, không đưa vào so sánh hiệu năng                                       |
| Overfitting ở Logistic Regression baseline                      | Với TF-IDF + data bé → mô hình nhớ toàn bộ dữ liệu                   | Khuyến nghị tăng dataset (IMDB 50k) trong hướng phát triển                                                         |
```
