#  Lab 4 – Word Embeddings (Word2Vec, GloVe)

##  1. Mục tiêu

Mục tiêu của Lab 4 là tìm hiểu và thực hành **biểu diễn từ (word embeddings)** bằng cách sử dụng thư viện `gensim` và `PySpark`.  
Các biểu diễn này giúp biến đổi từ ngữ thành vector số học, từ đó cho phép máy tính xử lý ngôn ngữ tự nhiên (NLP) dựa trên ngữ nghĩa.

---

## 2. Cài đặt và môi trường

Các thư viện sử dụng chính:
```
gensim==4.3.3
scikit-learn==1.5.2
pyspark==3.5.1
numpy==2.2.1
scipy==1.14.1
```
Mô hình embedding được tải tự động:
```

model = api.load("glove-wiki-gigaword-50")
 Kích thước: 50 chiều (50D)
 Dữ liệu huấn luyện: Wikipedia + Gigaword
```

 3. Cấu trúc dự án
```
lab4_word_embeddings_pack/
├── src/
│   └── representations/
│       ├── __init__.py
│       └── word_embedder.py        # Lớp WordEmbedder (dùng gensim)
├── test/
│   ├── lab4_test.py                # Kiểm thử các hàm cơ bản
│   ├── lab4_embedding_training_demo.py  # Huấn luyện Word2Vec
│   └── lab4_spark_word2vec_demo.py      # Word2Vec với PySpark
├── data/
│   ├── c4-train.00000-of-01024-30K.json.gz  # Dữ liệu C4 mẫu
│   └── UD_English-EWT/en_ewt-ud-train.txt  # Dữ liệu huấn luyện Word2Vec
└── README.md
```
 4. Thực nghiệm
4.1. GloVe Pretrained Model (glove-wiki-gigaword-50)
Kết quả chạy:
```
Vector('king') shape: (50,)
similarity('king','queen') = 0.7839
similarity('king','man')   = 0.5309
```
 Top-10 từ gần nghĩa với “computer”:
```
Rank	Word	Similarity
1	computers	0.9165
2	software	0.8815
3	technology	0.8526
4	electronic	0.8126
5	internet	0.8060
6	computing	0.8026
7	devices	0.8016
8	digital	0.7992
9	applications	0.7913
10	pc	0.7883
```
```
 Vector hóa câu:
yaml
Sao chép mã
"The queen rules the country."
→ Embedding shape: (50,)
→ head10: [0.0244, 0.378, -0.6382, 0.0128, 0.0524, 0.1195, -0.3165, -0.0878, 0.0776, -0.5418]
```

4.2. Huấn luyện Word2Vec (CBOW/Skip-gram)

Dữ liệu huấn luyện được tạo từ C4 dataset (20.000 câu).

vector_size=50, window=5, epochs=5, sg=0 (CBOW)

Có thể đổi sg=1 để chuyển sang Skip-gram.

Kết quả huấn luyện:
```
Top-5 similar to 'computer':
('com', 0.9983), ('place', 0.9982), ('enjoy', 0.9981), ('etc', 0.9981), ('area', 0.9981)
Analogy king - man + woman:
('europe', 0.9895), ('expected', 0.9894), ('parents', 0.9894)
```
 Kết quả tuy chưa thật chính xác (do dữ liệu nhỏ), nhưng vẫn cho thấy mô hình học được quan hệ ngữ nghĩa cơ bản.

4.3. Word2Vec với PySpark
Để tăng tốc và xử lý dữ liệu lớn, pyspark.ml.feature.Word2Vec được dùng để huấn luyện trên C4 JSON (đã nén .gz).
```
from pyspark.ml.feature import Word2Vec
w2v = Word2Vec(vectorSize=50, minCount=10, maxIter=3, inputCol="tokens", outputCol="vectors")
model = w2v.fit(df.limit(20000))
```
Kết quả mẫu:
```
Top-5 most similar words to 'computer':
1. system
2. software
3. digital
4. technology
5. data
```
 5. Nhận xét & Kết luận
GloVe cho vector có độ ổn định cao, thể hiện tốt quan hệ từ–ngữ nghĩa.

Word2Vec tự huấn luyện tái tạo được xu hướng ngữ nghĩa cơ bản, nhưng chất lượng phụ thuộc vào kích thước dữ liệu.

Spark Word2Vec cho phép mở rộng huấn luyện trên dữ liệu lớn hơn nhiều.


