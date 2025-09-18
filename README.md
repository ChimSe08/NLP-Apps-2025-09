Lab 2: Count Vectorization
1. Mục tiêu

Xây dựng CountVectorizer để biến văn bản thành vector số học theo phương pháp Bag-of-Words.

Đây là bước tiền xử lý quan trọng trong các bài toán xử lý ngôn ngữ tự nhiên (NLP) và Machine Learning.

2. Các bước triển khai
Bước 1. Xây dựng Vectorizer Interface

File: src/core/interfaces.py

Tạo abstract class Vectorizer với 3 phương thức:

fit(corpus) – học từ vựng từ danh sách văn bản.

transform(documents) – chuyển văn bản thành ma trận đếm.

fit_transform(corpus) – gọi fit và transform cùng lúc.

Bước 2. Xây dựng Tokenizer

File: src/core/tokenizer.py

Class RegexTokenizer:

Sử dụng regex (\w+) để tách từ.

Chuyển tất cả token thành chữ thường (lowercase).

Ví dụ:

text = "I love NLP."
# Kết quả: ["i", "love", "nlp"]

Bước 3. Xây dựng CountVectorizer

File: src/representations/count_vectorizer.py

Class CountVectorizer kế thừa Vectorizer:

Thuộc tính:

tokenizer: Tokenizer để tách từ.

vocabulary_: Lưu từ vựng với key = token, value = index.

Các phương thức:

fit → học từ vựng từ corpus.

transform → chuyển văn bản thành vector đếm.

fit_transform → kết hợp cả hai bước trên.

Bước 4. Tạo file test

File: test/lab2_test.py

Ví dụ corpus:

corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]


Chạy lệnh:

python -m test.lab2_test

3. Kết quả
Vocabulary:
{'ai': 0, 'i': 1, 'is': 2, 'love': 3, 'nlp': 4, 'of': 5, 'programming': 6, 'subfield': 7}

Document-Term Matrix:
[0, 1, 0, 1, 1, 0, 0, 0]
[0, 1, 0, 1, 0, 0, 1, 0]
[1, 0, 1, 0, 1, 1, 0, 1]

4. Khó khăn và cách giải quyết

Trong quá trình thực hiện Lab 2, em đã gặp phải một số khó khăn nhất định.
Trước hết, vấn đề phổ biến nhất là Python không nhận diện thư mục src như một module, dẫn đến lỗi ModuleNotFoundError.
Nguyên nhân là do các thư mục chưa có file __init__.py.
Để khắc phục, em đã thêm các file __init__.py rỗng vào các thư mục src, core và representations, đồng thời chạy chương trình bằng lệnh:

python -m test.lab2_test

Một khó khăn khác là thiếu file tokenizer.py, khiến cho CountVectorizer không thể hoạt động vì không có hàm tách từ.
Giải pháp là tự xây dựng một class RegexTokenizer sử dụng Regular Expressions (regex) để tách các từ trong câu, đồng thời chuyển tất cả về chữ thường nhằm đảm bảo tính thống nhất.
