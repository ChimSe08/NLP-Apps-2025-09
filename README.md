```
lab6_dependency_parsing/
│── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── visualize_dep.py
│   ├── inspect_tree.py
│   ├── extract_relations.py
│   ├── exercises.py
│   └── main.py
│
│── notebook/
│   └── lab6_dependency_parsing.ipynb   
│
│── test/
│   ├── test_exercises.py
│   └── test_imports.py
│
│── report/                            
│
│── data/
│                    
│
│── README.md

```

1. Giới thiệu

Phân tích cú pháp phụ thuộc (Dependency Parsing) là một kỹ thuật trong xử lý ngôn ngữ tự nhiên giúp xác định cấu trúc ngữ pháp của câu thông qua các quan hệ giữa từ điều khiển (head) và từ phụ thuộc (dependent).
Bài thực hành sử dụng thư viện spaCy để:

Phân tích cấu trúc câu

Xem cây phụ thuộc bằng displaCy

Truy cập thông tin của từng token

Duyệt cây để trích xuất chủ ngữ, tân ngữ, bổ ngữ

Thực hiện ba bài tập tự luyện: tìm ROOT, trích noun chunk, tìm đường đi đến ROOT

2. Kết quả thực hành
2.1 Phân tích câu và nhận diện ROOT

Khi phân tích câu:

“The quick brown fox jumps over the lazy dog.”

Kết quả chính:

ROOT của câu là: jumps

Chủ ngữ của câu là: fox

Cụm giới từ “over the lazy dog” là thành phần phụ thuộc của động từ jumps

Cấu trúc cây phụ thuộc rất rõ ràng khi hiển thị bằng displaCy

Việc xác định được ROOT và các quan hệ là nền tảng cho xử lý cú pháp nâng cao.

2.2 Truy cập token trong câu phức tạp

Với câu:

“Apple is looking at buying U.K. startup for $1 billion.”

Một số kết quả quan trọng:

ROOT là động từ looking

Apple → nsubj (chủ ngữ)

is → aux (trợ động từ)

at → prep (giới từ)

buying là thành phần bị điều khiển bởi giới từ at

startup là tân ngữ trực tiếp của động từ buying

Kết quả giúp hiểu cách spaCy xây dựng quan hệ head–dependent trong câu phức tạp nhiều tầng.

2.3 Trích xuất (Chủ ngữ – Động từ – Tân ngữ)

Trong câu:

“The cat chased the mouse and the dog watched them.”

Kết quả thu được:

(cat, chased, mouse)

(dog, watched, them) – tùy thuộc vào mô hình, có thể không nhận them là dobj do đại từ

Kết quả cho thấy mô hình có thể trích xuất tự động các bộ ba hành động (subject–verb–object), áp dụng vào Information Extraction.

2.4 Tìm tính từ bổ nghĩa cho danh từ

Với câu:

“The big, fluffy white cat is sleeping on the warm mat.”

Kết quả:

cat → [big, fluffy, white]

mat → [warm]

Điều này chứng minh spaCy nhận diện đúng nhãn amod (adjectival modifier).

2.5 Bài tập tự luyện – Kết quả
Bài 1 – Tìm động từ chính của câu

Mỗi câu thử nghiệm đều có thể xác định đúng động từ chính (token có nhãn ROOT).
Kết quả nhất quán với lý thuyết parser.

Bài 2 – Trích noun chunks thủ công

Khi tự trích xuất cụm danh từ bằng cách duyệt các từ bổ nghĩa:

Ví dụ câu thử:

“The warm small window is open.”

Kết quả tự trích xuất:

warm small window

So sánh với noun_chunks của spaCy:

the warm small window

Khác biệt chính: phương pháp thủ công chưa bao gồm mạo từ (det). Tuy nhiên logic chung là đúng.

Bài 3 – Đường đi từ token đến ROOT

Ví dụ với token “stranger” trong câu dài:

“The big dog in the garden is barking loudly at the stranger.”

Đường đi thu được:

stranger → at → barking (ROOT)


Điều này chứng minh thuật toán duyệt head nhiều lần sẽ luôn đi đến ROOT.

3. Khó khăn gặp phải trong quá trình thực hành
3.1 displaCy không hiển thị trên Colab/Notebook như mong muốn

displaCy.serve mở server nội bộ, không hiển thị trực tiếp trên Colab.

Cách khắc phục: dùng displacy.render(..., jupyter=True).

3.2 Một số đại từ không được parser gán nhãn dobj

Trong câu:

“… the dog watched them.”

"them" không luôn được nhận diện là tân ngữ trực tiếp (dobj) tùy thuộc mô hình.

Điều này ảnh hưởng đến việc trích bộ ba (subject, verb, object).

Giải pháp: mở rộng loại quan hệ cần xét (dobj, pobj, obj,…).

3.3 Noun chunk tự viết không giống 100% với noun_chunks của spaCy

Gặp khó khăn xác định thứ tự các modifier (compound, amod, det).

Không bao gồm các thành phần mở rộng như “in the garden”.

Điều này là bình thường do mô hình noun_chunks của spaCy dựa trên thuật toán chuẩn phức tạp hơn.

3.4 Một số token HEAD gây khó hiểu

Ví dụ:

Dấu chấm “.” được xem là dependent của ROOT.

Một số giới từ điều khiển các cụm động từ (pcomp).

Cần quan sát trực quan bằng displaCy để hiểu rõ.

3.5 Khó khăn khi đọc quan hệ DEP do nhãn nhiều và phức tạp

Ví dụ nhãn:

aux, pcomp, quantmod, compound, pobj

Khi mới học dễ nhầm lẫn. Cách khắc phục: xem bảng quan hệ trong tài liệu spaCy.
