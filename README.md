Lab 1: Tokenizer – Tiền xử lý dữ liệu văn bản
1. Mục tiêu

Mục tiêu của Lab 1 là:

Hiểu về quá trình tiền xử lý dữ liệu văn bản trong các bài toán Xử lý Ngôn ngữ Tự nhiên (NLP).

Xây dựng một Tokenizer cơ bản để tách văn bản thành các tokens (từ hoặc ký tự đặc biệt).

Thiết kế mã nguồn với kiến trúc hướng đối tượng, dễ mở rộng cho các bài toán sau như Vectorization, TF-IDF, Word Embedding.

2. Cấu trúc thư mục dự án

Cấu trúc dự án được thiết kế như sau để dễ quản lý:

Lab_1/
```
│
├── src/
│   ├── core/
│   │   ├── interfaces.py         # Định nghĩa interface cho Tokenizer
│   │
│   └── preprocessing/
│       └── simple_tokenizer.py   # Triển khai SimpleTokenizer
│
├── test/
│   └── lab1_test.py              # File kiểm thử
│
├── data/
│   └── sample.txt                 # File dữ liệu văn bản thô
│
└── README.md                      # Báo cáo Lab 1
```
3. Các bước triển khai
Bước 1. Định nghĩa Interface Tokenizer

File: src/core/interfaces.py

Sử dụng Abstract Base Class (ABC) trong Python để định nghĩa interface.

Bắt buộc các lớp kế thừa phải triển khai phương thức tokenize.

from abc import ABC, abstractmethod
from typing import List

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass


Ý nghĩa:
Giúp đảm bảo tính nhất quán khi phát triển các loại Tokenizer khác nhau như SimpleTokenizer hoặc RegexTokenizer.

Bước 2. Xây dựng SimpleTokenizer

File: src/preprocessing/simple_tokenizer.py

Chức năng:

Chuyển toàn bộ văn bản về chữ thường.

Chuẩn hóa khoảng trắng.

Thêm khoảng trắng quanh các dấu câu như . , ! ?.

Cuối cùng tách văn bản thành danh sách token bằng phương thức split().

import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        
        # Thêm khoảng trắng quanh dấu câu
        text = re.sub(r'([.,!?])', r' \1 ', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tách thành các token
        tokens = text.split(" ")
        
        return tokens

Bước 3. Xây dựng RegexTokenizer

File: src/core/regex_tokenizer.py

Sử dụng Regular Expression để tách từ và ký tự đặc biệt.

Biểu thức \w+|[^\w\s]:

\w+ → nhóm ký tự chữ, số, hoặc dấu gạch dưới.

[^\w\s] → nhóm ký tự đặc biệt, không phải chữ/số hoặc khoảng trắng.

import re
from typing import List
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

Bước 4. Hàm đọc dữ liệu văn bản

Đọc file văn bản thô để dùng làm dữ liệu kiểm thử.

def load_raw_text_data(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

Bước 5. Viết test kiểm thử

File: test/lab1_test.py

from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.core.regex_tokenizer import RegexTokenizer
from src.utils import load_raw_text_data

# Đọc dữ liệu
text = load_raw_text_data("data/sample.txt")

# Kiểm thử SimpleTokenizer
simple_tokenizer = SimpleTokenizer()
print("SimpleTokenizer:", simple_tokenizer.tokenize(text))

# Kiểm thử RegexTokenizer
regex_tokenizer = RegexTokenizer()
print("RegexTokenizer:", regex_tokenizer.tokenize(text))

4. Cách chạy code và kiểm tra kết quả
Bước 1. Chuẩn bị dữ liệu

Tạo file data/sample.txt với nội dung:

I love NLP! NLP is fun, exciting, and challenging.

Bước 2. Chạy chương trình

Mở terminal tại thư mục gốc của project và chạy lệnh:

python -m test.lab1_test

5. Kết quả chạy
--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample: Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the
mosque in the town of ...
SimpleTokenizer Output (first 20 tokens): ['al-zaman', ':', 'american', 'forces', 'killed', 'shaikh', 'abdullah', 'al-ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'qaim', ',']
RegexTokenizer Output (first 20 tokens): ['-', ':', '-', ',', 'w', ',', '.', '[', 'w', '.', ']', ':', '.', 'w', 'w', '!', ',', 'w', '.', 'w']


Nhận xét:

Cả hai tokenizer đều tách chính xác từ và dấu câu.

RegexTokenizer mạnh mẽ hơn, có thể mở rộng cho các bài toán phức tạp.

6. Khó khăn và cách giải quyết

Trong quá trình làm Lab 1, em gặp một số khó khăn như:

Lỗi không nhận diện module (ModuleNotFoundError):

Nguyên nhân: thiếu file __init__.py trong các thư mục src/, core/, preprocessing/.

Giải pháp: thêm file __init__.py rỗng vào mỗi thư mục.

Sai biểu thức Regex:

Ban đầu sử dụng r"w+" nên tokenizer không tách chính xác.

Giải pháp: sửa thành r"\w+|[^\w\s]".

Lỗi định dạng (IndentationError):

Do thừa hoặc thiếu khoảng trắng trong code.

Khắc phục: dùng Shift + Tab trong VS Code để căn chỉnh lại code.
