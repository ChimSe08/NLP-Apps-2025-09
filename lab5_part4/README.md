# Lab 5 – Part 4: RNN cho bài toán NER (CoNLL2003)

Repo này được tổ chức theo cấu trúc chuẩn:

- `src/`:
  - `data_loader.py`: Tải và trích xuất dữ liệu CoNLL2003 từ Hugging Face.
  - `vocab_builder.py`: Xây dựng `word_to_ix`, `tag_to_ix`.
  - `dataset.py`: Lớp `NERDataset` (PyTorch Dataset).
  - `collate_fn.py`: Hàm `ner_collate_fn` dùng để padding batch.
  - `model_rnn.py`: Mô hình `RNNForNER` dùng `nn.Embedding`, `nn.RNN`, `nn.Linear`.
  - `train.py`: Hàm `train_model` – huấn luyện mô hình.
  - `evaluate.py`: Hàm đánh giá accuracy và `predict_sentence`.
  - `main.py`: Pipeline chính `run_pipeline()` nối tất cả các bước.
- `notebook/`:
  - `lab5_part4.ipynb`: Notebook gốc dùng để code nhanh / thử nghiệm.
- `report/`: Nơi lưu báo cáo (vd: `lab5_part4.md`, `lab5_part4.pdf`).
- `test/`: Các test cơ bản dùng pytest.
- `data/`: Thư mục dữ liệu, KHÔNG commit dataset lớn.
- `.gitignore`: Bỏ qua data lớn, cache, file tạm.
- `README.md`: File này.

## Cách chạy pipeline

```bash
# Từ thư mục gốc repo
python -m src.main
```

Hoặc trong Python:

```python
from src.main import run_pipeline
run_pipeline()
```

Pipeline sẽ:
1. Tải CoNLL2003 bằng Hugging Face `datasets`.
2. Xây dựng vocabulary cho từ và nhãn.
3. Tạo `NERDataset` và `DataLoader` với `collate_fn`.
4. Khởi tạo mô hình RNN (Embedding + RNN + Linear).
5. Huấn luyện vài epoch, in loss theo epoch.
6. Đánh giá accuracy trên tập validation (bỏ padding).
7. Chạy ví dụ `predict_sentence` cho câu:
   `"VNU University is located in Hanoi"`.

Bạn có thể chỉnh các tham số trong `run_pipeline()` (embedding_dim, hidden_dim, batch_size, num_epochs).
