# Lab 5 – Part 3: RNN cho bài toán POS Tagging (UD_English-EWT)

Repo này tổ chức các phần đúng theo yêu cầu trong file lab5_rnn_for_pos_tagging.pdf.

## Cấu trúc

- `src/`:
  - `data_loader.py`: Hàm `load_conllu` để đọc file .conllu, trả về danh sách các câu (words, tags).
  - `vocab.py`: Xây dựng `word_to_ix`, `tag_to_ix`, và hàm đảo `invert_mapping`.
  - `dataset.py`: Lớp `POSDataset` kế thừa `torch.utils.data.Dataset`.
  - `collate.py`: Hàm `pos_collate_fn` dùng `pad_sequence` để pad batch.
  - `model_rnn.py`: Mô hình `SimpleRNNForTokenClassification` dùng `nn.Embedding`, `nn.RNN`, `nn.Linear`.
  - `train.py`: Hàm `train_model`, huấn luyện qua nhiều epoch, in loss và accuracy.
  - `evaluate.py`: Hàm `evaluate_accuracy` và `predict_sentence`.
  - `main.py`: Hàm `run_pipeline()` nối tất cả các bước Task 1 → Task 5.
- `data/`:
  - **Bạn cần tự đặt**:
    - `en_ewt-ud-train.conllu`
    - `en_ewt-ud-dev.conllu`
  - `README.md`: Nhắc không commit dataset thật lên Git.
- `notebook/`:
  - `Lab5_part3.ipynb`: Notebook gốc dùng để thử nghiệm.
- `test/`:
  - Các file test cơ bản cho từng phần (data_loader, vocab, dataset, model, training).
- `report/`:
  - Thư mục trống cho báo cáo (ví dụ: `lab5_part3_report.md`).
- `.gitignore`, `README.md`: file meta cho repo.

## Cách chạy pipeline

1. Chuẩn bị dữ liệu:
   - Tải bộ UD_English-EWT (hoặc sử dụng dataset mà giảng viên cung cấp).
   - Đặt file:
     - `en_ewt-ud-train.conllu`
     - `en_ewt-ud-dev.conllu`
     vào thư mục `data/`.

2. Cài đặt thư viện cần thiết (PyTorch, tqdm).

3. Chạy pipeline:

```bash
# Từ thư mục gốc của repo
python -m src.main
```

Hoặc trong Python:

```python
from src.main import run_pipeline
run_pipeline(
    train_path="data/en_ewt-ud-train.conllu",
    dev_path="data/en_ewt-ud-dev.conllu",
    embedding_dim=100,
    hidden_dim=128,
    batch_size=32,
    num_epochs=3,
)
```

Pipeline sẽ:
1. Đọc file .conllu cho train/dev.
2. Xây dựng vocabulary cho từ và nhãn.
3. Tạo `POSDataset` và `DataLoader` với padding.
4. Khởi tạo mô hình RNN (Embedding + RNN + Linear).
5. Huấn luyện mô hình, in loss và accuracy trên train/dev.
6. Đánh giá final accuracy trên dev.
7. Dự đoán POS cho câu: `"I love NLP"` bằng `predict_sentence`.
