# Lab 5 – Part 2: Text Classification (HWU Intent)

Cấu trúc repo:

- `src/`:
  - `lab5_part2.py`: chứa toàn bộ pipeline (load dữ liệu, TF-IDF + LR, Word2Vec + Dense, LSTM pre-trained, LSTM scratch, so sánh).
- `notebook/`:
  - `Lab5_part2.ipynb`: notebook gốc để exploratory / chạy thử.
- `report/`: nơi lưu báo cáo (`lab5_part2.md`, `lab5_part2.pdf`, ...).
- `test/`:
  - `test_lab5_part2.py`: test đơn giản (import được module, có hàm `run_all`).
- `data/`: chứa file dữ liệu (KHÔNG commit dataset lớn lên git).
- `README.md`: mô tả project.
- `.gitignore`: bỏ qua data lớn, cache, file tạm.

## Cách chạy

```bash
# Chạy toàn bộ pipeline từ terminal (từ thư mục gốc repo)
python -m src.lab5_part2
```

Hoặc trong Python:

```python
from src.lab5_part2 import run_all
run_all()
```

Nhớ chuẩn bị dữ liệu HWU:
- Đặt `hwu.tar.gz` vào `data/` **hoặc**
- Đặt `train.csv`, `val.csv`, `test.csv` vào `data/` / `data/hwu/`
