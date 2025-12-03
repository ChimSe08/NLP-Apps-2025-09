# Lab 5 – Part 1: Giới thiệu PyTorch

Repo này tổ chức lại nội dung từ notebook `Lab5_PyTorch.ipynb` thành các module rõ ràng:

- `src/tensors_intro.py`: các Task 1.x về Tensor (tạo tensor, toán tử, indexing, reshape, device).
- `src/autograd_intro.py`: các Task 2.x về autograd (tính gradient, computational graph).
- `src/nn_intro.py`: các Task 3.x về `nn.Linear`, `nn.Embedding`, và mô hình `nn.Module` đơn giản.
- `src/main.py`: hàm `run_all()` gọi lần lượt các phần trên.
- `notebook/`: chứa notebook gốc để bạn mở trên Colab/Jupyter.
- `test/`: một số smoke test cơ bản (import module, chạy hàm không lỗi).
- `report/`: nơi để bạn đặt báo cáo Part 1 nếu cần.
- `data/`: để trống (Part 1 không dùng dữ liệu riêng).

## Cách chạy

Từ thư mục gốc repo:

```bash
python -m src.main
```

Hoặc:

```python
from src.main import run_all
run_all()
```

Các hàm sẽ in ra kết quả minh hoạ cho từng Task (giống notebook).
