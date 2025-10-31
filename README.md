 Mục tiêu bài lab

Làm quen với Tensor trong PyTorch

Hiểu hoạt động của Autograd và tính đạo hàm tự động

Xây dựng mô hình đầu tiên bằng nn.Module

Thực hành với nn.Linear và nn.Embedding

 Phần 1 — Khám phá Tensor
 Tạo Tensor
 ```
Tensor từ list:
 tensor([[1, 2],
        [3, 4]])

Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Shape của tensor: torch.Size([2, 2])
Datatype của tensor: torch.float32
Device lưu trữ tensor: cpu
```
 Các phép toán trên Tensor
 ```
x_data + x_data =
 tensor([[2, 4],
        [6, 8]])

x_data * 5 =
 tensor([[ 5, 10],
        [15, 20]])

x_data @ x_data.T =
 tensor([[ 5, 11],
        [11, 25]])
```
 Indexing & Slicing + Reshape
 ```
Hàng đầu tiên: tensor([1, 2])
Cột thứ hai: tensor([2, 4])
Giá trị (1,1): 4
Shape: torch.Size([16, 1])
```

 Tổng kết phần Tensor:
→ Dễ thao tác, hỗ trợ GPU & tự động đạo hàm → nền tảng của mọi mô hình DL.

 Phần 2 — Autograd
 Gradient Example
```
Hàm:

𝑧=3(𝑥+2)2⇒𝑑𝑧𝑑𝑥=18 (𝑘ℎ𝑖 𝑥=1)
z=3(x+2)2⇒dxdz=18 (khi x=1)
```
 Output
```
Đạo hàm dz/dx tại x=1 (kỳ vọng 18): 18.0
```
 Lỗi khi backward() lần 2
RuntimeError: Trying to backward through the graph a second time


 Giải thích:
Biểu đồ tính toán sẽ được giải phóng sau backward() để tiết kiệm bộ nhớ.
→ Muốn dùng lại: retain_graph=True

 Phần 3 — torch.nn
 ```
 Linear Layer
Input shape: torch.Size([3, 5])
Output shape: torch.Size([3, 2])
```
 Embedding Layer
 ```
Input shape: torch.Size([4])
Output shape: torch.Size([4, 3])
```
 Mô hình đầu tiên MyFirstModel

Pipeline:

indices → Embedding → Mean Pooling → Linear + ReLU → Output

 Kết quả:
 ```
Input shape: torch.Size([2, 4])
Model output shape: torch.Size([2, 2])
```

 Ý nghĩa:
Mô hình phân loại hai lớp hoạt động đúng → nền tảng để học RNN/NLP.
 Kết luận
 ```
| Kiến thức          | Giá trị thực tiễn                   |
| ------------------ | ----------------------------------- |
| Tensor             | Thao tác dữ liệu nhanh & hỗ trợ GPU |
| Autograd           | Tối ưu mô hình nhờ Backpropagation  |
| nn.Module          | Xây dựng mạng học sâu phức tạp      |
| Linear + Embedding | Nền tảng mô hình NLP / NN           |
```
