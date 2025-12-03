# Auto-generated from Lab5_PyTorch notebook

def run_all():

    # ==== Task 1.1: Tạo Tensor ====
    import torch
    import numpy as np

    # Tạo tensor từ list
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    print("Tensor từ list:\n", x_data, "\n")

    # Tạo tensor từ NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    print("Tensor từ NumPy array:\n", x_np, "\n")

    # Tạo tensor với các giá trị ngẫu nhiên hoặc hằng số
    x_ones = torch.ones_like(x_data)  # cùng shape với x_data, toàn số 1
    print("Ones Tensor:\n", x_ones, "\n")

    x_rand = torch.rand_like(x_data, dtype=torch.float)  # tensor ngẫu nhiên
    print("Random Tensor:\n", x_rand, "\n")

    # In ra shape, dtype, device
    print("Shape của tensor:", x_rand.shape)
    print("Datatype của tensor:", x_rand.dtype)
    print("Device lưu trữ tensor:", x_rand.device)



    # ==== Task 1.2: Các phép toán trên Tensor ====
    # 1) Cộng x_data với chính nó
    add_self = x_data + x_data
    print("x_data + x_data =\n", add_self, "\n")

    # 2) Nhân x_data với 5
    mul_5 = x_data * 5
    print("x_data * 5 =\n", mul_5, "\n")

    # 3) Nhân ma trận x_data với x_data.T
    # Lưu ý: dùng toán tử @ để nhân ma trận
    matmul_res = x_data @ x_data.T
    print("x_data @ x_data.T =\n", matmul_res)



    # ==== Task 1.3: Indexing và Slicing ====
    print("x_data =\n", x_data, "\n")

    # 1) Lấy ra hàng đầu tiên
    row0 = x_data[0]
    print("Hàng đầu tiên:", row0)

    # 2) Lấy ra cột thứ hai
    # Dùng slicing tất cả hàng, cột index = 1
    col1 = x_data[:, 1]
    print("Cột thứ hai:", col1)

    # 3) Lấy ra giá trị ở hàng thứ hai, cột thứ hai (index 1,1)
    val_11 = x_data[1, 1]
    print("Giá trị tại (1,1):", val_11.item())



    # ==== Task 1.4: Thay đổi hình dạng Tensor ====
    rand_4x4 = torch.rand(4, 4)
    print("Tensor (4,4):\n", rand_4x4, "\n")

    # Biến thành (16, 1) bằng view hoặc reshape
    reshaped = rand_4x4.reshape(16, 1)  # hoặc: rand_4x4.view(16, 1)
    print("Reshape -> (16,1):\n", reshaped, "\n")
    print("Shape:", reshaped.shape)



    # ==== Task 2.1: Thực hành với autograd ====
    import torch

    # Tạo một tensor và yêu cầu tính đạo hàm cho nó
    x = torch.ones(1, requires_grad=True)
    print("x:", x)

    # Thực hiện một phép toán
    y = x + 2
    print("y:", y)
    print("grad_fn của y:", y.grad_fn)  # vì y được tạo từ phép toán nên có grad_fn

    # Thực hiện thêm các phép toán
    z = y * y * 3  # z = 3 * (x+2)^2

    # Tính đạo hàm của z theo x
    z.backward()  # tương đương z.backward(torch.tensor(1.))
    print("Đạo hàm dz/dx tại x=1 (kỳ vọng 6*(1+2)=18):", x.grad.item())

    # Thử gọi backward lần nữa để minh hoạ lỗi nếu không giữ biểu đồ
    try:
        z.backward()
    except Exception as e:
        print("\nGọi backward() lần 2 -> lỗi (vì biểu đồ đã giải phóng):")
        print(type(e).__name__ + ":", e)



    # Ví dụ giữ lại biểu đồ để backward nhiều lần
    x2 = torch.tensor([2.0], requires_grad=True)
    y2 = x2**3  # y = x^3
    # Lần 1
    y2.backward(retain_graph=True)
    print("dy/dx tại x=2 (lần 1):", x2.grad.item())  # 3*x^2 = 12

    # Lần 2 (vẫn OK nhờ retain_graph)
    y2.backward()
    print("dy/dx tích luỹ (lần 2):", x2.grad.item(), "(do gradient được cộng dồn)")  # sẽ là 24



    # ==== Task 3.1: Lớp nn.Linear ====
    import torch
    from torch import nn

    # Khởi tạo một lớp Linear biến đổi từ 5 chiều -> 2 chiều
    linear_layer = nn.Linear(in_features=5, out_features=2)

    # Tạo một tensor đầu vào mẫu: 3 mẫu, mỗi mẫu 5 chiều
    input_tensor = torch.randn(3, 5)
    output = linear_layer(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output.shape)
    print("Output:\n", output)



    # ==== Task 3.2: Lớp nn.Embedding ====
    import torch
    from torch import nn

    # Khởi tạo lớp Embedding cho từ điển 10 từ, mỗi từ biểu diễn bằng vector 3 chiều
    embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=3)

    # Tạo một tensor đầu vào chứa các chỉ số từ (ví dụ một câu)
    input_indices = torch.LongTensor([1, 5, 0, 8])  # các chỉ số phải < 10
    embeddings = embedding_layer(input_indices)

    print("Input shape:", input_indices.shape)
    print("Output shape:", embeddings.shape)
    print("Embeddings:\n", embeddings)



    # ==== Task 3.3: Kết hợp thành một nn.Module ====
    import torch
    from torch import nn

    class MyFirstModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(MyFirstModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.linear = nn.Linear(embedding_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        def forward(self, indices):
            # indices shape: (batch, seq_len)
            embeds = self.embedding(indices)          # -> (batch, seq_len, embedding_dim)
            # Đơn giản hoá: lấy trung bình theo chiều seq_len để gom thành vector cố định
            pooled = embeds.mean(dim=1)               # -> (batch, embedding_dim)
            hidden = self.activation(self.linear(pooled))
            output = self.output_layer(hidden)        # -> (batch, output_dim)
            return output

    # Khởi tạo và kiểm tra mô hình
    model = MyFirstModel(vocab_size=100, embedding_dim=16, hidden_dim=8, output_dim=2)
    input_data = torch.LongTensor([[1, 2, 5, 9], [3, 3, 7, 0]])  # batch=2, seq_len=4
    output_data = model(input_data)

    print("Input shape:", input_data.shape)
    print("Model output shape:", output_data.shape)
    print("Model output:\n", output_data)


if __name__=='__main__':
    run_all()
