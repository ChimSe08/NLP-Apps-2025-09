"""Module auto-generated from notebook for Task 1.* (Tensors)."""

def run_task1_tensors():

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

