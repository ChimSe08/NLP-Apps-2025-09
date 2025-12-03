"""Module auto-generated from notebook for Task 3.* (nn.Linear, nn.Embedding, nn.Module)."""

def run_task3_nn():

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

