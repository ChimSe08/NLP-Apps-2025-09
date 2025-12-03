"""Module auto-generated from notebook for Task 2.* (autograd)."""

def run_task2_autograd():

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

