"""Main pipeline for Lab 5 Part 1: PyTorch Introduction.

Chạy tuần tự các task:
- Task 1: Tensor cơ bản
- Task 2: Autograd
- Task 3: nn.Linear, nn.Embedding, nn.Module
"""

from .tensors_intro import run_task1_tensors
from .autograd_intro import run_task2_autograd
from .nn_intro import run_task3_nn


def run_all():
    print("=== Running Task 1 (Tensors) ===")
    run_task1_tensors()
    print("\n=== Running Task 2 (Autograd) ===")
    run_task2_autograd()
    print("\n=== Running Task 3 (nn modules) ===")
    run_task3_nn()


if __name__ == "__main__":
    run_all()
