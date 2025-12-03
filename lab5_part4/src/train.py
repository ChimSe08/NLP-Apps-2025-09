"""Hàm huấn luyện mô hình RNN cho NER."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_epochs: int = 3,
) -> None:
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} - training"):
            sentences, tags = batch
            sentences = sentences.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            logits = model(sentences)  # (batch, seq, num_labels)

            # reshape để dùng CrossEntropyLoss: (N, C) và target (N,)
            batch_size, seq_len, num_labels = logits.shape
            loss = loss_fn(
                logits.view(batch_size * seq_len, num_labels),
                tags.view(batch_size * seq_len),
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

        # Đánh giá nhanh trên validation
        if val_loader is not None:
            from .evaluate import evaluate_accuracy

            val_acc = evaluate_accuracy(model, val_loader, device, loss_fn.ignore_index)
            print(f"  Validation accuracy (token-level, no PAD): {val_acc:.4f}")
