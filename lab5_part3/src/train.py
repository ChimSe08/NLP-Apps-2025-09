"""Hàm huấn luyện mô hình RNN cho POS tagging."""

from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    dev_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device,
    num_epochs: int = 3,
    ignore_index: int = -100,
):
    model.to(device)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for sentences, tags in tqdm(train_loader, desc=f"Epoch {epoch} - training"):
            sentences = sentences.to(device)
            tags = tags.to(device)

            optimizer.zero_grad()
            logits = model(sentences)  # (batch, seq, num_tags)

            batch_size, seq_len, num_tags = logits.shape
            loss = loss_fn(
                logits.view(batch_size * seq_len, num_tags),
                tags.view(batch_size * seq_len),
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

        if dev_loader is not None:
            from .evaluate import evaluate_accuracy

            acc_train = evaluate_accuracy(model, train_loader, device, ignore_index)
            acc_dev = evaluate_accuracy(model, dev_loader, device, ignore_index)
            print(
                f"  Accuracy (train): {acc_train:.4f} | Accuracy (dev): {acc_dev:.4f}"
            )
