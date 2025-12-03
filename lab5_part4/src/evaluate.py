"""Các hàm đánh giá mô hình NER."""

from typing import List
import torch
from torch.utils.data import DataLoader


def evaluate_accuracy(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    ignore_index: int = -100,
) -> float:
    """Tính độ chính xác token-level, bỏ qua vị trí padding (ignore_index)."""
    model.eval()
    model.to(device)

    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for sentences, tags in data_loader:
            sentences = sentences.to(device)
            tags = tags.to(device)

            logits = model(sentences)  # (batch, seq, num_labels)
            preds = torch.argmax(logits, dim=-1)  # (batch, seq)

            mask = tags != ignore_index
            correct_tokens += (preds[mask] == tags[mask]).sum().item()
            total_tokens += mask.sum().item()

    if total_tokens == 0:
        return 0.0
    return correct_tokens / total_tokens


def predict_sentence(
    sentence: str,
    model: torch.nn.Module,
    word_to_ix: dict,
    ix_to_tag: dict,
    device: torch.device,
) -> List[tuple]:
    """Dự đoán nhãn NER cho một câu mới (chuỗi)."""
    model.eval()
    tokens = sentence.split()  # tokenizer đơn giản, có thể thay bằng tokenizer tốt hơn
    unk_idx = word_to_ix.get("<UNK>", 1)
    ids = [word_to_ix.get(tok, unk_idx) for tok in tokens]
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)  # (1, seq)

    with torch.no_grad():
        logits = model(input_ids)  # (1, seq, num_labels)
        preds = torch.argmax(logits, dim=-1)[0].cpu().tolist()  # (seq,)

    tags = [ix_to_tag[p] for p in preds]
    return list(zip(tokens, tags))
