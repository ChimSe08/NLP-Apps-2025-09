"""Pipeline chính cho Lab 5 Part 3: RNN cho POS Tagging (UD_English-EWT).

Thực hiện các bước tương ứng với file lab5_rnn_for_pos_tagging.pdf:
- Task 1: Tải & tiền xử lý dữ liệu CoNLL-U
- Task 2: Tạo Dataset & DataLoader
- Task 3: Xây dựng mô hình RNN
- Task 4: Huấn luyện
- Task 5: Đánh giá + predict_sentence
"""

import os
from typing import List

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .data_loader import load_conllu
from .vocab import build_word_vocab, build_tag_vocab, invert_mapping
from .dataset import POSDataset
from .collate import pos_collate_fn
from .model_rnn import SimpleRNNForTokenClassification
from .train import train_model
from .evaluate import evaluate_accuracy, predict_sentence


def run_pipeline(
    train_path: str = "data/en_ewt-ud-train.conllu",
    dev_path: str = "data/en_ewt-ud-dev.conllu",
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    batch_size: int = 32,
    num_epochs: int = 3,
):
    # ===== Task 1: Tải & tiền xử lý dữ liệu =====
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Không tìm thấy file train: {train_path}. Hãy đặt en_ewt-ud-train.conllu vào thư mục data/."
        )
    if not os.path.exists(dev_path):
        raise FileNotFoundError(
            f"Không tìm thấy file dev: {dev_path}. Hãy đặt en_ewt-ud-dev.conllu vào thư mục data/."
        )

    print("==> Đang đọc dữ liệu train từ", train_path)
    train_sentences_pairs = load_conllu(train_path)
    print("==> Đang đọc dữ liệu dev từ", dev_path)
    dev_sentences_pairs = load_conllu(dev_path)

    # Tách thành danh sách token và danh sách tag
    train_words_seqs: List[List[str]] = [w for (w, t) in train_sentences_pairs]
    train_tag_seqs: List[List[str]] = [t for (w, t) in train_sentences_pairs]

    dev_words_seqs: List[List[str]] = [w for (w, t) in dev_sentences_pairs]
    dev_tag_seqs: List[List[str]] = [t for (w, t) in dev_sentences_pairs]

    print(f"Số câu train: {len(train_words_seqs)}, dev: {len(dev_words_seqs)}")

    # Xây vocab
    print("==> Xây dựng vocab ...")
    word_to_ix = build_word_vocab(train_words_seqs, min_freq=1)
    tag_to_ix = build_tag_vocab(train_tag_seqs)
    ix_to_tag = invert_mapping(tag_to_ix)

    print(f"Kích thước vocab từ: {len(word_to_ix)} (bao gồm <UNK>)")
    print(f"Số lượng nhãn POS: {len(tag_to_ix)}")

    # ===== Task 2: Dataset & DataLoader =====
    print("==> Tạo Dataset & DataLoader ...")
    train_dataset = POSDataset(train_words_seqs, train_tag_seqs, word_to_ix, tag_to_ix)
    dev_dataset = POSDataset(dev_words_seqs, dev_tag_seqs, word_to_ix, tag_to_ix)

    pad_token_idx = 0  # chúng ta dùng 0 cho PAD (embedding padding_idx=0)
    pad_tag_value = -100  # ignore_index trong CrossEntropyLoss

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: pos_collate_fn(
            batch, pad_token_idx=pad_token_idx, pad_tag_value=pad_tag_value
        ),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: pos_collate_fn(
            batch, pad_token_idx=pad_token_idx, pad_tag_value=pad_tag_value
        ),
    )

    # ===== Task 3: Mô hình RNN =====
    print("==> Khởi tạo mô hình RNN ...")
    vocab_size = max(word_to_ix.values()) + 1  # vì index bắt đầu từ 0..max
    tagset_size = len(tag_to_ix)

    model = SimpleRNNForTokenClassification(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        bidirectional=False,
    )

    # ===== Task 4: Huấn luyện =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Sử dụng device:", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_tag_value)

    train_model(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
        ignore_index=pad_tag_value,
    )

    # ===== Task 5: Đánh giá cuối cùng =====
    print("==> Đánh giá final trên tập dev ...")
    acc_dev = evaluate_accuracy(model, dev_loader, device, ignore_index=pad_tag_value)
    print(f"Final token-level accuracy (dev): {acc_dev:.4f}")

    # Ví dụ predict_sentence trong đề bài
    example_sentence = "I love NLP"
    print("\n==> Ví dụ dự đoán cho câu:")
    print("Sentence:", example_sentence)
    pred_pairs = predict_sentence(example_sentence, model, word_to_ix, ix_to_tag, device)
    print("Predictions:")
    for word, tag in pred_pairs:
        print(f"{word}\t{tag}")


if __name__ == "__main__":
    run_pipeline()
