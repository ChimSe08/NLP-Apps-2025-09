"""Pipeline chính cho Lab 5 Part 4: RNN cho NER với CoNLL2003.

Thực hiện các bước tương ứng với file lab5_rnn_for_ner.pdf:
- Task 1: Tải & tiền xử lý dữ liệu
- Task 2: Tạo Dataset & DataLoader
- Task 3: Xây dựng mô hình RNN
- Task 4: Huấn luyện
- Task 5: Đánh giá + predict_sentence
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from .data_loader import load_conll2003, extract_sentences_and_tags, convert_tag_ids_to_strings
from .vocab_builder import build_word_vocab, build_tag_vocab, invert_mapping, PAD_TOKEN
from .dataset import NERDataset
from .collate_fn import ner_collate_fn
from .model_rnn import RNNForNER
from .train import train_model
from .evaluate import evaluate_accuracy, predict_sentence


def run_pipeline(
    embedding_dim: int = 100,
    hidden_dim: int = 128,
    batch_size: int = 32,
    num_epochs: int = 3,
):
    # ===== Task 1: Tải & tiền xử lý dữ liệu =====
    print("==> Loading CoNLL2003 dataset ...")
    dataset = load_conll2003()

    train_split = dataset["train"]
    val_split = dataset["validation"]

    train_tokens, train_tag_ids, label_names = extract_sentences_and_tags(train_split)
    val_tokens, val_tag_ids, _ = extract_sentences_and_tags(val_split)

    # Chuyển tag id -> string
    train_tags_str = convert_tag_ids_to_strings(train_tag_ids, label_names)
    val_tags_str = convert_tag_ids_to_strings(val_tag_ids, label_names)

    print(f"Số câu train: {len(train_tokens)}, validation: {len(val_tokens)}")
    print(f"Các nhãn NER: {label_names}")

    # Xây vocabulary
    print("==> Building vocabularies ...")
    word_to_ix = build_word_vocab(train_tokens, min_freq=1)
    tag_to_ix = build_tag_vocab(train_tags_str)
    ix_to_tag = invert_mapping(tag_to_ix)

    print(f"Vocab size (words): {len(word_to_ix)}")
    print(f"Tag set size: {len(tag_to_ix)}")

    # ===== Task 2: Dataset & DataLoader =====
    print("==> Creating Dataset & DataLoader ...")
    train_dataset = NERDataset(train_tokens, train_tags_str, word_to_ix, tag_to_ix)
    val_dataset = NERDataset(val_tokens, val_tags_str, word_to_ix, tag_to_ix)

    pad_token_idx = word_to_ix[PAD_TOKEN]
    pad_tag_value = -100  # sẽ dùng làm ignore_index

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: ner_collate_fn(
            batch, pad_tag_value=pad_tag_value, pad_token_idx=pad_token_idx
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: ner_collate_fn(
            batch, pad_tag_value=pad_tag_value, pad_token_idx=pad_token_idx
        ),
    )

    # ===== Task 3: Mô hình RNN =====
    print("==> Initializing RNN model ...")
    vocab_size = len(word_to_ix)
    tagset_size = len(tag_to_ix)

    model = RNNForNER(
        vocab_size=vocab_size,
        tagset_size=tagset_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        bidirectional=False,  # theo yêu cầu có thể là RNN đơn giản
    )

    # ===== Task 4: Huấn luyện =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_tag_value)

    print("==> Training ...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=num_epochs,
    )

    # ===== Task 5: Đánh giá cuối cùng trên validation =====
    print("==> Final evaluation on validation set ...")
    val_acc = evaluate_accuracy(model, val_loader, device, ignore_index=pad_tag_value)
    print(f"Final token-level accuracy (val): {val_acc:.4f}")

    # Ví dụ predict_sentence trong đề bài
    example_sentence = "VNU University is located in Hanoi"
    print("\n==> Example prediction for:")
    print("Sentence:", example_sentence)
    pred_pairs = predict_sentence(
        example_sentence, model, word_to_ix, ix_to_tag, device
    )
    print("Predictions:")
    for word, tag in pred_pairs:
        print(f"{word}\t{tag}")


if __name__ == "__main__":
    run_pipeline()
