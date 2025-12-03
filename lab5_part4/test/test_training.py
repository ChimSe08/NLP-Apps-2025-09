import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.model_rnn import RNNForNER
from src.dataset import NERDataset
from src.vocab_builder import build_word_vocab, build_tag_vocab, PAD_TOKEN
from src.collate_fn import ner_collate_fn
from src.train import train_model


def test_train_one_epoch_smoke():
    # Tạo dữ liệu toy rất nhỏ, chỉ để kiểm tra code không crash
    sentences = [["Hello", "world"], ["VNU", "Hanoi"], ["I", "love", "NLP"]]
    tags = [["O", "O"], ["B-ORG", "B-LOC"], ["O", "O", "B-MISC"]]

    word_to_ix = build_word_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    ds = NERDataset(sentences, tags, word_to_ix, tag_to_ix)

    pad_token_idx = word_to_ix[PAD_TOKEN]
    pad_tag_value = -100

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: ner_collate_fn(
            batch, pad_tag_value=pad_tag_value, pad_token_idx=pad_token_idx
        ),
    )

    model = RNNForNER(vocab_size=len(word_to_ix), tagset_size=len(tag_to_ix))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_tag_value)
    device = torch.device("cpu")

    # Train 1 epoch để kiểm tra không lỗi
    train_model(
        model=model,
        train_loader=loader,
        val_loader=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=1,
    )
