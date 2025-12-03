import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from src.dataset import POSDataset
from src.vocab import build_word_vocab, build_tag_vocab
from src.collate import pos_collate_fn
from src.model_rnn import SimpleRNNForTokenClassification
from src.train import train_model


def test_train_smoke_one_epoch():
    # Dữ liệu toy rất nhỏ
    sentences = [["I", "love", "NLP"], ["You", "like", "AI"]]
    tags = [["PRON", "VERB", "PROPN"], ["PRON", "VERB", "PROPN"]]

    word_to_ix = build_word_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    ds = POSDataset(sentences, tags, word_to_ix, tag_to_ix)

    pad_token_idx = 0
    pad_tag_value = -100

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: pos_collate_fn(
            batch, pad_token_idx=pad_token_idx, pad_tag_value=pad_tag_value
        ),
    )

    vocab_size = max(word_to_ix.values()) + 1
    tagset_size = len(tag_to_ix)

    model = SimpleRNNForTokenClassification(vocab_size=vocab_size, tagset_size=tagset_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_tag_value)
    device = torch.device("cpu")

    train_model(
        model=model,
        train_loader=loader,
        dev_loader=None,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        num_epochs=1,
        ignore_index=pad_tag_value,
    )
