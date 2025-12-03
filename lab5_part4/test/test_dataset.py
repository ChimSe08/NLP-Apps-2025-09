import torch
from src.dataset import NERDataset
from src.vocab_builder import build_word_vocab, build_tag_vocab


def test_ner_dataset_shapes():
    sentences = [["Hello", "world"], ["VNU", "Hanoi"]]
    tags = [["O", "O"], ["B-ORG", "B-LOC"]]

    word_to_ix = build_word_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    ds = NERDataset(sentences, tags, word_to_ix, tag_to_ix)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == y.shape[0]
