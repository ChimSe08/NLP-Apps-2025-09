import torch
from src.dataset import POSDataset
from src.vocab import build_word_vocab, build_tag_vocab


def test_pos_dataset_shapes():
    sentences = [["I", "love", "NLP"], ["Hello", "world"]]
    tags = [["PRON", "VERB", "PROPN"], ["INTJ", "NOUN"]]

    word_to_ix = build_word_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    ds = POSDataset(sentences, tags, word_to_ix, tag_to_ix)
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == y.shape[0]
