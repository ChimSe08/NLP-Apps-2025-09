"""Hàm collate_fn để pad batch cho NER."""

from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence

from .vocab_builder import PAD_TOKEN


def ner_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_tag_value: int = -100,
    pad_token_idx: int = 0,
):
    """Pad batch về cùng độ dài.

    Args:
        batch: List các cặp (sentence_ids, tag_ids), mỗi phần tử là 1D LongTensor.
        pad_tag_value: Giá trị padding cho nhãn (sẽ dùng làm ignore_index trong loss).
        pad_token_idx: Index của token PAD trong word_to_ix.

    Returns:
        sentences_padded: Tensor (batch_size, max_len)
        tags_padded: Tensor (batch_size, max_len)
    """
    sentences, tags = zip(*batch)  # tuple of tensors

    sentences_padded = pad_sequence(
        sentences, batch_first=True, padding_value=pad_token_idx
    )
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=pad_tag_value)

    return sentences_padded, tags_padded
