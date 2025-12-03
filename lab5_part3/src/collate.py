"""Hàm collate_fn dùng để pad batch cho POS tagging."""

from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence


def pos_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_token_idx: int = 0,
    pad_tag_value: int = -100,
):
    """Pad batch về cùng độ dài.

    Args:
        batch: List các cặp (sentence_ids, tag_ids), dạng 1D LongTensor.
        pad_token_idx: giá trị padding cho token (mặc định 0).
        pad_tag_value: giá trị padding cho nhãn, dùng làm ignore_index trong loss.

    Returns:
        sentences_padded: Tensor (batch_size, max_len)
        tags_padded: Tensor (batch_size, max_len)
    """
    sentences, tags = zip(*batch)

    sentences_padded = pad_sequence(
        sentences, batch_first=True, padding_value=pad_token_idx
    )
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=pad_tag_value)

    return sentences_padded, tags_padded
