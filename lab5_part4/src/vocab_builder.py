"""Vocabulary builder for NER (Lab 5 Part 4).

Xây dựng word_to_ix và tag_to_ix từ dữ liệu CoNLL2003.
"""

from collections import Counter
from typing import Dict, List, Tuple

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_word_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """Xây dựng từ điển word_to_ix từ danh sách câu.

    Args:
        sentences: Danh sách các câu, mỗi câu là list token.
        min_freq: Ngưỡng tần suất tối thiểu để giữ lại từ.

    Returns:
        word_to_ix: Ánh xạ từ -> index.
    """
    counter: Counter = Counter()
    for sent in sentences:
        counter.update(sent)

    # Bắt đầu với token đặc biệt
    word_to_ix = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    idx = len(word_to_ix)

    for word, freq in counter.items():
        if freq >= min_freq:
            if word not in word_to_ix:
                word_to_ix[word] = idx
                idx += 1

    return word_to_ix


def build_tag_vocab(tag_sequences: List[List[str]]) -> Dict[str, int]:
    """Xây dựng tag_to_ix từ danh sách nhãn (dạng string).

    Args:
        tag_sequences: Danh sách chuỗi nhãn, mỗi phần tử là list các nhãn string.

    Returns:
        tag_to_ix: Ánh xạ tag string -> index.
    """
    tags = set()
    for seq in tag_sequences:
        tags.update(seq)
    tag_to_ix = {tag: idx for idx, tag in enumerate(sorted(tags))}
    return tag_to_ix


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    """Đảo ánh xạ str->int thành int->str."""
    return {v: k for k, v in mapping.items()}
