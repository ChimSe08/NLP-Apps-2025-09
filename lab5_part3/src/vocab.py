"""Xây dựng vocabulary cho từ và nhãn POS."""

from typing import Dict, List

UNK_TOKEN = "<UNK>"


def build_word_vocab(sentences: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    """Xây dựng word_to_ix từ danh sách câu (mỗi câu là list token).

    Thêm token đặc biệt <UNK> cho từ không xuất hiện trong vocab.
    Không thêm <PAD> ở đây vì padding có thể xử lý riêng (dùng chỉ số 0).
    """
    from collections import Counter

    counter = Counter()
    for sent in sentences:
        counter.update(sent)

    # 0 sẽ dành cho PAD, 1 dành cho UNK
    word_to_ix = {UNK_TOKEN: 1}
    idx = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            if word not in word_to_ix:
                word_to_ix[word] = idx
                idx += 1

    return word_to_ix


def build_tag_vocab(tag_sequences: List[List[str]]) -> Dict[str, int]:
    """Xây dựng tag_to_ix từ danh sách nhãn (string)."""
    tags = set()
    for seq in tag_sequences:
        tags.update(seq)
    tag_to_ix = {tag: idx for idx, tag in enumerate(sorted(tags))}
    return tag_to_ix


def invert_mapping(mapping: Dict[str, int]) -> Dict[int, str]:
    """Đảo ánh xạ str->int thành int->str."""
    return {v: k for k, v in mapping.items()}
