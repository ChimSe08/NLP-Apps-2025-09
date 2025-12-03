"""PyTorch Dataset cho bài toán POS Tagging."""

from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

from .vocab import UNK_TOKEN


class POSDataset(Dataset):
    """Dataset cho POS.

    Input:
        sentences: List[List[str]] - mỗi câu là list token
        tags:      List[List[str]] - mỗi câu là list nhãn UPOS
    """

    def __init__(
        self,
        sentences: List[List[str]],
        tag_sequences: List[List[str]],
        word_to_ix: Dict[str, int],
        tag_to_ix: Dict[str, int],
    ) -> None:
        assert len(sentences) == len(tag_sequences)
        self.sentences = sentences
        self.tag_sequences = tag_sequences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        words = self.sentences[idx]
        tags = self.tag_sequences[idx]

        unk_idx = self.word_to_ix.get(UNK_TOKEN, 1)

        word_ids = [self.word_to_ix.get(w, unk_idx) for w in words]
        tag_ids = [self.tag_to_ix[t] for t in tags]

        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(
            tag_ids, dtype=torch.long
        )
