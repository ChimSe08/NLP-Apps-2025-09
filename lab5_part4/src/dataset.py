"""PyTorch Dataset cho bài toán NER (Lab 5 Part 4)."""

from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

from .vocab_builder import UNK_TOKEN, PAD_TOKEN


class NERDataset(Dataset):
    """Dataset cho NER.

    Mỗi phần tử là một cặp (sentence_indices, tag_indices),
    chưa được padding (padding được xử lý ở collate_fn).
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
        tokens = self.sentences[idx]
        tags = self.tag_sequences[idx]

        word_ids = [
            self.word_to_ix.get(tok, self.word_to_ix[UNK_TOKEN]) for tok in tokens
        ]
        tag_ids = [self.tag_to_ix[tag] for tag in tags]

        return torch.tensor(word_ids, dtype=torch.long), torch.tensor(
            tag_ids, dtype=torch.long
        )
