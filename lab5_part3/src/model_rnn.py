"""Mô hình RNN đơn giản cho POS Tagging.

Sử dụng:
- nn.Embedding
- nn.RNN
- nn.Linear
"""

from typing import Tuple
import torch
import torch.nn as nn


class SimpleRNNForTokenClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        tagset_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,  # input: (batch, seq, emb)
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, tagset_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            input_ids: Tensor (batch_size, seq_len) chứa chỉ số từ.

        Returns:
            logits: Tensor (batch_size, seq_len, tagset_size)
        """
        emb = self.embedding(input_ids)  # (batch, seq, emb_dim)
        rnn_out, _ = self.rnn(emb)       # (batch, seq, hidden*dir)
        logits = self.fc(rnn_out)        # (batch, seq, tagset)
        return logits
