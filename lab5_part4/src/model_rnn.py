"""Mô hình RNN đơn giản cho NER (Lab 5 Part 4).

Sử dụng:
- nn.Embedding
- nn.RNN
- nn.Linear
"""

from typing import Tuple
import torch
import torch.nn as nn


class RNNForNER(nn.Module):
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=False,  # (seq_len, batch, input_size)
        )
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, tagset_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Tensor (batch_size, seq_len) chứa chỉ số từ.

        Returns:
            logits: Tensor (batch_size, seq_len, tagset_size)
        """
        # (batch, seq) -> (seq, batch)
        x = input_ids.transpose(0, 1)
        emb = self.embedding(x)  # (seq, batch, emb_dim)
        rnn_out, _ = self.rnn(emb)  # (seq, batch, hidden)
        logits = self.fc(rnn_out)  # (seq, batch, tagset)
        # Đưa về (batch, seq, tagset)
        logits = logits.transpose(0, 1)
        return logits
