import torch
from src.model_rnn import SimpleRNNForTokenClassification


def test_rnn_forward_shape():
    vocab_size = 50
    tagset_size = 10
    model = SimpleRNNForTokenClassification(vocab_size=vocab_size, tagset_size=tagset_size)

    batch_size, seq_len = 4, 7
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, tagset_size)
