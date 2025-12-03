import src.vocab_builder as vb


def test_build_word_vocab_basic():
    sentences = [["hello", "world"], ["hello", "ner"]]
    word_to_ix = vb.build_word_vocab(sentences, min_freq=1)
    assert vb.PAD_TOKEN in word_to_ix
    assert vb.UNK_TOKEN in word_to_ix
    assert word_to_ix["hello"] != word_to_ix[vb.UNK_TOKEN]


def test_build_tag_vocab_basic():
    tags = [["O", "B-PER"], ["B-LOC", "O"]]
    tag_to_ix = vb.build_tag_vocab(tags)
    assert "O" in tag_to_ix
    assert "B-PER" in tag_to_ix
