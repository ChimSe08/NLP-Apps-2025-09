from src.vocab import build_word_vocab, build_tag_vocab, UNK_TOKEN


def test_build_vocab_basic():
    sentences = [["I", "love", "NLP"], ["I", "like", "ML"]]
    tags = [["PRON", "VERB", "PROPN"], ["PRON", "VERB", "PROPN"]]

    word_to_ix = build_word_vocab(sentences)
    tag_to_ix = build_tag_vocab(tags)

    assert UNK_TOKEN in word_to_ix
    assert "I" in word_to_ix
    assert "PRON" in tag_to_ix
