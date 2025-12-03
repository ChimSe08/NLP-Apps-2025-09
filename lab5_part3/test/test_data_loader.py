from src.data_loader import load_conllu
import os


def test_load_conllu_dummy(tmp_path):
    # Tạo file conllu nhỏ để test
    content = """# sent_id = dummy-1
# text = I love NLP
1	I	_	PRON	_	_	0	_	_	_
2	love	_	VERB	_	_	0	_	_	_
3	NLP	_	PROPN	_	_	0	_	_	_

"""
    file_path = tmp_path / "dummy.conllu"
    file_path.write_text(content, encoding="utf-8")

    sentences = load_conllu(str(file_path))
    assert len(sentences) == 1
    words, tags = sentences[0]
    assert words == ["I", "love", "NLP"]
    assert tags == ["PRON", "VERB", "PROPN"]
