"""Đọc dữ liệu POS tagging từ file CoNLL-U (UD_English-EWT).

File .conllu được kỳ vọng đặt trong thư mục `data/`, ví dụ:
- data/en_ewt-ud-train.conllu
- data/en_ewt-ud-dev.conllu
"""

from typing import List, Tuple


def load_conllu(file_path: str) -> List[Tuple[List[str], List[str]]]:
    """Đọc file .conllu và trả về danh sách câu.

    Mỗi câu là một tuple (words, tags), trong đó:
        - words: List[str] danh sách token.
        - tags:  List[str] danh sách nhãn UPOS tương ứng.

    Dữ liệu CoNLL-U:
        - Dòng bắt đầu bằng '#' là comment -> bỏ qua.
        - Dòng trống -> kết thúc một câu.
        - Các cột được ngăn bởi tab. Ta dùng:
            cột 1: ID
            cột 2: FORM (từ)
            cột 4: UPOS (nhãn POS)
    """
    sentences: List[Tuple[List[str], List[str]]] = []
    words: List[str] = []
    tags: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 4:
                continue

            token_id = cols[0]
            # Bỏ qua token multi-word hoặc empty node trong chuẩn UD
            if "-" in token_id or "." in token_id:
                continue

            form = cols[1]
            upos = cols[3]

            words.append(form)
            tags.append(upos)

    # Câu cuối cùng nếu không kết thúc bằng dòng trống
    if words:
        sentences.append((words, tags))

    return sentences
