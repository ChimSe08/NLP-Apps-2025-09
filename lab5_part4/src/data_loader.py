"""Load và chuẩn hóa dữ liệu CoNLL2003 từ Hugging Face datasets."""

from typing import List, Tuple, Dict
from datasets import load_dataset


def load_conll2003():
    """Tải bộ dữ liệu CoNLL2003.

    Returns:
        dataset: DatasetDict chứa train/validation/test.
    """
    dataset = load_dataset("conll2003")
    return dataset


def extract_sentences_and_tags(dataset_split) -> Tuple[List[List[str]], List[List[int]], List[str]]:
    """Trích xuất câu (tokens) và nhãn (ner_tags) từ split.

    Args:
        dataset_split: dataset['train'] hoặc dataset['validation'] ...

    Returns:
        tokens_list: List[List[str]]
        tag_ids_list: List[List[int]]
        label_names: List[str] ánh xạ id -> string label
    """
    tokens_list: List[List[str]] = dataset_split["tokens"]
    tag_ids_list: List[List[int]] = dataset_split["ner_tags"]
    label_feature = dataset_split.features["ner_tags"].feature
    label_names: List[str] = list(label_feature.names)
    return tokens_list, tag_ids_list, label_names


def convert_tag_ids_to_strings(
    tag_ids_list: List[List[int]], label_names: List[str]
) -> List[List[str]]:
    """Chuyển ner_tags dạng số sang dạng string (B-PER, I-ORG, ...)."""
    all_tag_str: List[List[str]] = []
    for seq in tag_ids_list:
        all_tag_str.append([label_names[i] for i in seq])
    return all_tag_str
