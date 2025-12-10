"""Extract meaningful relations from a dependency tree.

Includes:
- Subject-Verb-Object triplets.
- Adjectives that modify a noun (amod).
"""

from typing import List, Tuple, Dict
from spacy.tokens import Doc


def find_svo_triplets(doc: Doc) -> List[Tuple[str, str, str]]:
    """Find (subject, verb, object) triplets in a sentence.

    Follows the lab example:
    - For each verb, look at its children with dep_ == "nsubj" and "dobj".
    """
    triplets: List[Tuple[str, str, str]] = []
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.text
            subject = ""
            obj = ""
            for child in token.children:
                if child.dep_ == "nsubj":
                    subject = child.text
                if child.dep_ == "dobj":
                    obj = child.text
            if subject and obj:
                triplets.append((subject, verb, obj))
    return triplets


def print_svo_triplets(doc: Doc):
    """Print SVO triplets found in a Doc."""
    for s, v, o in find_svo_triplets(doc):
        print(f"Found Triplet: ({s}, {v}, {o})")


def find_noun_adjectives(doc: Doc) -> Dict[str, List[str]]:
    """For each noun, collect adjectives (amod) that modify it.

    Returns
    -------
    mapping : dict
        Mapping from noun text -> list of adjectives texts.
    """
    result: Dict[str, List[str]] = {}
    for token in doc:
        if token.pos_ == "NOUN":
            adjectives: List[str] = []
            for child in token.children:
                if child.dep_ == "amod":
                    adjectives.append(child.text)
            if adjectives:
                result[token.text] = adjectives
    return result


def print_noun_adjectives(doc: Doc):
    """Print noun and its modifying adjectives."""
    mapping = find_noun_adjectives(doc)
    for noun, adjs in mapping.items():
        print(f"Danh từ '{noun}' được bổ nghĩa bởi các tính từ: {adjs}")
