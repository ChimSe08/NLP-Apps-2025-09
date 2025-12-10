"""Solutions for Lab 6 self-practice exercises:

Bài 1: find_main_verb(doc)
Bài 2: extract_noun_chunks_simple(doc)
Bài 3: get_path_to_root(token)
"""

from typing import List
from spacy.tokens import Doc, Token


def find_main_verb(doc: Doc) -> Token:
    """Return the main verb of the sentence.

    The main verb is typically the token with dep_ == "ROOT" and pos_ == "VERB".
    If no such token is found, fall back to the ROOT token.
    """
    root_candidates = [t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"]
    if root_candidates:
        return root_candidates[0]
    # fallback: any ROOT
    for t in doc:
        if t.dep_ == "ROOT":
            return t
    # as a last resort, return the first token
    return doc[0]


def extract_noun_chunks_simple(doc: Doc) -> List[List[Token]]:
    """Extract simple noun chunks without using doc.noun_chunks.

    A simple heuristic:
    - Start from each NOUN token.
    - Collect its modifiers (children) with dep_ in {det, amod, compound}.
    - Form a chunk consisting of [modifiers..., noun], ordered by token index.
    """
    chunks: List[List[Token]] = []
    for token in doc:
        if token.pos_ == "NOUN":
            modifiers = [
                child for child in token.children
                if child.dep_ in {"det", "amod", "compound"}
            ]
            chunk_tokens = modifiers + [token]
            chunk_tokens_sorted = sorted(chunk_tokens, key=lambda t: t.i)
            chunks.append(chunk_tokens_sorted)
    return chunks


def get_path_to_root(token: Token) -> List[Token]:
    """Return the path from a token up to the ROOT of the dependency tree.

    The path includes the starting token and the ROOT token.
    """
    path = [token]
    current = token
    while current.head != current:
        current = current.head
        path.append(current)
    return path
