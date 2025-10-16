from __future__ import annotations
from typing import List, Optional
import numpy as np
import gensim
import gensim.downloader as api

try:
    from src.preprocessing.tokenizer import Tokenizer  # adjust if your Lab1 path differs
except Exception:
    import re
    class Tokenizer:
        def __init__(self, lower: bool = True):
            self.lower = lower
        def tokenize(self, text: str) -> List[str]:
            if self.lower:
                text = text.lower()
            try:
                return re.findall(r"[\p{L}0-9']+", text, flags=re.UNICODE)
            except re.error:
                return re.findall(r"[A-Za-z0-9']+", text)

class WordEmbedder:
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50') -> None:
        self.model_name = model_name
        self.model = api.load(model_name)
        self.dim = int(getattr(self.model, 'vector_size', 0))
        self.tokenizer = Tokenizer()

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        try:
            return self.model[word]
        except KeyError:
            return None

    def get_similarity(self, word1: str, word2: str) -> Optional[float]:
        v1 = self.get_vector(word1)
        v2 = self.get_vector(word2)
        if v1 is None or v2 is None:
            return None
        num = float(np.dot(v1, v2))
        den = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        return (num / den) if den else None

    def get_most_similar(self, word: str, top_n: int = 10):
        try:
            return self.model.most_similar(word, topn=top_n)
        except KeyError:
            return []

    def embed_document(self, document: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(document)
        vecs = []
        for tok in tokens:
            v = self.get_vector(tok)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.vstack(vecs).mean(axis=0, dtype=np.float32)

    def embed_corpus(self, docs: List[str]) -> np.ndarray:
        return np.vstack([self.embed_document(d) for d in docs])
