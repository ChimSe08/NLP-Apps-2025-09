# src/representations/count_vectorizer.py

from src.core.interfaces import Vectorizer
from typing import List, Dict

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]):
        unique_tokens = set()

        # Collect all unique tokens
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)

        # Assign indices to each unique token
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: List[str]) -> List[List[int]]:
        vectors = []

        for document in documents:
            # Initialize zero vector
            vector = [0] * len(self.vocabulary_)

            # Tokenize and update counts
            tokens = self.tokenizer.tokenize(document)
            for token in tokens:
                if token in self.vocabulary_:
                    vector[self.vocabulary_[token]] += 1

            vectors.append(vector)

        return vectors
 
