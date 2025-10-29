"""
A thin wrapper over scikit-learn's TfidfVectorizer to allow plugging in our RegexTokenizer.
"""
from typing import List, Iterable, Optional, Callable, Any
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVectorizer

class TfidfVectorizer:
    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        ngram_range=(1,1),
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
        stop_words: Optional[Iterable[str]] = None
    ):
        self._external_tokenizer = tokenizer
        analyzer: Optional[Callable[[str], List[str]]] = None
        if tokenizer is not None:
            analyzer = tokenizer.tokenize

        self._vec = SkTfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase if analyzer is None else False,
            stop_words=stop_words
        )

    def fit_transform(self, texts: List[str]):
        return self._vec.fit_transform(texts)

    def transform(self, texts: List[str]):
        return self._vec.transform(texts)
