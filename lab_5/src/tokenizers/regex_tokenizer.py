"""
Simple RegexTokenizer used in Lab 5.
Splits on non-alphanumeric boundaries, lowercases by default, and keeps numbers.
"""
import re
from typing import List

class RegexTokenizer:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self._pattern = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", re.UNICODE)

    def tokenize(self, text: str) -> List[str]:
        if text is None:
            return []
        if self.lowercase:
            text = text.lower()
        return self._pattern.findall(text)
