# src/core/tokenizer.py
import re
from typing import List

class RegexTokenizer:
    def __init__(self, pattern: str = r"\w+"):
        
        self.pattern = pattern

    def tokenize(self, text: str) -> List[str]:
       
        return re.findall(self.pattern, text.lower())
 
