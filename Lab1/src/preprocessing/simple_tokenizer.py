# src/preprocessing/simple_tokenizer.py
import re
from typing import List
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        "
        text = re.sub(r'([.,!?])', r' \1 ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        tokens = text.split(" ")
        
        return tokens
 
