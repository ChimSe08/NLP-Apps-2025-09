# main.py
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.dataset_loaders import load_raw_text_data

def main():
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # ---- Load dataset ----
    dataset_path = "data/en_ewt-ud-train.txt"   
    raw_text = load_raw_text_data(dataset_path)

    # Lấy một đoạn nhỏ để demo
    sample_text = raw_text[:500]

    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample: {sample_text[:100]}...")

    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")

    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")

if __name__ == "__main__":
    main()
