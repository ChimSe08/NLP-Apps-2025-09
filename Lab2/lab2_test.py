from src.representations.count_vectorizer import CountVectorizer
from src.core.tokenizer import RegexTokenizer  # Assuming you have this from Lab 1

# Initialize tokenizer
tokenizer = RegexTokenizer()

# Initialize CountVectorizer
vectorizer = CountVectorizer(tokenizer)

# Sample corpus
corpus = [
    "I love NLP.",
    "I love programming.",
    "NLP is a subfield of AI."
]

# Fit and transform
vectors = vectorizer.fit_transform(corpus)

# Print results
print("Vocabulary:", vectorizer.vocabulary_)
print("Document-Term Matrix:")
for vec in vectors:
    print(vec)

