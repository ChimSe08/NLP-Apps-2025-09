from __future__ import annotations
from typing import Iterator, List
from pathlib import Path
import gensim
from gensim.models import Word2Vec

DATA_PATH = Path('data/UD_English-EWT/en_ewt-ud-train.txt')
OUT_DIR = Path('results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT_DIR / 'word2vec_ewt.model'

class LineCorpus:
    def __init__(self, path: Path):
        self.path = path
    def __iter__(self) -> Iterator[List[str]]:
        with self.path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield gensim.utils.simple_preprocess(line, min_len=1)

def main():
    if not DATA_PATH.exists():
        print(f"[WARN] {DATA_PATH} not found. Provide a text file with one sentence per line.")
        return

    corpus = LineCorpus(DATA_PATH)
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=0,
        epochs=5,
    )
    model.save(str(MODEL_PATH))
    wv = model.wv
    if 'computer' in wv:
        print("Top-5 similar to 'computer':", wv.most_similar('computer', topn=5))
    if all(w in wv for w in ['king','man','woman']):
        print("Analogy king - man + woman:", wv.most_similar(positive=['king','woman'], negative=['man'], topn=3))

if __name__ == '__main__':
    main()
