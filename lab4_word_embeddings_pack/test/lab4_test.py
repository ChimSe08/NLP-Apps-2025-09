from __future__ import annotations
from pprint import pprint
import numpy as np
from src.representations.word_embedder import WordEmbedder

def main():
    we = WordEmbedder('glove-wiki-gigaword-50')

    v_king = we.get_vector('king')
    print("Vector('king') shape:", None if v_king is None else v_king.shape)

    sim_k_q = we.get_similarity('king', 'queen')
    sim_k_m = we.get_similarity('king', 'man')
    print(f"similarity('king','queen') = {sim_k_q}")
    print(f"similarity('king','man')   = {sim_k_m}")

    top = we.get_most_similar('computer', top_n=10)
    print("\nTop-10 most similar to 'computer':")
    for rank, (w, score) in enumerate(top, 1):
        print(f"{rank:2d}. {w:15s}  {score:.4f}")

    sent = "The queen rules the country."
    doc_vec = we.embed_document(sent)
    print("\nDoc embedding for:", sent)
    print("shape:", doc_vec.shape)
    print("head10:", np.array2string(doc_vec[:10], precision=4))

if __name__ == '__main__':
    main()
