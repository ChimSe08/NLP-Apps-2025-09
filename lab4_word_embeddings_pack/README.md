# Lab 4 — Word Embeddings (Word2Vec / GloVe)

## Files
- `src/representations/word_embedder.py` — WordEmbedder with: get_vector, get_similarity, get_most_similar, embed_document
- `test/lab4_test.py` — Evaluation script
- `test/lab4_embedding_training_demo.py` — Train Word2Vec (gensim) on UD EWT
- `test/lab4_spark_word2vec_demo.py` — Spark Word2Vec demo
- `requirements.txt` — gensim / sklearn (Spark optional)

## Run
pip install -r requirements.txt
python -m test.lab4_test
python -m test.lab4_embedding_training_demo
python -m test.lab4_spark_word2vec_demo
