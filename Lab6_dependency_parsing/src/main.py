"""Main pipeline for Lab 6: Dependency Parsing with spaCy.

This module demonstrates:
- Loading spaCy model
- Analyzing and visualizing dependency tree
- Printing dependency table
- Extracting SVO triplets and noun-adjective relations
- Running self-practice exercise functions
"""

from .model_loader import load_en_model
from .visualize_dep import analyze_sentence
from .inspect_tree import print_dependency_table
from .extract_relations import print_svo_triplets, print_noun_adjectives
from .exercises import find_main_verb, extract_noun_chunks_simple, get_path_to_root


def run_demo(model_name: str = "en_core_web_md"):
    """Run a full demonstration for Lab 6 on several example sentences."""
    nlp = load_en_model(model_name)

    # Example 1
    text1 = "The quick brown fox jumps over the lazy dog."
    doc1 = analyze_sentence(nlp, text1)
    print("=== Example 1: Basic dependency info ===")
    print("Sentence:", text1)
    print_dependency_table(doc1)
    print()

    # Example 2
    text2 = "Apple is looking at buying U.K. startup for $1 billion"
    doc2 = analyze_sentence(nlp, text2)
    print("=== Example 2: Apple buying U.K. startup ===")
    print("Sentence:", text2)
    print_dependency_table(doc2)
    print()

    # Example 3
    text3 = "The cat chased the mouse and the dog watched them."
    doc3 = analyze_sentence(nlp, text3)
    print("=== Example 3: SVO triplets ===")
    print("Sentence:", text3)
    print_svo_triplets(doc3)
    print()

    # Example 4
    text4 = "The big, fluffy white cat is sleeping on the warm mat."
    doc4 = analyze_sentence(nlp, text4)
    print("=== Example 4: Noun adjectives ===")
    print("Sentence:", text4)
    print_noun_adjectives(doc4)
    print()

    # Exercises demo
    print("=== Exercises Demo ===")
    main_verb = find_main_verb(doc2)
    print(f"Main verb of sentence 2: {main_verb.text}")

    noun_chunks = extract_noun_chunks_simple(doc4)
    print("Simple noun chunks in sentence 4:")
    for chunk in noun_chunks:
        print("  -", " ".join([t.text for t in chunk]))

    # Path to root for token 'startup' in sentence 2 (if present)
    for token in doc2:
        if token.text.lower() == "startup":
            path = get_path_to_root(token)
            print("Path from 'startup' to ROOT:")
            print(" -> ".join([t.text for t in path]))
            break


if __name__ == "__main__":
    run_demo()
