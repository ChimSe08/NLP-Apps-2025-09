"""Inspect and print dependency information for each token in a sentence."""

from spacy.tokens import Doc


def print_dependency_table(doc: Doc):
    """Print a formatted table of dependency information for each token.

    Columns:
    - TEXT
    - DEP  (dependency label)
    - HEAD TEXT
    - HEAD POS
    - CHILDREN (list of token texts)
    """
    print(f"{'TEXT':<12} | {'DEP':<10} | {'HEAD TEXT':<12} | {'HEAD POS':<8} | CHILDREN")
    print("-" * 70)
    for token in doc:
        children = [child.text for child in token.children]
        print(
            f"{token.text:<12} | "
            f"{token.dep_:<10} | "
            f"{token.head.text:<12} | "
            f"{token.head.pos_:<8} | "
            f"{children}"
        )
