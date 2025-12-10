"""Visualization utilities for dependency trees using displaCy."""

from spacy.tokens import Doc
from spacy import displacy


def analyze_sentence(nlp, text: str) -> Doc:
    """Run the spaCy pipeline on the given text and return the Doc."""
    return nlp(text)


def visualize_dependency(doc: Doc, in_browser: bool = True):
    """Visualize dependency tree of a Doc.

    If in_browser=True, spin up a small HTTP server at http://127.0.0.1:5000
    using displacy.serve (similar to the lab instructions).
    """
    if in_browser:
        # This will block and open at http://127.0.0.1:5000 until Ctrl+C
        displacy.serve(doc, style="dep")
    else:
        # Return HTML/SVG as string (for notebooks, etc.)
        return displacy.render(doc, style="dep")
