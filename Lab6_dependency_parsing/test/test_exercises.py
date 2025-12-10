import pytest

from src.model_loader import load_en_model
from src.visualize_dep import analyze_sentence
from src.exercises import find_main_verb, extract_noun_chunks_simple, get_path_to_root
from src.extract_relations import find_svo_triplets, find_noun_adjectives


@pytest.fixture(scope="session")
def nlp():
    # Try medium model first, fallback to small if needed.
    try:
        return load_en_model("en_core_web_md")
    except OSError:
        return load_en_model("en_core_web_sm")


def test_find_main_verb(nlp):
    doc = analyze_sentence(nlp, "Apple is looking at buying U.K. startup for $1 billion")
    main_verb = find_main_verb(doc)
    assert main_verb.text.lower() in {"looking", "buying", "is"}


def test_svo_triplets(nlp):
    doc = analyze_sentence(nlp, "The cat chased the mouse.")
    triplets = find_svo_triplets(doc)
    assert any(
        s.lower() == "cat" and v.lower() == "chased" and o.lower() == "mouse"
        for s, v, o in triplets
    )


def test_noun_adjectives(nlp):
    doc = analyze_sentence(nlp, "The big white cat sat on the small mat.")
    mapping = find_noun_adjectives(doc)
    assert "cat" in mapping and len(mapping["cat"]) >= 1


def test_noun_chunks_simple(nlp):
    doc = analyze_sentence(
        nlp, "The big, fluffy white cat is sleeping on the warm mat."
    )
    chunks = extract_noun_chunks_simple(doc)
    assert any(any(t.text.lower() == "cat" for t in chunk) for chunk in chunks)


def test_path_to_root(nlp):
    doc = analyze_sentence(nlp, "The cat chased the mouse.")
    token_mouse = [t for t in doc if t.text.lower() == "mouse"][0]
    path = get_path_to_root(token_mouse)
    assert path[-1].dep_ == "ROOT"
