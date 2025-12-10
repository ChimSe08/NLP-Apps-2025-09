"""Load spaCy English model for dependency parsing."""

import spacy


def load_en_model(model_name: str = "en_core_web_md"):
    """Load English spaCy model.

    Parameters
    ----------
    model_name : str
        Name of spaCy model to load, e.g. "en_core_web_sm" or "en_core_web_md".

    Returns
    -------
    nlp : spacy.Language
        Loaded spaCy pipeline.
    """
    return spacy.load(model_name)
