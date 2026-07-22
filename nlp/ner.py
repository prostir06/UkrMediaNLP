"""
Named entity recognition for Ukrainian news texts.

Figure builders live in ``ui.charts``; this module only extracts entities.
"""

import logging

from nlp.model_registry import resolve_spacy_nlp

logger = logging.getLogger(__name__)


def extract_entities_batch(texts: list[str], entity: str, nlp=None) -> list[list[str]]:
    """Extract named entities of one type from multiple texts using spaCy pipe."""
    if nlp is None:
        nlp = resolve_spacy_nlp()

    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=32):
        results.append([ent.text for ent in doc.ents if ent.label_ == entity])
    return results
