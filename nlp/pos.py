"""
Part-of-speech analysis with spaCy for Ukrainian texts.

Figure builders live in ``ui.charts``; this module only extracts POS tags.
"""

import logging

from config import MAX_POS_ARTICLES, MAX_POS_CONTENT_CHARS
from nlp.model_registry import resolve_spacy_nlp

logger = logging.getLogger(__name__)


def extract_pos_batch(texts: list[str], nlp=None) -> list[list[str]]:
    """Return POS tags for each document using batched spaCy processing."""
    if nlp is None:
        nlp = resolve_spacy_nlp()

    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=32):
        tags = [
            token.pos_
            for token in doc
            if not token.is_space and not token.is_punct
        ]
        results.append(tags)
    return results


def sample_pos_corpus(content_series, max_articles: int = MAX_POS_ARTICLES) -> list[str]:
    """Limit POS analysis to a sample of truncated article bodies."""
    texts = []
    for text in content_series.head(max_articles):
        value = str(text or "").strip()
        if not value:
            continue
        texts.append(value[:MAX_POS_CONTENT_CHARS])
    return texts
