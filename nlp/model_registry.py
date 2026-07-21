"""
Shared model accessors — prefer Streamlit cache when available.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_spacy_cached():
    from nlp.preprocessing import load_spacy_model

    return load_spacy_model()


def resolve_spacy_nlp():
    """Return a spaCy pipeline, preferring Streamlit ``cache_resource``."""
    try:
        from cache import get_spacy_nlp

        return get_spacy_nlp()
    except Exception as exc:
        logger.debug("Falling back to process-local spaCy cache: %s", exc)
        return _load_spacy_cached()


def resolve_cosmus_pipeline():
    """Return COSMUS sentiment pipeline (Streamlit-cached when possible)."""
    try:
        from cache import get_cosmus_pipeline

        return get_cosmus_pipeline()
    except Exception as exc:
        logger.debug("Falling back to module COSMUS load: %s", exc)
        from nlp.sentiment import load_cosmus_pipeline

        return load_cosmus_pipeline()


def resolve_emotions_model():
    """Return emotions model tuple (Streamlit-cached when possible)."""
    try:
        from cache import get_emotions_model

        return get_emotions_model()
    except Exception as exc:
        logger.debug("Falling back to module emotions load: %s", exc)
        from nlp.sentiment import load_emotions_model

        return load_emotions_model()
