"""
Shared model accessors — prefer Streamlit ``cache_resource`` when available.

Keeps NLP modules free of top-level Streamlit imports while still benefiting
from session-scoped model caching inside the Streamlit process.

Fallback chain
--------------
1. ``cache.get_*`` (Streamlit ``@st.cache_resource``)
2. Process-local ``lru_cache`` / direct ``load_*`` for CLI and unit tests
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_spacy_cached():
    """Process-local spaCy singleton used outside Streamlit."""
    from nlp.preprocessing import load_spacy_model

    return load_spacy_model()


@lru_cache(maxsize=1)
def _load_cosmus_cached():
    """Process-local COSMUS pipeline singleton used outside Streamlit."""
    from nlp.sentiment import load_cosmus_pipeline

    return load_cosmus_pipeline()


@lru_cache(maxsize=1)
def _load_emotions_cached():
    """Process-local emotions model singleton used outside Streamlit."""
    from nlp.sentiment import load_emotions_model

    return load_emotions_model()


def resolve_spacy_nlp():
    """
    Return a spaCy pipeline, preferring Streamlit ``cache_resource``.

    Raises:
        NLPAnalysisError: Propagated from ``load_spacy_model`` when the model
            is missing and Streamlit cache is unavailable.
    """
    try:
        from cache import get_spacy_nlp

        return get_spacy_nlp()
    except Exception as exc:
        # ImportError, Streamlit runtime gaps, or cache miss → local load.
        logger.debug("Falling back to process-local spaCy cache: %s", exc)
        try:
            return _load_spacy_cached()
        except Exception:
            # Clear broken cache entry so the next call can retry install.
            clear_fn = getattr(_load_spacy_cached, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()
            raise


def resolve_cosmus_pipeline():
    """Return COSMUS sentiment pipeline (Streamlit-cached when possible)."""
    try:
        from cache import get_cosmus_pipeline

        return get_cosmus_pipeline()
    except Exception as exc:
        logger.debug("Falling back to process-local COSMUS cache: %s", exc)
        try:
            return _load_cosmus_cached()
        except Exception:
            clear_fn = getattr(_load_cosmus_cached, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()
            raise


def resolve_emotions_model():
    """Return ``(tokenizer, model, torch)`` for the emotions classifier."""
    try:
        from cache import get_emotions_model

        return get_emotions_model()
    except Exception as exc:
        logger.debug("Falling back to process-local emotions cache: %s", exc)
        try:
            return _load_emotions_cached()
        except Exception:
            clear_fn = getattr(_load_emotions_cached, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()
            raise


def clear_process_model_caches() -> None:
    """Clear process-local LRU caches (useful in tests)."""
    _load_spacy_cached.cache_clear()
    _load_cosmus_cached.cache_clear()
    _load_emotions_cached.cache_clear()
