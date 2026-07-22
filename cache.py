"""
Streamlit cache wrappers — keeps core modules free of Streamlit imports.

Article loading is a thin facade over SQLite-backed ``fetch_articles``
(shared TTL via ``ARTICLE_CACHE_TTL``). Model loaders use ``@st.cache_resource``.
"""

import logging

import streamlit as st

from config import NEWS_SOURCES
from data_loader import fetch_articles
from exceptions import DataLoaderError, NLPAnalysisError

logger = logging.getLogger(__name__)


def load_articles(source_name: str, progress_callback=None):
    """
    Load articles for a news source (SQLite TTL cache inside ``fetch_articles``).

    This is intentionally **not** ``@st.cache_data``: persistence and TTL live
    in ``article_cache`` so Streamlit stays a thin facade. Optional
    ``progress_callback`` receives ``(done, total)`` during scrape.

    Args:
        source_name: Key from ``NEWS_SOURCES``.
        progress_callback: Optional ``(done, total)`` scrape progress hook.

    Raises:
        DataLoaderError: Unknown source or RSS failure.
    """
    try:
        config = NEWS_SOURCES[source_name]
    except KeyError as exc:
        raise DataLoaderError(
            f"Невідоме джерело: {source_name}",
            source_name=source_name,
        ) from exc

    try:
        return fetch_articles(
            source_name=source_name,
            feed_url=config["rss_url"],
            scraper_name=config.get("scraper", "generic"),
            progress_callback=progress_callback,
        )
    except DataLoaderError:
        raise
    except Exception as exc:
        logger.exception("Unexpected article load failure for %s", source_name)
        raise DataLoaderError(
            f"Не вдалося завантажити статті: {exc}",
            source_name=source_name,
        ) from exc


@st.cache_resource
def get_spacy_nlp():
    """Load and cache the Ukrainian spaCy pipeline."""
    try:
        from nlp.preprocessing import load_spacy_model

        return load_spacy_model()
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unexpected spaCy load failure")
        raise NLPAnalysisError(
            f"Не вдалося завантажити spaCy: {exc}",
            step="spacy_load",
        ) from exc


@st.cache_resource(show_spinner="Завантаження моделі тональності (перший раз 5–10 хв)...")
def get_cosmus_pipeline():
    """Load RoBERTa-COSMUS sentiment pipeline."""
    try:
        from nlp.sentiment import load_cosmus_pipeline

        return load_cosmus_pipeline()
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unexpected COSMUS load failure")
        raise NLPAnalysisError(
            f"Не вдалося завантажити модель тональності: {exc}",
            step="cosmus_load",
        ) from exc


@st.cache_resource(show_spinner="Завантаження моделі емоцій (перший раз 5–10 хв)...")
def get_emotions_model():
    """Load Ukrainian emotions classifier."""
    try:
        from nlp.sentiment import load_emotions_model

        return load_emotions_model()
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unexpected emotions load failure")
        raise NLPAnalysisError(
            f"Не вдалося завантажити модель емоцій: {exc}",
            step="emotions_load",
        ) from exc
