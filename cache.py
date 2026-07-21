"""
Streamlit cache wrappers — keeps core modules free of Streamlit imports.

Each wrapper delegates to a pure function in ``data_loader`` or ``nlp/*``.
Model-loading wrappers propagate ``NLPAnalysisError`` to the UI layer.
"""

import logging

import streamlit as st

from config import NEWS_SOURCES
from data_loader import fetch_articles
from exceptions import DataLoaderError, NLPAnalysisError

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner="Завантаження статей...")
def load_articles(source_name: str):
    """
    Cached wrapper around ``fetch_articles`` for the Streamlit UI.

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

    return fetch_articles(
        source_name=source_name,
        feed_url=config["rss_url"],
        scraper_name=config.get("scraper", "generic"),
    )


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


@st.cache_data
def cached_preprocess(texts):
    """Cached title/content preprocessing."""
    from nlp.preprocessing import preprocess_texts

    return preprocess_texts(texts)


@st.cache_data
def cached_top_words(corpus, n=10):
    from nlp.ngrams import get_top_n_words

    return get_top_n_words(corpus, n)


@st.cache_data
def cached_top_bigrams(corpus, n=10):
    from nlp.ngrams import get_top_n_bigram

    return get_top_n_bigram(corpus, n)


@st.cache_data
def cached_top_trigrams(corpus, n=10):
    from nlp.ngrams import get_top_n_trigram

    return get_top_n_trigram(corpus, n)


@st.cache_data
def cached_topic_modeling(content, number_topics=8, number_words=6):
    from nlp.topics import run_topic_modeling

    return run_topic_modeling(content, number_topics, number_words)


@st.cache_resource
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


@st.cache_resource
def get_emotions_model():
    """Load Ukrainian emotions classifier."""
    try:
        from nlp.sentiment import load_emotions_model

        return load_emotions_model()
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("Unexpected emotions model load failure")
        raise NLPAnalysisError(
            f"Не вдалося завантажити модель емоцій: {exc}",
            step="emotions_load",
        ) from exc
