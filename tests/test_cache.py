"""Unit tests for Streamlit cache wrappers."""

import pytest

from cache import load_articles
from exceptions import DataLoaderError, NLPAnalysisError


def test_load_articles_unknown_source_raises():
    if hasattr(load_articles, "clear"):
        load_articles.clear()

    with pytest.raises(DataLoaderError) as exc_info:
        load_articles("Невідоме медіа XYZ")

    assert exc_info.value.source_name == "Невідоме медіа XYZ"


def test_get_spacy_nlp_wraps_unexpected_errors(monkeypatch):
    import cache

    def broken_load():
        raise RuntimeError("disk full")

    monkeypatch.setattr("nlp.preprocessing.load_spacy_model", broken_load)

    if hasattr(cache.get_spacy_nlp, "clear"):
        cache.get_spacy_nlp.clear()

    with pytest.raises(NLPAnalysisError) as exc_info:
        cache.get_spacy_nlp()

    assert exc_info.value.step == "spacy_load"


def test_get_cosmus_pipeline_wraps_unexpected_errors(monkeypatch):
    import cache

    def broken_load():
        raise RuntimeError("model missing")

    monkeypatch.setattr("nlp.sentiment.load_cosmus_pipeline", broken_load)

    if hasattr(cache.get_cosmus_pipeline, "clear"):
        cache.get_cosmus_pipeline.clear()

    with pytest.raises(NLPAnalysisError) as exc_info:
        cache.get_cosmus_pipeline()

    assert exc_info.value.step == "cosmus_load"
