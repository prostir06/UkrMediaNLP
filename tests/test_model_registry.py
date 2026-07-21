"""Unit tests for nlp.model_registry fallbacks."""

import pytest

import nlp.model_registry as registry
from exceptions import NLPAnalysisError


def test_resolve_spacy_falls_back_to_local(monkeypatch):
    registry.clear_process_model_caches()

    sentinel = object()

    def boom():
        raise RuntimeError("no streamlit cache")

    monkeypatch.setattr(
        "cache.get_spacy_nlp",
        boom,
        raising=False,
    )

    # Force ImportError path by making cache import fail via fake module attr.
    import sys
    import types

    fake_cache = types.ModuleType("cache")

    def broken_get():
        raise RuntimeError("cache broken")

    fake_cache.get_spacy_nlp = broken_get
    monkeypatch.setitem(sys.modules, "cache", fake_cache)
    monkeypatch.setattr(registry, "_load_spacy_cached", lambda: sentinel)

    assert registry.resolve_spacy_nlp() is sentinel


def test_resolve_spacy_propagates_missing_model(monkeypatch):
    registry.clear_process_model_caches()

    import sys
    import types

    fake_cache = types.ModuleType("cache")

    def broken_get():
        raise RuntimeError("cache broken")

    fake_cache.get_spacy_nlp = broken_get
    monkeypatch.setitem(sys.modules, "cache", fake_cache)

    def raise_missing():
        raise NLPAnalysisError("missing", step="spacy_load")

    monkeypatch.setattr(registry, "_load_spacy_cached", raise_missing)

    with pytest.raises(NLPAnalysisError):
        registry.resolve_spacy_nlp()


def test_resolve_cosmus_falls_back(monkeypatch):
    import sys
    import types

    fake_cache = types.ModuleType("cache")
    fake_cache.get_cosmus_pipeline = lambda: (_ for _ in ()).throw(RuntimeError("x"))
    monkeypatch.setitem(sys.modules, "cache", fake_cache)

    sentinel = object()
    monkeypatch.setattr(
        "nlp.sentiment.load_cosmus_pipeline",
        lambda: sentinel,
    )
    assert registry.resolve_cosmus_pipeline() is sentinel


def test_clear_process_model_caches():
    registry.clear_process_model_caches()
