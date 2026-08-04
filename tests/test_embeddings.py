"""Tests for optional semantic embeddings module."""

import pytest

from exceptions import NLPAnalysisError
from nlp import embeddings


def test_embeddings_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_EMBEDDINGS", raising=False)
    assert embeddings.embeddings_enabled() is False
    with pytest.raises(NLPAnalysisError, match="вимкнені"):
        embeddings.embed_texts(["привіт"])


def test_hash_embed_deterministic(monkeypatch):
    monkeypatch.setenv("ALLOW_EMBEDDINGS", "1")
    a = embeddings.embed_texts(["Київ новини"])
    b = embeddings.embed_texts(["Київ новини"])
    assert a == b
    assert len(a[0]) == 64
    assert embeddings.cosine_similarity(a[0], b[0]) == pytest.approx(1.0)


def test_cosine_handles_empty():
    assert embeddings.cosine_similarity([], []) == 0.0
    assert embeddings.cosine_similarity([1.0], [1.0, 2.0]) == 0.0
