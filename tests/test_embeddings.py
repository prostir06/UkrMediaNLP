"""Unit tests for ``nlp.embeddings``."""

from __future__ import annotations

import math

import pytest

from exceptions import NLPAnalysisError
from nlp import embeddings


@pytest.fixture
def enable_embeddings(monkeypatch):
    """Turn on ALLOW_EMBEDDINGS for the duration of a test."""
    monkeypatch.setenv("ALLOW_EMBEDDINGS", "1")


def test_embeddings_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_EMBEDDINGS", raising=False)
    assert embeddings.embeddings_enabled() is False
    with pytest.raises(NLPAnalysisError, match="вимкнені"):
        embeddings.embed_texts(["привіт"])


@pytest.mark.parametrize("value", ["1", "true", "YES", "On", " 1 "])
def test_embeddings_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("ALLOW_EMBEDDINGS", value)
    assert embeddings.embeddings_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "", "no", "off"])
def test_embeddings_enabled_falsy(monkeypatch, value):
    monkeypatch.setenv("ALLOW_EMBEDDINGS", value)
    assert embeddings.embeddings_enabled() is False


def test_embeddings_enabled_soft_fails_on_env_error(monkeypatch):
    class BrokenEnviron(dict):
        def get(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("env broken")

    monkeypatch.setattr(embeddings.os, "environ", BrokenEnviron())
    assert embeddings.embeddings_enabled() is False


def test_hash_embed_deterministic(enable_embeddings):
    a = embeddings.embed_texts(["Київ новини"])
    b = embeddings.embed_texts(["Київ новини"])
    assert a == b
    assert len(a[0]) == 64
    assert embeddings.cosine_similarity(a[0], b[0]) == pytest.approx(1.0)


def test_embed_texts_empty_string_and_none(enable_embeddings):
    vectors = embeddings.embed_texts(["", None])  # type: ignore[list-item]
    assert len(vectors) == 2
    assert vectors[0] == [0.0] * 64
    assert vectors[1] == [0.0] * 64


def test_embed_texts_custom_dim(enable_embeddings):
    vectors = embeddings.embed_texts(["тест"], dim=16)
    assert len(vectors[0]) == 16
    norm = math.sqrt(sum(v * v for v in vectors[0]))
    assert norm == pytest.approx(1.0)


def test_embed_texts_rejects_non_positive_dim(enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="positive"):
        embeddings.embed_texts(["x"], dim=0)
    with pytest.raises(NLPAnalysisError, match="positive"):
        embeddings.embed_texts(["x"], dim=-3)


def test_embed_texts_rejects_bad_dim_type(enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="цілим"):
        embeddings.embed_texts(["x"], dim="wide")  # type: ignore[arg-type]


def test_embed_texts_rejects_non_iterable(enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="ітерованим"):
        embeddings.embed_texts(123)  # type: ignore[arg-type]


def test_cosine_handles_empty_and_mismatch():
    assert embeddings.cosine_similarity([], []) == 0.0
    assert embeddings.cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cosine_orthogonal_and_identical():
    assert embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert embeddings.cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_soft_fails_on_bad_values():
    assert embeddings.cosine_similarity([1.0], ["x"]) == 0.0  # type: ignore[list-item]


def test_similar_ukrainian_texts_rank_above_unrelated(enable_embeddings):
    query = embeddings.embed_texts(["Київ футбол"])[0]
    near = embeddings.embed_texts(["футбол у Києві"])[0]
    far = embeddings.embed_texts(["біржові котирування"])[0]
    assert embeddings.cosine_similarity(query, near) > embeddings.cosine_similarity(
        query, far
    )


def test_hash_embed_direct_empty():
    assert embeddings._hash_embed("", 8) == [0.0] * 8
