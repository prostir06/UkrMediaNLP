"""Unit tests for semantic corpus search (S18)."""

from __future__ import annotations

import pandas as pd
import pytest

from exceptions import NLPAnalysisError
from nlp.corpus import (
    _parse_semantic_search_params,
    _resolve_semantic_text_column,
    search_corpus_semantic,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Київ футбол", "Біржові котирування", "футбол у Києві", "Погода"],
            "content": ["", "", "", ""],
            "source": ["A", "B", "C", "D"],
            "published": ["2024-01-01"] * 4,
            "link": ["u1", "u2", "u3", "u4"],
        }
    )


@pytest.fixture
def enable_embeddings(monkeypatch):
    monkeypatch.setenv("ALLOW_EMBEDDINGS", "1")


def test_search_corpus_semantic_empty_query_returns_empty(sample_df, enable_embeddings):
    out = search_corpus_semantic(sample_df, "   ")
    assert out.empty
    assert list(out.columns) == list(sample_df.columns)


def test_search_corpus_semantic_empty_dataframe(enable_embeddings):
    out = search_corpus_semantic(pd.DataFrame(), "тест")
    assert out.empty


def test_search_corpus_semantic_ranks_similar_higher(sample_df, enable_embeddings):
    out = search_corpus_semantic(sample_df, "Київ футбол", min_score=0.0, top_k=4)
    assert len(out) >= 2
    titles = out["title"].tolist()
    assert titles[0] in {"Київ футбол", "футбол у Києві"}
    assert "relevance" in out.columns
    assert "snippet" in out.columns
    assert out["relevance"].iloc[0] >= out["relevance"].iloc[-1]


def test_search_corpus_semantic_min_score_filters(sample_df, enable_embeddings):
    out = search_corpus_semantic(sample_df, "Київ футбол", min_score=0.99, top_k=10)
    assert out.empty or (out["relevance"] >= 0.99).all()


def test_search_corpus_semantic_top_k_limits(sample_df, enable_embeddings):
    out = search_corpus_semantic(sample_df, "футбол", min_score=0.0, top_k=1)
    assert len(out) <= 1


def test_search_corpus_semantic_disabled_raises(sample_df, monkeypatch):
    monkeypatch.delenv("ALLOW_EMBEDDINGS", raising=False)
    with pytest.raises(NLPAnalysisError, match="вимкнені"):
        search_corpus_semantic(sample_df, "тест")


def test_search_corpus_semantic_rejects_bad_top_k(sample_df, enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="top_k"):
        search_corpus_semantic(sample_df, "тест", top_k=0)


def test_search_corpus_semantic_rejects_bad_min_score(sample_df, enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="min_score"):
        search_corpus_semantic(sample_df, "тест", min_score=1.5)


def test_search_corpus_semantic_rejects_non_numeric_params(sample_df, enable_embeddings):
    with pytest.raises(NLPAnalysisError, match="числами"):
        search_corpus_semantic(sample_df, "тест", min_score="high")  # type: ignore[arg-type]


def test_parse_semantic_search_params_ok():
    score, limit = _parse_semantic_search_params(0.1, 5)
    assert score == pytest.approx(0.1)
    assert limit == 5


def test_resolve_semantic_text_column_fallback():
    df = pd.DataFrame({"search_blob": ["a"], "title": ["t"]})
    assert _resolve_semantic_text_column(df, "missing") == "search_blob"


def test_resolve_semantic_text_column_missing_raises():
    df = pd.DataFrame({"title": ["t"]})
    with pytest.raises(NLPAnalysisError, match="відсутня"):
        _resolve_semantic_text_column(df, "content")


def test_search_corpus_semantic_wraps_embedding_batch_mismatch(
    sample_df, enable_embeddings, monkeypatch
):
    def bad_embed(texts):
        return [[0.0] * 64]  # only query vector

    monkeypatch.setattr("nlp.embeddings.embed_texts", bad_embed)
    with pytest.raises(NLPAnalysisError, match="Невідповідність"):
        search_corpus_semantic(sample_df, "тест")
