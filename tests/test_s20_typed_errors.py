"""
S20 unit tests: typed-error fallbacks in corpus helpers, store, and data_loader.

HTML5 / CSS3 / StandardJS do not apply — Python-only (PEP 8 / pytest).
"""

from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from corpus_store.repository import _as_utc, _row_to_article_dict, canonical_url_hash
from exceptions import NLPAnalysisError
from nlp.corpus import cap_corpus, filter_by_date, merge_source_frames, search_corpus
from nlp.news_sentiment import classify_news_sentiment


def test_filter_by_date_typed_failure_returns_empty(monkeypatch):
    df = pd.DataFrame({"title": ["a"], "published": ["2024-01-01"]})
    monkeypatch.setattr(
        "nlp.corpus.parse_published",
        lambda _v: (_ for _ in ()).throw(TypeError("bad date")),
    )
    out = filter_by_date(df, date_from=pd.Timestamp("2024-01-01"), date_to=None)
    assert out.empty


def test_cap_corpus_typed_sort_fallback(monkeypatch):
    df = pd.DataFrame({"title": ["a", "b"], "published": ["2024-01-02", "2024-01-01"]})
    monkeypatch.setattr(
        pd.DataFrame,
        "sort_values",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("sort")),
    )
    out = cap_corpus(df, max_rows=1)
    assert len(out) == 1


def test_merge_skips_uncopyable_frame():
    class Bad:
        empty = False

        def copy(self):
            raise AttributeError("no copy")

    merged = merge_source_frames([Bad()], max_rows=5)  # type: ignore[list-item]
    assert merged.empty


def test_search_corpus_wraps_unexpected_as_nlp_error(monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["футбол"],
            "content": ["x"],
            "published": ["2024-01-01"],
        }
    )

    def boom(*_a, **_k):
        raise RuntimeError("vectorizer exploded")

    monkeypatch.setattr("nlp.corpus._series_matches", boom)
    with pytest.raises(NLPAnalysisError, match="не вдався"):
        search_corpus(df, "футбол")


def test_canonical_url_hash_typed_failure(monkeypatch):
    calls = {"n": 0}
    real = __import__("hashlib").sha256

    def flaky(data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("hash")
        return real(data)

    monkeypatch.setattr("corpus_store.repository.hashlib.sha256", flaky)
    digest = canonical_url_hash("https://example.com")
    assert len(digest) == 64


def test_as_utc_typed_failure_returns_none():
    class Boom:
        def __str__(self):
            raise TypeError("no str")

    assert _as_utc(Boom()) is None


def test_row_to_article_dict_skips_unreadable_mapping():
    from datetime import datetime

    class BadDict(dict):
        def get(self, *_a, **_k):
            raise TypeError("broken")

    assert _row_to_article_dict(BadDict(), datetime.now(timezone.utc)) is None


def test_news_sentiment_typed_failure_is_neutral(monkeypatch):
    monkeypatch.setattr(
        "nlp.news_sentiment._safe_lower",
        lambda _t: (_ for _ in ()).throw(TypeError("x")),
    )
    assert classify_news_sentiment("перемога") == "Нейтральна"


def test_ensure_published_dt_typed_map_failure(monkeypatch):
    from nlp.corpus import ensure_published_dt

    df = pd.DataFrame({"published": ["2024-01-01"]})
    monkeypatch.setattr(
        pd.Series,
        "map",
        lambda *a, **k: (_ for _ in ()).throw(AttributeError("map")),
    )
    out = ensure_published_dt(df)
    assert out["published_dt"].isna().all()


def test_embeddings_coerce_none_and_text():
    from nlp.embeddings import _coerce_text

    assert _coerce_text(None) == ""
    assert _coerce_text("Київ") == "Київ"


def test_lemma_err_tuple_includes_runtime_error():
    from nlp.corpus import _LEMMA_ERR

    assert RuntimeError in _LEMMA_ERR
    assert NLPAnalysisError in _LEMMA_ERR

