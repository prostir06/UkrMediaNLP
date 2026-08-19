"""
Unit tests for durable corpus store (SQLite in-memory).

Covers happy paths, soft-fail branches, and SQLAlchemy error propagation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from corpus_store import db as db_mod
from corpus_store.db import (
    get_database_url,
    get_engine,
    is_store_configured,
    reset_engine,
    session_scope,
)
from corpus_store.models import Base
from corpus_store.repository import (
    _as_utc,
    _row_to_article_dict,
    canonical_url_hash,
    load_corpus_from_store,
    purge_older_than,
    upsert_articles,
)


@pytest.fixture
def session():
    """Provide an isolated in-memory SQLite session per test."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    sess = factory()
    try:
        yield sess
        sess.commit()
    finally:
        sess.close()
        engine.dispose()


# --- db helpers -------------------------------------------------------------


def test_is_store_configured_reads_env(monkeypatch):
    reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert is_store_configured() is False
    monkeypatch.setenv("DATABASE_URL", "   ")
    assert is_store_configured() is False
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x")
    assert is_store_configured() is True


def test_get_database_url_none_when_unset(monkeypatch):
    reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url() is None


def test_get_engine_requires_url(monkeypatch):
    reset_engine()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        get_engine()


def test_get_engine_caches_and_reset(monkeypatch):
    reset_engine()
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    first = get_engine()
    second = get_engine()
    assert first is second
    reset_engine()
    assert db_mod._ENGINE is None


def test_session_scope_commits_and_rollbacks(tmp_path, monkeypatch):
    reset_engine()
    url = f"sqlite:///{tmp_path / 't.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    engine = get_engine(force_new=True)
    Base.metadata.create_all(engine)

    with session_scope() as sess:
        assert upsert_articles(
            sess,
            pd.DataFrame(
                {
                    "title": ["T"],
                    "link": ["https://example.com/x"],
                    "content": [""],
                    "source": ["S"],
                    "category": ["Новини"],
                    "published": ["2024-01-01"],
                    "scraped_ok": [True],
                }
            ),
        ) == 1

    with session_scope() as sess:
        loaded = load_corpus_from_store(sess, sources=["S"])
        assert len(loaded) == 1

    with pytest.raises(RuntimeError, match="boom"):
        with session_scope() as sess:
            raise RuntimeError("boom")


# --- hash / date helpers ----------------------------------------------------


def test_canonical_url_hash_stable_and_strips():
    left = canonical_url_hash(" https://nv.ua/a ")
    right = canonical_url_hash("https://nv.ua/a")
    assert left == right
    assert len(left) == 64
    assert left != canonical_url_hash("https://nv.ua/b")


def test_canonical_url_hash_empty_and_none():
    assert canonical_url_hash("") == canonical_url_hash(None)  # type: ignore[arg-type]
    assert len(canonical_url_hash("")) == 64


def test_as_utc_parses_and_rejects_junk():
    aware = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert _as_utc(aware) == aware
    naive = datetime(2024, 1, 1)
    assert _as_utc(naive).tzinfo == timezone.utc
    assert _as_utc("2024-06-01T12:00:00+00:00") is not None
    assert _as_utc("not-a-date") is None
    assert _as_utc(None) is None
    assert _as_utc(float("nan")) is None


def test_row_to_article_dict_skips_empty_link():
    row = pd.Series({"link": "  ", "title": "x"})
    assert _row_to_article_dict(row, datetime.now(timezone.utc)) is None


def test_row_to_article_dict_defaults_title():
    row = pd.Series(
        {
            "link": "https://example.com/a",
            "title": "",
            "source": "NV",
            "category": "Новини",
            "content": "body",
            "scraped_ok": True,
        }
    )
    payload = _row_to_article_dict(row, datetime.now(timezone.utc))
    assert payload is not None
    assert payload["title"] == "(без заголовка)"


# --- upsert / load / purge --------------------------------------------------


def test_upsert_empty_and_missing_link(session: Session):
    assert upsert_articles(session, pd.DataFrame()) == 0
    assert upsert_articles(session, None) == 0  # type: ignore[arg-type]
    assert upsert_articles(session, pd.DataFrame({"title": ["x"]})) == 0


def test_upsert_skips_blank_urls(session: Session):
    df = pd.DataFrame(
        {
            "title": ["a", "b"],
            "link": ["", "https://example.com/ok"],
            "content": ["", "c"],
            "source": ["NV", "NV"],
            "category": ["Новини", "Новини"],
            "published": [None, "2024-01-01"],
            "scraped_ok": [False, True],
        }
    )
    assert upsert_articles(session, df) == 1
    session.commit()
    loaded = load_corpus_from_store(session, sources=["NV"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["link"] == "https://example.com/ok"


def test_upsert_dedupes_by_url_hash(session: Session):
    df = pd.DataFrame(
        {
            "title": ["One", "One updated"],
            "link": ["https://example.com/a", "https://example.com/a"],
            "content": ["body1", "body2"],
            "source": ["NV", "NV"],
            "category": ["Новини", "Новини"],
            "published": ["2024-01-01", "2024-01-02"],
            "scraped_ok": [True, True],
        }
    )
    assert upsert_articles(session, df) == 1
    session.commit()
    loaded = load_corpus_from_store(session, sources=["NV"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "One updated"
    assert loaded.iloc[0]["content"] == "body2"
    assert "One updated" in str(loaded.iloc[0]["search_blob"])
    assert "body2" in str(loaded.iloc[0]["search_blob"])


def test_upsert_preserves_published_when_incoming_null(session: Session):
    first = pd.DataFrame(
        {
            "title": ["T"],
            "link": ["https://example.com/p"],
            "content": ["c"],
            "source": ["NV"],
            "category": ["Новини"],
            "published": ["2024-06-01T12:00:00+00:00"],
            "scraped_ok": [True],
        }
    )
    upsert_articles(session, first)
    session.commit()

    second = pd.DataFrame(
        {
            "title": ["T2"],
            "link": ["https://example.com/p"],
            "content": ["c2"],
            "source": ["NV"],
            "category": ["Новини"],
            "published": [None],
            "scraped_ok": [False],
        }
    )
    upsert_articles(session, second)
    session.commit()
    loaded = load_corpus_from_store(session, sources=["NV"])
    assert "2024-06-01" in str(loaded.iloc[0]["published"])


def test_load_filters_by_source_category_and_dates(session: Session):
    frame = pd.DataFrame(
        {
            "title": ["in", "out-src", "out-cat", "missing-date"],
            "link": [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
                "https://example.com/4",
            ],
            "content": ["", "", "", ""],
            "source": ["NV", "Other", "NV", "NV"],
            "category": ["Новини", "Новини", "Спорт", "Новини"],
            "published": [
                "2024-06-15",
                "2024-06-15",
                "2024-06-15",
                None,
            ],
            "scraped_ok": [True, True, True, True],
        }
    )
    upsert_articles(session, frame)
    session.commit()

    loaded = load_corpus_from_store(
        session,
        sources=["NV"],
        categories=["Новини"],
        date_from=datetime(2024, 6, 1, tzinfo=timezone.utc),
        date_to=datetime(2024, 6, 30, tzinfo=timezone.utc),
        include_missing_dates=True,
    )
    titles = set(loaded["title"])
    assert "in" in titles
    assert "missing-date" in titles
    assert "out-src" not in titles
    assert "out-cat" not in titles

    strict = load_corpus_from_store(
        session,
        sources=["NV"],
        categories=["Новини"],
        date_from=datetime(2024, 6, 1, tzinfo=timezone.utc),
        date_to=datetime(2024, 6, 30, tzinfo=timezone.utc),
        include_missing_dates=False,
    )
    assert set(strict["title"]) == {"in"}


def test_purge_older_than_uses_coalesce(session: Session):
    now = datetime(2024, 8, 1, tzinfo=timezone.utc)
    old = pd.DataFrame(
        {
            "title": ["old"],
            "link": ["https://example.com/old"],
            "content": [""],
            "source": ["A"],
            "category": ["Новини"],
            "published": [(now - timedelta(days=100)).isoformat()],
            "scraped_ok": [True],
        }
    )
    recent = pd.DataFrame(
        {
            "title": ["new"],
            "link": ["https://example.com/new"],
            "content": [""],
            "source": ["A"],
            "category": ["Новини"],
            "published": [(now - timedelta(days=10)).isoformat()],
            "scraped_ok": [True],
        }
    )
    upsert_articles(session, old)
    upsert_articles(session, recent)
    session.commit()
    deleted = purge_older_than(session, days=90, now=now)
    session.commit()
    assert deleted == 1
    loaded = load_corpus_from_store(session, sources=["A"])
    assert list(loaded["title"]) == ["new"]


def test_purge_zero_days_is_noop(session: Session):
    assert purge_older_than(session, days=0) == 0
    assert purge_older_than(session, days=-5) == 0


def test_upsert_propagates_sqlalchemy_error(session: Session, monkeypatch):
    df = pd.DataFrame(
        {
            "title": ["T"],
            "link": ["https://example.com/err"],
            "content": [""],
            "source": ["NV"],
            "category": ["Новини"],
            "published": ["2024-01-01"],
            "scraped_ok": [True],
        }
    )

    def boom(*_args, **_kwargs):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(session, "execute", boom)
    with pytest.raises(SQLAlchemyError):
        upsert_articles(session, df)


def test_load_propagates_sqlalchemy_error(session: Session, monkeypatch):
    def boom(*_args, **_kwargs):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(session, "scalars", boom)
    with pytest.raises(SQLAlchemyError):
        load_corpus_from_store(session, sources=["NV"])


def test_purge_propagates_sqlalchemy_error(session: Session, monkeypatch):
    def boom(*_args, **_kwargs):
        raise SQLAlchemyError("db down")

    monkeypatch.setattr(session, "execute", boom)
    with pytest.raises(SQLAlchemyError):
        purge_older_than(session, days=90)
