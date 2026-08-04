"""Postgres dialect smoke tests (skipped without TEST_DATABASE_URL)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from corpus_store.models import Article, Base
from corpus_store.repository import (
    load_corpus_from_store,
    purge_older_than,
    upsert_articles,
)

pytestmark = pytest.mark.postgres

_URL = (os.environ.get("TEST_DATABASE_URL") or "").strip()


@pytest.fixture(scope="module")
def pg_session():
    if not _URL:
        pytest.skip("TEST_DATABASE_URL not set")
    engine = create_engine(_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    finally:
        session.rollback()
        session.execute(text("DELETE FROM articles"))
        session.commit()
        session.close()
        engine.dispose()


@pytest.fixture(autouse=True)
def _clean_articles(pg_session):
    pg_session.execute(text("DELETE FROM articles"))
    pg_session.commit()
    yield
    pg_session.execute(text("DELETE FROM articles"))
    pg_session.commit()


def test_postgres_upsert_and_load(pg_session):
    df = pd.DataFrame(
        {
            "title": ["PG One", "PG One updated"],
            "link": ["https://example.com/pg-a", "https://example.com/pg-a"],
            "content": ["a", "b"],
            "source": ["NV", "NV"],
            "category": ["Новини", "Новини"],
            "published": ["2024-05-01", "2024-05-02"],
            "scraped_ok": [True, True],
        }
    )
    assert upsert_articles(pg_session, df) == 1
    pg_session.commit()
    loaded = load_corpus_from_store(pg_session, sources=["NV"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "PG One updated"
    assert loaded.iloc[0]["content"] == "b"
    assert "search_blob" in loaded.columns
    assert "PG One updated" in str(loaded.iloc[0]["search_blob"])


def test_postgres_null_published_preserves_existing(pg_session):
    first = pd.DataFrame(
        {
            "title": ["Dated"],
            "link": ["https://example.com/pg-null"],
            "content": ["body"],
            "source": ["NV"],
            "category": ["Новини"],
            "published": ["2024-06-15"],
            "scraped_ok": [True],
        }
    )
    second = pd.DataFrame(
        {
            "title": ["Dated again"],
            "link": ["https://example.com/pg-null"],
            "content": ["body2"],
            "source": ["NV"],
            "category": ["Новини"],
            "published": [None],
            "scraped_ok": [True],
        }
    )
    assert upsert_articles(pg_session, first) == 1
    pg_session.commit()
    assert upsert_articles(pg_session, second) == 1
    pg_session.commit()
    loaded = load_corpus_from_store(pg_session, sources=["NV"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "Dated again"
    assert "2024-06-15" in str(loaded.iloc[0]["published"])


def test_postgres_conflict_upsert_updates_fields(pg_session):
    row = pd.DataFrame(
        {
            "title": ["v1"],
            "link": ["https://example.com/pg-conflict"],
            "content": ["c1"],
            "source": ["BBC"],
            "category": ["Новини"],
            "published": ["2024-01-01"],
            "scraped_ok": [False],
        }
    )
    upsert_articles(pg_session, row)
    pg_session.commit()
    row2 = row.copy()
    row2["title"] = ["v2"]
    row2["content"] = ["c2"]
    row2["scraped_ok"] = [True]
    upsert_articles(pg_session, row2)
    pg_session.commit()
    loaded = load_corpus_from_store(pg_session, sources=["BBC"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "v2"
    assert loaded.iloc[0]["content"] == "c2"
    assert bool(loaded.iloc[0]["scraped_ok"]) is True


def test_postgres_purge_uses_coalesce_published_scraped(pg_session):
    now = datetime(2024, 8, 1, tzinfo=timezone.utc)
    old_pub = Article(
        url_hash="a" * 64,
        url="https://example.com/old-pub",
        source="NV",
        category="Новини",
        title="old pub",
        content="x",
        search_blob="old pub\nx",
        published_at=now - timedelta(days=120),
        scraped_at=now,
        scraped_ok=True,
    )
    old_scrape_only = Article(
        url_hash="b" * 64,
        url="https://example.com/old-scrape",
        source="NV",
        category="Новини",
        title="old scrape",
        content="y",
        search_blob="old scrape\ny",
        published_at=None,
        scraped_at=now - timedelta(days=100),
        scraped_ok=True,
    )
    recent = Article(
        url_hash="c" * 64,
        url="https://example.com/recent",
        source="NV",
        category="Новини",
        title="recent",
        content="z",
        search_blob="recent\nz",
        published_at=now - timedelta(days=10),
        scraped_at=now,
        scraped_ok=True,
    )
    pg_session.add_all([old_pub, old_scrape_only, recent])
    pg_session.commit()
    deleted = purge_older_than(pg_session, days=90, now=now)
    pg_session.commit()
    assert deleted == 2
    loaded = load_corpus_from_store(pg_session, sources=["NV"])
    assert len(loaded) == 1
    assert loaded.iloc[0]["title"] == "recent"


def test_postgres_search_blob_roundtrip(pg_session):
    df = pd.DataFrame(
        {
            "title": ["Blob Title"],
            "link": ["https://example.com/pg-blob"],
            "content": ["blob body"],
            "source": ["NV"],
            "category": ["Новини"],
            "published": ["2024-07-01"],
            "scraped_ok": [True],
            "search_blob": ["custom persisted blob"],
        }
    )
    upsert_articles(pg_session, df)
    pg_session.commit()
    loaded = load_corpus_from_store(pg_session, sources=["NV"])
    assert loaded.iloc[0]["search_blob"] == "custom persisted blob"
