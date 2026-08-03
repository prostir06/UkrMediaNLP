"""Postgres dialect smoke tests (skipped without TEST_DATABASE_URL)."""

from __future__ import annotations

import os

import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from corpus_store.models import Base
from corpus_store.repository import load_corpus_from_store, upsert_articles

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
