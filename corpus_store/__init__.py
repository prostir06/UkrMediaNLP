"""Durable article corpus store (Postgres via SQLAlchemy)."""

from corpus_store.db import get_engine, is_store_configured, session_scope
from corpus_store.repository import (
    canonical_url_hash,
    load_corpus_from_store,
    purge_older_than,
    upsert_articles,
)

__all__ = [
    "canonical_url_hash",
    "get_engine",
    "is_store_configured",
    "load_corpus_from_store",
    "purge_older_than",
    "session_scope",
    "upsert_articles",
]
