"""
SQLite-backed article cache with TTL (persistent across Streamlit restarts).

Design
------
* Key = SHA-256 of ``source|feed_url|max_articles``.
* Payload = JSON records from a pandas DataFrame.
* Expired rows are ignored on read; call ``clear_expired()`` to reclaim disk.

Failures never abort the scrape pipeline: every public function catches
I/O / SQLite / JSON errors and logs a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_ttl_seconds() -> int:
    """Parse ``ARTICLE_CACHE_TTL`` safely (default 12 hours)."""
    raw = os.environ.get("ARTICLE_CACHE_TTL", str(12 * 3600))
    try:
        value = int(raw)
        return value if value > 0 else 12 * 3600
    except (TypeError, ValueError):
        logger.warning("Invalid ARTICLE_CACHE_TTL=%r; using 12h default", raw)
        return 12 * 3600


DEFAULT_CACHE_DIR = Path(os.environ.get("ARTICLE_CACHE_DIR", ".cache/articles"))
DEFAULT_TTL_SECONDS = _parse_ttl_seconds()


def _cache_path() -> Path:
    """Ensure cache directory exists and return the SQLite file path."""
    try:
        DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create cache directory %s: %s", DEFAULT_CACHE_DIR, exc)
        raise
    return DEFAULT_CACHE_DIR / "articles.sqlite3"


def _connect() -> sqlite3.Connection:
    """Open SQLite and ensure the schema exists."""
    conn = sqlite3.connect(str(_cache_path()), timeout=30)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_cache (
                cache_key TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def make_cache_key(source_name: str, feed_url: str, max_articles: int) -> str:
    """
    Build a stable cache key for one load configuration.

    Args:
        source_name: Media label (e.g. ``NV``).
        feed_url: RSS endpoint.
        max_articles: Slice limit applied after RSS parse.
    """
    try:
        raw = f"{source_name}|{feed_url}|{int(max_articles)}"
    except (TypeError, ValueError) as exc:
        logger.debug("Falling back to string max_articles in cache key: %s", exc)
        raw = f"{source_name}|{feed_url}|{max_articles}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_articles(
    cache_key: str,
    ttl_seconds: int | None = None,
) -> pd.DataFrame | None:
    """
    Return a cached DataFrame or ``None`` on miss / expiry / corruption.

    Never raises to callers — scrape must continue without cache.
    """
    ttl = DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    if not cache_key:
        return None

    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM article_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None

        payload, created_at = row
        age = time.time() - float(created_at)
        if age > ttl:
            logger.debug("Cache entry expired (age=%.0fs, ttl=%s)", age, ttl)
            return None

        records = json.loads(payload)
        if not isinstance(records, list):
            logger.warning("Cache payload is not a list; ignoring")
            return None
        return pd.DataFrame.from_records(records)
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Article cache read failed: %s", exc)
        return None


def store_articles(cache_key: str, source_name: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame as JSON. Silently skips on empty or invalid input."""
    if not cache_key or df is None or getattr(df, "empty", True):
        return

    try:
        payload = df.to_json(orient="records", force_ascii=False)
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO article_cache
                    (cache_key, source_name, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, source_name, payload, time.time()),
            )
            conn.commit()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.warning("Article cache write failed: %s", exc)


def clear_expired(ttl_seconds: int | None = None) -> int:
    """Delete expired rows; returns number of deleted rows (0 on failure)."""
    ttl = DEFAULT_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    cutoff = time.time() - ttl
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "DELETE FROM article_cache WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return int(cursor.rowcount or 0)
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.warning("Article cache cleanup failed: %s", exc)
        return 0
