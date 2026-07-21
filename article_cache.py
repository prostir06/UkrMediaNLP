"""
SQLite-backed article cache with TTL (persistent across Streamlit restarts).
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

DEFAULT_CACHE_DIR = Path(os.environ.get("ARTICLE_CACHE_DIR", ".cache/articles"))
DEFAULT_TTL_SECONDS = int(os.environ.get("ARTICLE_CACHE_TTL", str(12 * 3600)))


def _cache_path() -> Path:
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_CACHE_DIR / "articles.sqlite3"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_cache_path()), timeout=30)
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
    return conn


def make_cache_key(source_name: str, feed_url: str, max_articles: int) -> str:
    raw = f"{source_name}|{feed_url}|{max_articles}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_articles(cache_key: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> pd.DataFrame | None:
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM article_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        payload, created_at = row
        if time.time() - float(created_at) > ttl_seconds:
            return None
        records = json.loads(payload)
        return pd.DataFrame.from_records(records)
    except (OSError, sqlite3.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Article cache read failed: %s", exc)
        return None


def store_articles(cache_key: str, source_name: str, df: pd.DataFrame) -> None:
    try:
        payload = df.to_json(orient="records", force_ascii=False)
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO article_cache (cache_key, source_name, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (cache_key, source_name, payload, time.time()),
            )
            conn.commit()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        logger.warning("Article cache write failed: %s", exc)


def clear_expired(ttl_seconds: int = DEFAULT_TTL_SECONDS) -> int:
    cutoff = time.time() - ttl_seconds
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "DELETE FROM article_cache WHERE created_at < ?",
                (cutoff,),
            )
            conn.commit()
            return cursor.rowcount
    except (OSError, sqlite3.Error) as exc:
        logger.warning("Article cache cleanup failed: %s", exc)
        return 0
