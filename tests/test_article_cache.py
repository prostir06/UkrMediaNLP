"""Unit tests for article_cache error paths and helpers."""

import json
import sqlite3

import pandas as pd
import pytest

import article_cache as ac


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "DEFAULT_CACHE_DIR", tmp_path)
    return tmp_path


def test_make_cache_key_is_stable(cache_dir):
    key_a = ac.make_cache_key("NV", "https://nv.ua/rss", 50)
    key_b = ac.make_cache_key("NV", "https://nv.ua/rss", 50)
    key_c = ac.make_cache_key("NV", "https://nv.ua/rss", 10)
    assert key_a == key_b
    assert key_a != key_c
    assert len(key_a) == 64


def test_make_cache_key_handles_bad_max_articles(cache_dir):
    key = ac.make_cache_key("NV", "https://nv.ua/rss", "oops")  # type: ignore[arg-type]
    assert isinstance(key, str) and len(key) == 64


def test_store_skips_empty_dataframe(cache_dir):
    ac.store_articles("key", "NV", pd.DataFrame())
    assert ac.get_cached_articles("key") is None


def test_get_cached_articles_empty_key(cache_dir):
    assert ac.get_cached_articles("") is None


def test_corrupt_payload_returns_none(cache_dir):
    key = ac.make_cache_key("NV", "https://nv.ua/rss", 5)
    path = ac._cache_path()
    with sqlite3.connect(str(path)) as conn:
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
        conn.execute(
            "INSERT INTO article_cache VALUES (?, ?, ?, ?)",
            (key, "NV", "{not-json", 1_700_000_000.0),
        )
        conn.commit()

    assert ac.get_cached_articles(key, ttl_seconds=10**9) is None


def test_non_list_payload_returns_none(cache_dir):
    key = ac.make_cache_key("TSN", "https://tsn.ua/rss", 3)
    with ac._connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO article_cache VALUES (?, ?, ?, ?)",
            (key, "TSN", json.dumps({"title": "x"}), 1_700_000_000.0),
        )
        conn.commit()
    assert ac.get_cached_articles(key, ttl_seconds=10**9) is None


def test_clear_expired_removes_old_rows(cache_dir):
    key = ac.make_cache_key("NV", "https://nv.ua/rss", 1)
    ac.store_articles(key, "NV", pd.DataFrame([{"title": "T"}]))
    deleted = ac.clear_expired(ttl_seconds=0)
    assert deleted >= 1
    assert ac.get_cached_articles(key, ttl_seconds=10**9) is None


def test_parse_ttl_seconds_invalid(monkeypatch):
    monkeypatch.setenv("ARTICLE_CACHE_TTL", "not-a-number")
    assert ac._parse_ttl_seconds() == 12 * 3600
