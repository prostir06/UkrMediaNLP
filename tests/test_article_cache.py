"""Tests for article SQLite cache."""

import pandas as pd

from article_cache import get_cached_articles, make_cache_key, store_articles


def test_article_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTICLE_CACHE_DIR", str(tmp_path))
    import article_cache as ac

    monkeypatch.setattr(ac, "DEFAULT_CACHE_DIR", tmp_path)

    key = make_cache_key("NV", "https://nv.ua/rss", 10)
    df = pd.DataFrame(
        [{"title": "T", "link": "https://nv.ua/1", "content": "body", "scraped_ok": True}]
    )
    store_articles(key, "NV", df)
    loaded = get_cached_articles(key, ttl_seconds=3600)
    assert loaded is not None
    assert loaded.iloc[0]["title"] == "T"


def test_article_cache_ttl_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTICLE_CACHE_DIR", str(tmp_path))
    import article_cache as ac

    monkeypatch.setattr(ac, "DEFAULT_CACHE_DIR", tmp_path)

    key = make_cache_key("NV", "https://nv.ua/rss", 5)
    store_articles(key, "NV", pd.DataFrame([{"title": "Old"}]))
    assert get_cached_articles(key, ttl_seconds=0) is None
