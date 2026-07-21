"""Tests for data loading and article enrichment."""

from pathlib import Path

import pandas as pd
import pytest

from data_loader import _fallback_content, fetch_articles
from exceptions import DataLoaderError, RSSFeedError

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_rss_url():
    return (FIXTURES / "sample_feed.xml").as_uri()


def test_fallback_content_prefers_existing_content():
    row = pd.Series({"content": "Повний текст", "description": "Опис"})
    assert _fallback_content(row) == "Повний текст"


def test_fallback_content_uses_description():
    row = pd.Series({"content": "", "description": "Короткий опис"})
    assert _fallback_content(row) == "Короткий опис"


def test_fetch_articles_enriches_from_rss_fixture(sample_rss_url, monkeypatch):
    monkeypatch.setattr(
        "data_loader.scrape_links_parallel",
        lambda links, scraper_name: {
            index: f"Scraped body for {url}" for index, url in links
        },
    )

    df = fetch_articles("Тест", sample_rss_url, "pravda")
    assert len(df) == 2
    assert bool(df.iloc[0]["scraped_ok"]) is True
    assert "Scraped body" in df.iloc[0]["content"]


def test_fetch_articles_uses_description_when_scrape_empty(sample_rss_url, monkeypatch):
    monkeypatch.setattr(
        "data_loader.scrape_links_parallel",
        lambda links, scraper_name: {index: "" for index, _ in links},
    )

    df = fetch_articles("Тест", sample_rss_url, "pravda")
    assert len(df) == 2
    assert bool(df.iloc[0]["scraped_ok"]) is False
    assert df.iloc[0]["content"]


def test_fetch_articles_raises_on_rss_error(sample_rss_url, monkeypatch):
    class BrokenFeed:
        def __init__(self, feed_url, source=""):
            self.feed_url = feed_url

        def parse(self):
            raise RSSFeedError("bad feed", feed_url=self.feed_url)

    monkeypatch.setattr("data_loader.RSSFeed", BrokenFeed)

    with pytest.raises(DataLoaderError):
        fetch_articles("Тест", sample_rss_url, "generic")


def test_fetch_articles_respects_max_articles(sample_rss_url, monkeypatch):
    monkeypatch.setattr("data_loader.scrape_links_parallel", lambda links, scraper: {})

    df = fetch_articles("Тест", sample_rss_url, "pravda", max_articles=1)
    assert len(df) == 1
    assert df.attrs["total_in_feed"] == 2


def test_load_articles_unknown_source_raises():
    from cache import load_articles

    if hasattr(load_articles, "clear"):
        load_articles.clear()

    with pytest.raises(DataLoaderError) as exc_info:
        load_articles("Невідоме медіа")

    assert exc_info.value.source_name == "Невідоме медіа"


def test_fallback_content_handles_broken_row():
    row = pd.Series({"unexpected": "value"})
    assert _fallback_content(row) == ""


def test_apply_scrape_result_uses_description_fallback():
    df = pd.DataFrame(
        [{"title": "T", "content": "", "description": "RSS опис", "scraped_ok": False}]
    )
    from data_loader import _apply_scrape_result

    _apply_scrape_result(df, 0, "")
    assert df.at[0, "content"] == "RSS опис"
    assert not df.at[0, "scraped_ok"]


def test_fetch_articles_adds_missing_columns(sample_rss_url, monkeypatch):
    class MinimalFeed:
        def __init__(self, feed_url, source=""):
            pass

        def parse(self):
            return pd.DataFrame([{"title": "Only title", "link": "https://x.com/1"}])

    monkeypatch.setattr("data_loader.RSSFeed", MinimalFeed)
    monkeypatch.setattr("data_loader.scrape_links_parallel", lambda links, scraper: {})

    df = fetch_articles("Тест", sample_rss_url, "generic")
    assert "scraped_ok" in df.columns
    assert "description" in df.columns
