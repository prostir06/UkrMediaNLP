"""Tests for RSS feed parsing."""

from pathlib import Path

import pandas as pd
import pytest

from exceptions import RSSFeedError
from rss import RSSFeed

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_fixture_feed():
    feed_url = (FIXTURES / "sample_feed.xml").as_uri()
    df = RSSFeed(feed_url, source="Тест").parse()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "title", "link", "description", "published",
        "category", "content", "source", "scraped_ok",
    ]
    assert df.iloc[0]["title"] == "Президент України виступив у Києві"
    assert df.iloc[0]["category"] == "Політика"
    assert df.iloc[0]["source"] == "Тест"
    assert not df.iloc[0]["scraped_ok"]
    assert "Короткий опис" in df.iloc[0]["description"]


def test_parse_empty_feed_raises(monkeypatch):
    class EmptyFeed:
        bozo = True
        bozo_exception = type("E", (), {"message": "bad xml"})()
        entries = []

    monkeypatch.setattr("rss.feedparser.parse", lambda url: EmptyFeed())
    monkeypatch.setattr("url_utils.is_allowed_url", lambda url: True)

    with pytest.raises(RSSFeedError) as exc_info:
        RSSFeed("https://invalid.example/feed.xml").parse()

    assert exc_info.value.feed_url == "https://invalid.example/feed.xml"


def test_clean_description_strips_html():
    result = RSSFeed._clean_description("<p>Текст <b>новини</b></p>")
    assert result == "Текст новини"


def test_clean_description_handles_invalid_input():
    assert RSSFeed._clean_description("") == ""


def test_extract_category_from_list():
    entry = {"category": ["Політика", "Економіка"]}
    assert RSSFeed._extract_category(entry) == "Політика, Економіка"


def test_skips_malformed_entry(monkeypatch):
    class MixedFeed:
        bozo = False
        entries = [
            {"title": "OK", "link": "https://example.com/1", "description": "Desc"},
            None,
        ]

    monkeypatch.setattr("rss.feedparser.parse", lambda url: MixedFeed())
    monkeypatch.setattr("url_utils.is_allowed_url", lambda url: True)

    df = RSSFeed("https://example.com/rss").parse()
    assert len(df) == 1
    assert df.iloc[0]["title"] == "OK"


def test_all_malformed_entries_raises(monkeypatch):
    class BrokenFeed:
        bozo = False
        entries = [None, None]

    monkeypatch.setattr("rss.feedparser.parse", lambda url: BrokenFeed())
    monkeypatch.setattr("url_utils.is_allowed_url", lambda url: True)

    with pytest.raises(RSSFeedError) as exc_info:
        RSSFeed("https://example.com/empty.rss").parse()

    assert "no valid entries" in str(exc_info.value).lower()


def test_rss_ssrf_blocks_unknown_host():
    with pytest.raises(RSSFeedError) as exc_info:
        RSSFeed("https://evil.example.com/feed.xml").parse()
    assert "ssrf" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()
