"""Unit tests for RSS HTTP download helper (shared scrape stack)."""

from unittest.mock import MagicMock

import pytest
import requests

from exceptions import RSSFeedError
from rss import fetch_feed_bytes


def test_fetch_feed_bytes_ssrf_blocked(monkeypatch):
    monkeypatch.setattr("rss.is_allowed_url", lambda url: False)

    with pytest.raises(RSSFeedError) as exc_info:
        fetch_feed_bytes("https://evil.example/feed.xml")

    assert "ssrf" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()


def test_fetch_feed_bytes_success(monkeypatch):
    monkeypatch.setattr("rss.is_allowed_url", lambda url: True)

    response = MagicMock()
    response.url = "https://nv.ua/ukr/rss/all.xml"
    response.raise_for_status = MagicMock()

    monkeypatch.setattr("scraping._rate_limiter.wait", lambda host: None)
    monkeypatch.setattr("scraping._execute_http_get", lambda *a, **k: response)
    monkeypatch.setattr("scraping._read_limited_content", lambda resp, max_b: b"<rss/>")

    assert fetch_feed_bytes("https://nv.ua/ukr/rss/all.xml") == b"<rss/>"


def test_fetch_feed_bytes_empty_body_raises(monkeypatch):
    monkeypatch.setattr("rss.is_allowed_url", lambda url: True)

    response = MagicMock()
    response.url = "https://nv.ua/ukr/rss/all.xml"
    response.raise_for_status = MagicMock()

    monkeypatch.setattr("scraping._rate_limiter.wait", lambda host: None)
    monkeypatch.setattr("scraping._execute_http_get", lambda *a, **k: response)
    monkeypatch.setattr("scraping._read_limited_content", lambda resp, max_b: b"")

    with pytest.raises(RSSFeedError) as exc_info:
        fetch_feed_bytes("https://nv.ua/ukr/rss/all.xml")

    assert "empty" in str(exc_info.value).lower()


def test_fetch_feed_bytes_redirect_ssrf_raises(monkeypatch):
    monkeypatch.setattr(
        "rss.is_allowed_url",
        lambda url: "nv.ua" in url,
    )

    response = MagicMock()
    response.url = "https://evil.example/steal"
    response.raise_for_status = MagicMock()

    monkeypatch.setattr("scraping._rate_limiter.wait", lambda host: None)
    monkeypatch.setattr("scraping._execute_http_get", lambda *a, **k: response)

    with pytest.raises(RSSFeedError) as exc_info:
        fetch_feed_bytes("https://nv.ua/ukr/rss/all.xml")

    assert "redirect" in str(exc_info.value).lower() or "blocked" in str(exc_info.value).lower()


def test_fetch_feed_bytes_http_error_raises(monkeypatch):
    monkeypatch.setattr("rss.is_allowed_url", lambda url: True)

    def boom(*args, **kwargs):
        raise requests.Timeout("slow")

    monkeypatch.setattr("scraping._rate_limiter.wait", lambda host: None)
    monkeypatch.setattr("scraping._execute_http_get", boom)

    with pytest.raises(RSSFeedError) as exc_info:
        fetch_feed_bytes("https://nv.ua/ukr/rss/all.xml")

    assert exc_info.value.feed_url == "https://nv.ua/ukr/rss/all.xml"


def test_fetch_feed_bytes_unexpected_error_raises(monkeypatch):
    monkeypatch.setattr("rss.is_allowed_url", lambda url: True)
    monkeypatch.setattr("scraping._rate_limiter.wait", lambda host: None)

    def boom(*args, **kwargs):
        raise RuntimeError("helper crashed")

    monkeypatch.setattr("scraping._execute_http_get", boom)

    with pytest.raises(RSSFeedError):
        fetch_feed_bytes("https://nv.ua/ukr/rss/all.xml")
