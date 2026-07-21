"""Tests for article scraping."""

from pathlib import Path

import pytest

from exceptions import ScrapingError
from scrapers import get_scraper
from scrapers.generic import GenericScraper
from scrapers.site_scrapers import PRAVDA_SCRAPER, UNIAN_SCRAPER
from scraping import fetch_html, full_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_generic_scraper_extracts_main_paragraphs():
    html = (FIXTURES / "sample_article.html").read_bytes()
    text = GenericScraper().extract(html, "https://example.com/news/1")

    assert "енергетичної безпеки" in text
    assert "Міністр зазначив" in text
    assert "Меню сайту" not in text
    assert "Футер сайту" not in text


def test_selector_scraper_falls_back_gracefully():
    html = (
        "<html><body><article>"
        "<p>Текст статті для перевірки селектора.</p>"
        "</article></body></html>"
    ).encode("utf-8")
    text = PRAVDA_SCRAPER.extract(html, "https://example.com/a")
    assert "Текст статті" in text

    text = UNIAN_SCRAPER.extract(html, "https://example.com/b")
    assert "Текст статті" in text


def test_full_text_returns_empty_for_invalid_url(monkeypatch):
    monkeypatch.setattr("scraping.fetch_html", lambda url, timeout=10: b"")
    assert full_text("https://example.com/missing", scraper_name="pravda") == ""


def test_full_text_raises_scraping_error_on_parser_failure(monkeypatch):
    class BrokenScraper:
        def extract(self, html, url):
            raise ValueError("broken parser")

    monkeypatch.setattr("scraping.fetch_html", lambda url, timeout=10: b"<html></html>")
    monkeypatch.setattr("scraping.get_scraper", lambda name: BrokenScraper())

    with pytest.raises(ScrapingError):
        full_text("https://example.com/article", scraper_name="pravda")


def test_fetch_html_handles_timeout(monkeypatch):
    import requests

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr("scraping.requests.get", raise_timeout)
    assert fetch_html("https://example.com") == b""


def test_get_scraper_unknown_name_returns_generic():
    scraper = get_scraper("unknown-media")
    assert scraper.__class__.__name__ == "GenericScraper"


def test_fetch_html_returns_empty_on_403(monkeypatch):
    class Response403:
        status_code = 403
        url = "https://www.unian.ua/article"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scraping.is_allowed_url", lambda url: True)
    monkeypatch.setattr("scraping.requests.get", lambda *a, **k: Response403())
    assert fetch_html("https://www.unian.ua/article") == b""


def test_fetch_html_returns_empty_on_404(monkeypatch):
    class Response404:
        status_code = 404
        url = "https://www.unian.ua/missing"

        def raise_for_status(self):
            return None

    monkeypatch.setattr("scraping.is_allowed_url", lambda url: True)
    monkeypatch.setattr("scraping.requests.get", lambda *a, **k: Response404())
    assert fetch_html("https://www.unian.ua/missing") == b""


def test_read_limited_content_handles_stream_error(monkeypatch):
    import scraping

    class BrokenResponse:
        def iter_content(self, chunk_size=65536):
            raise OSError("connection reset")

    result = scraping._read_limited_content(BrokenResponse(), max_bytes=1024)
    assert result == b""
