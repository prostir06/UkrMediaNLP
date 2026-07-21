"""Per-site HTML fixture tests for Ukrainian media scrapers."""

from pathlib import Path

from scrapers.site_scrapers import (
    LIGA_SCRAPER,
    NV_SCRAPER,
    PRAVDA_SCRAPER,
    TSN_SCRAPER,
    UNIAN_SCRAPER,
)

SITES = Path(__file__).parent / "fixtures" / "sites"


def test_pravda_fixture_extracts_body():
    html = (SITES / "pravda.html").read_bytes()
    text = PRAVDA_SCRAPER.extract(html, "https://www.pravda.com.ua/news/test")
    assert "енергетичної безпеки" in text


def test_unian_fixture_extracts_body():
    html = (SITES / "unian.html").read_bytes()
    text = UNIAN_SCRAPER.extract(html, "https://www.unian.ua/news/test")
    assert "енергопостачання" in text


def test_nv_fixture_extracts_body():
    html = (SITES / "nv.html").read_bytes()
    text = NV_SCRAPER.extract(html, "https://nv.ua/ukr/news/test")
    assert "цифровізації" in text


def test_liga_fixture_extracts_body():
    html = (SITES / "liga.html").read_bytes()
    text = LIGA_SCRAPER.extract(html, "https://news.liga.net/ua/test")
    assert "оподаткування" in text


def test_tsn_fixture_extracts_body():
    html = (SITES / "tsn.html").read_bytes()
    text = TSN_SCRAPER.extract(html, "https://tsn.ua/test")
    assert "медалі" in text
