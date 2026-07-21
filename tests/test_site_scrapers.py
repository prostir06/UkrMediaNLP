"""Error-handling tests for site-specific scrapers."""

from scrapers.site_scrapers import PRAVDA_SCRAPER, _extract_by_selectors


def test_extract_by_selectors_returns_empty_for_empty_html():
    assert _extract_by_selectors(b"", ["article p"]) == ""


def test_extract_by_selectors_skips_invalid_css():
    html = b"<!DOCTYPE html><html><body><p>Valid paragraph text here.</p></body></html>"
    # Invalid selector should be skipped; fallback selectors may still match.
    text = _extract_by_selectors(html, ["[[[invalid", "p"])
    assert "Valid paragraph" in text


def test_selector_scraper_handles_corrupt_html():
    text = PRAVDA_SCRAPER.extract(b"\xff\xfe not html", "https://example.com/x")
    assert text == ""
