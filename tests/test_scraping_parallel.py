"""Tests for parallel scraping helpers."""

from scraping import ThreadSafeRateLimiter, scrape_links_parallel


def test_rate_limiter_is_thread_safe():
    limiter = ThreadSafeRateLimiter(0.01)
    limiter.wait("example.com")
    limiter.wait("example.com")
    limiter.wait("other.com")


def test_scrape_links_parallel(monkeypatch):
    def mock_full_text(url, scraper_name="generic", timeout=10):
        return f"Text from {url}"

    monkeypatch.setattr("scraping.full_text", mock_full_text)
    links = [(0, "https://www.unian.ua/a"), (1, "https://www.unian.ua/b")]
    results = scrape_links_parallel(links, scraper_name="unian", max_workers=2)
    assert results[0].startswith("Text from")
    assert results[1].startswith("Text from")
