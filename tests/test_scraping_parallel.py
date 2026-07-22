"""Tests for parallel scraping helpers."""

from scraping import ThreadSafeRateLimiter, scrape_links_parallel


def test_rate_limiter_is_thread_safe():
    limiter = ThreadSafeRateLimiter(0.01)
    limiter.wait("example.com")
    limiter.wait("example.com")
    limiter.wait("other.com")


def test_rate_limiter_sleeps_outside_lock(monkeypatch):
    """Ensure sleep is not called while the lock is held."""
    limiter = ThreadSafeRateLimiter(0.05)
    holding_lock_during_sleep = []

    real_sleep = __import__("time").sleep

    def tracked_sleep(seconds):
        holding_lock_during_sleep.append(limiter._lock.locked())
        real_sleep(0)  # do not actually delay the test suite

    monkeypatch.setattr("scraping.time.sleep", tracked_sleep)

    limiter.wait("host-a")
    limiter.wait("host-a")  # second call should sleep

    assert holding_lock_during_sleep
    assert all(held is False for held in holding_lock_during_sleep)


def test_scrape_links_parallel_reports_progress(monkeypatch):
    def mock_full_text(url, scraper_name="generic", timeout=10):
        return f"Text from {url}"

    monkeypatch.setattr("scraping.full_text", mock_full_text)
    events = []

    def on_progress(done, total):
        events.append((done, total))

    links = [(0, "https://www.unian.ua/a"), (1, "https://www.unian.ua/b")]
    results = scrape_links_parallel(
        links,
        scraper_name="unian",
        max_workers=2,
        progress_callback=on_progress,
    )
    assert len(results) == 2
    assert events[-1] == (2, 2)
