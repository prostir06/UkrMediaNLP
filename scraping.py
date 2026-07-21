"""
Web scraping utilities for extracting article body text from news pages.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from config import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_USER_AGENT,
    HTTP_MAX_RETRIES,
    HTTP_RETRY_BACKOFF,
    SCRAPE_DELAY_SECONDS,
    SCRAPE_MAX_WORKERS,
)
from exceptions import ScrapingError
from scrapers import GENERIC_SCRAPER, STRUCTURED_SCRAPER, get_scraper
from url_utils import MAX_RESPONSE_BYTES, is_allowed_url

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ThreadSafeRateLimiter:
    """Per-host rate limiter safe for concurrent worker threads."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self._last_request_by_host: dict[str, float] = {}

    def wait(self, host: str = "") -> None:
        key = (host or "_default").lower()
        with self._lock:
            now = time.monotonic()
            last = self._last_request_by_host.get(key, 0.0)
            elapsed = now - last
            if elapsed < self.delay_seconds:
                time.sleep(self.delay_seconds - elapsed)
            self._last_request_by_host[key] = time.monotonic()


_rate_limiter = ThreadSafeRateLimiter(SCRAPE_DELAY_SECONDS)


def _host_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _read_limited_content(response: requests.Response, max_bytes: int) -> bytes:
    """
    Read response body up to *max_bytes* to prevent memory exhaustion.

    Streaming reads stop early when the limit is exceeded; partial content
    up to the limit is still returned.
    """
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                logger.warning("Response exceeded %s bytes, truncating", max_bytes)
                break
            chunks.append(chunk)
    except (OSError, requests.exceptions.ChunkedEncodingError) as exc:
        logger.warning("Stream read interrupted: %s", exc)
    return b"".join(chunks)


def fetch_html(url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> bytes:
    """
    Download raw HTML for a URL with SSRF guard, size limit, and retries.
    """
    if not url:
        return b""

    if not is_allowed_url(url):
        logger.warning("Blocked fetch for disallowed URL: %s", url)
        return b""

    headers = {"User-Agent": DEFAULT_USER_AGENT}

    for attempt in range(HTTP_MAX_RETRIES):
        _rate_limiter.wait(_host_from_url(url))
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
                stream=True,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "Retryable HTTP %s for %s (attempt %s)",
                    response.status_code,
                    url,
                    attempt + 1,
                )
                time.sleep(HTTP_RETRY_BACKOFF * (2 ** attempt))
                continue

            if response.status_code == 403:
                logger.warning("Access forbidden (403) for %s", url)
                return b""
            if response.status_code == 404:
                logger.info("Article not found (404) for %s", url)
                return b""

            response.raise_for_status()

            final_url = response.url or url
            if not is_allowed_url(final_url):
                logger.warning("Blocked redirect target: %s", final_url)
                return b""

            return _read_limited_content(response, MAX_RESPONSE_BYTES)

        except requests.Timeout as exc:
            logger.warning(
                "Timeout for %s (attempt %s): %s",
                url,
                attempt + 1,
                exc,
            )
            time.sleep(HTTP_RETRY_BACKOFF * (2 ** attempt))
        except requests.HTTPError as exc:
            logger.warning("HTTP error for %s: %s", url, exc)
            return b""
        except requests.RequestException as exc:
            logger.warning("Request failed for %s: %s", url, exc)
            time.sleep(HTTP_RETRY_BACKOFF * (2 ** attempt))
        except OSError as exc:
            logger.warning("Network error for %s: %s", url, exc)
            return b""

    return b""


def full_text(
    url: str,
    scraper_name: str = "generic",
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> str:
    """Scrape the main article text from a news article URL."""
    if not url:
        return ""

    html = fetch_html(url, timeout=timeout)
    if not html:
        return ""

    scraper = get_scraper(scraper_name)
    try:
        text = STRUCTURED_SCRAPER.extract(html, url)
        if text.strip():
            return text.strip()

        text = scraper.extract(html, url)
        if text.strip():
            return text.strip()

        if scraper is not GENERIC_SCRAPER:
            text = GENERIC_SCRAPER.extract(html, url)
            if text.strip():
                return text.strip()
    except (ValueError, TypeError, KeyError) as exc:
        logger.warning("HTML parsing failed for %s: %s", url, exc)
        raise ScrapingError(f"Cannot parse HTML from {url}", url=url) from exc
    except Exception as exc:
        logger.exception("Unexpected scrape failure for %s", url)
        raise ScrapingError(f"Unexpected scrape error for {url}", url=url) from exc

    return ""


def scrape_links_parallel(
    links: list[tuple[int, str]],
    scraper_name: str,
    max_workers: int = SCRAPE_MAX_WORKERS,
) -> dict[int, str]:
    """
    Scrape multiple article URLs concurrently while preserving index mapping.

    Args:
        links: List of (dataframe_index, url) tuples.
        scraper_name: Scraper registry key.
        max_workers: Thread pool size.

    Returns:
        Mapping of index to scraped plain text (may be empty on failure).
    """
    results: dict[int, str] = {}

    if not links:
        return results

    workers = min(max_workers, len(links))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(full_text, url, scraper_name): index
            for index, url in links
            if url
        }
        for future in as_completed(future_map):
            index = future_map[future]
            try:
                results[index] = future.result()
            except ScrapingError as exc:
                logger.warning("Parallel scrape error index=%s: %s", index, exc)
                results[index] = ""
            except Exception as exc:
                logger.warning("Parallel scrape failed index=%s: %s", index, exc)
                results[index] = ""

    return results
