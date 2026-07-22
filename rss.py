"""
RSS feed parser module.

Fetches and normalises news articles from RSS/Atom feeds into a pandas
DataFrame. Individual entry parsing errors are logged and skipped so one
malformed item does not abort the entire feed.
"""

import logging
from typing import Any
from urllib.parse import urlparse

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import ARTICLE_COLUMNS, DEFAULT_REQUEST_TIMEOUT, DEFAULT_USER_AGENT
from exceptions import RSSFeedError
from url_utils import MAX_RESPONSE_BYTES, is_allowed_url

logger = logging.getLogger(__name__)


def fetch_feed_bytes(feed_url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> bytes:
    """
    Download RSS/Atom bytes via the shared scrape HTTP stack.

    Applies SSRF allowlist, per-host rate limit, and response size limit.
    Local ``file://`` feeds are handled by ``RSSFeed.parse`` and never
    reach this helper.

    Raises:
        RSSFeedError: Blocked URL, empty body, HTTP failure, or network error.
    """
    from scraping import (
        _execute_http_get,
        _host_from_url,
        _rate_limiter,
        _read_limited_content,
    )

    if not is_allowed_url(feed_url):
        raise RSSFeedError(
            f"RSS URL blocked by SSRF guard: {feed_url}",
            feed_url=feed_url,
        )

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        _rate_limiter.wait(_host_from_url(feed_url))
        response = _execute_http_get(feed_url, timeout=timeout, headers=headers)
        response.raise_for_status()
        final_url = response.url or feed_url
        if not is_allowed_url(final_url):
            raise RSSFeedError(
                f"RSS redirect blocked by SSRF guard: {final_url}",
                feed_url=feed_url,
            )
        content = _read_limited_content(response, MAX_RESPONSE_BYTES)
        if not content:
            raise RSSFeedError(f"Empty RSS response: {feed_url}", feed_url=feed_url)
        return content
    except RSSFeedError:
        raise
    except requests.RequestException as exc:
        logger.exception("Failed to fetch RSS feed: %s", feed_url)
        raise RSSFeedError(
            f"Cannot fetch RSS feed: {feed_url}",
            feed_url=feed_url,
        ) from exc
    except OSError as exc:
        logger.exception("Network error fetching RSS feed: %s", feed_url)
        raise RSSFeedError(
            f"Cannot fetch RSS feed: {feed_url}",
            feed_url=feed_url,
        ) from exc
    except Exception as exc:
        # Unexpected scraper helper failures still surface as RSSFeedError.
        logger.exception("Unexpected RSS download failure: %s", feed_url)
        raise RSSFeedError(
            f"Cannot fetch RSS feed: {feed_url}",
            feed_url=feed_url,
        ) from exc


class RSSFeed:
    """Parse an RSS feed URL and return structured article metadata."""

    def __init__(self, feed_url: str, source: str = "") -> None:
        """
        Args:
            feed_url: Public RSS/Atom endpoint for a news publication.
            source: Human-readable media name stored in each row.
        """
        self.feed_url = feed_url
        self.source = source

    def parse(self) -> pd.DataFrame:
        """
        Download and parse the configured RSS feed.

        Returns:
            DataFrame with columns from ``ARTICLE_COLUMNS``.

        Raises:
            RSSFeedError: When the feed cannot be fetched or is empty/invalid.
        """
        parsed = urlparse(self.feed_url)
        try:
            if parsed.scheme in {"http", "https"}:
                payload = fetch_feed_bytes(self.feed_url)
                feed = feedparser.parse(payload)
            elif parsed.scheme in {"", "file"}:
                feed = feedparser.parse(self.feed_url)
            else:
                raise RSSFeedError(
                    f"Unsupported RSS URL scheme: {parsed.scheme}",
                    feed_url=self.feed_url,
                )
        except RSSFeedError:
            raise
        except Exception as exc:
            logger.exception("Failed to fetch RSS feed: %s", self.feed_url)
            raise RSSFeedError(
                f"Cannot fetch RSS feed: {self.feed_url}",
                feed_url=self.feed_url,
            ) from exc

        if getattr(feed, "bozo", False) and not feed.entries:
            message = getattr(
                getattr(feed, "bozo_exception", None),
                "message",
                "Invalid feed",
            )
            logger.warning(
                "RSS feed parse warning for %s: %s",
                self.feed_url,
                message,
            )
            raise RSSFeedError(
                f"Invalid RSS feed: {message}",
                feed_url=self.feed_url,
            )

        rows: list[dict[str, str | bool]] = []
        for entry in feed.entries:
            try:
                rows.append(self._entry_to_row(entry))
            except (AttributeError, TypeError, KeyError) as exc:
                # Skip a single broken item; keep processing the rest of the feed.
                logger.warning("Skipping malformed RSS entry: %s", exc)

        if not rows:
            logger.warning("RSS feed contained no valid entries: %s", self.feed_url)
            raise RSSFeedError(
                "RSS feed has no valid entries",
                feed_url=self.feed_url,
            )

        return pd.DataFrame(rows, columns=ARTICLE_COLUMNS)

    def _entry_to_row(self, entry: Any) -> dict[str, str | bool]:
        """Convert one feedparser entry into a normalised article dictionary."""
        content = self._extract_content(entry)
        category = self._extract_category(entry)

        return {
            "title": str(entry.get("title", "") or "").strip(),
            "link": str(entry.get("link", "") or "").strip(),
            "description": self._clean_description(entry.get("description", "")),
            "published": str(entry.get("published", "") or ""),
            "category": category,
            "content": content,
            "source": self.source,
            "scraped_ok": False,
        }

    @staticmethod
    def _extract_content(entry: Any) -> str:
        """
        Read embedded HTML content when the feed provides a ``content`` block.

        Many Ukrainian feeds expose only ``description``; this field is often
        empty and scraping fills ``content`` later.
        """
        try:
            if entry.get("content"):
                return str(entry["content"][0].get("value", "") or "").strip()
        except (TypeError, IndexError, AttributeError) as exc:
            logger.debug(
                "No embedded content for entry '%s': %s",
                entry.get("title", ""),
                exc,
            )
        return ""

    @staticmethod
    def _extract_category(entry: Any) -> str:
        """Normalise RSS category tags to a comma-separated string."""
        try:
            category = entry.get("category", "")
            if isinstance(category, list):
                return ", ".join(str(item) for item in category if item)
            return str(category).strip() if category else ""
        except (AttributeError, TypeError) as exc:
            logger.debug("Category extraction failed: %s", exc)
            return ""

    @staticmethod
    def _clean_description(description: str) -> str:
        """Strip HTML markup from RSS description fields."""
        if not description:
            return ""
        try:
            return BeautifulSoup(description, "html.parser").get_text(
                " ",
                strip=True,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Description HTML cleanup failed: %s", exc)
            return str(description).strip()
