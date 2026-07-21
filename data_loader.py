"""
Cached data-loading layer: RSS parsing plus article scraping.

``fetch_articles`` is the pure core function (no Streamlit). The UI wraps it
via ``cache.load_articles``.
"""

import logging

import pandas as pd

from config import ARTICLE_COLUMNS, ARTICLE_CACHE_ENABLED, MAX_ARTICLES
from exceptions import DataLoaderError, RSSFeedError, ScrapingError
from rss import RSSFeed
from scraping import scrape_links_parallel

logger = logging.getLogger(__name__)


def _fallback_content(row: pd.Series) -> str:
    """
    Use RSS description when full-page scraping did not yield body text.

    Prefers any existing ``content`` field (e.g. from embedded RSS HTML)
    before falling back to the plain-text ``description``.
    """
    try:
        content = str(row.get("content", "") or "").strip()
        if content:
            return content
        return str(row.get("description", "") or "").strip()
    except (AttributeError, TypeError) as exc:
        logger.debug("Fallback content read failed: %s", exc)
        return ""


def _apply_scrape_result(df: pd.DataFrame, index: int, scraped_text: str) -> None:
    """Write scraped or fallback content into one dataframe row in-place."""
    try:
        if scraped_text:
            df.at[index, "content"] = scraped_text
            df.at[index, "scraped_ok"] = True
            return

        fallback = _fallback_content(df.loc[index])
        if fallback:
            df.at[index, "content"] = fallback
        df.at[index, "scraped_ok"] = False
    except (KeyError, IndexError) as exc:
        logger.warning("Cannot update row %s: %s", index, exc)


def _ensure_article_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee expected columns exist before downstream NLP steps."""
    for column in ARTICLE_COLUMNS:
        if column not in df.columns:
            if column == "scraped_ok":
                df[column] = False
            else:
                df[column] = ""
    return df


def fetch_articles(
    source_name: str,
    feed_url: str,
    scraper_name: str,
    max_articles: int | None = None,
) -> pd.DataFrame:
    """
    Load articles from RSS and enrich each row with scraped page content.

    Args:
        source_name: Human-readable media label stored in each row.
        feed_url: RSS/Atom endpoint URL.
        scraper_name: Key from ``SCRAPER_REGISTRY`` (e.g. ``pravda``).
        max_articles: Optional cap; defaults to ``MAX_ARTICLES`` from config.

    Returns:
        DataFrame with ``ARTICLE_COLUMNS`` and attrs ``total_in_feed``,
        ``max_articles``.

    Raises:
        DataLoaderError: When the RSS feed cannot be fetched or parsed.
    """
    limit = max_articles if max_articles is not None else MAX_ARTICLES

    if ARTICLE_CACHE_ENABLED:
        try:
            from article_cache import get_cached_articles, make_cache_key, store_articles

            cache_key = make_cache_key(source_name, feed_url, limit)
            cached = get_cached_articles(cache_key)
            if cached is not None and not cached.empty:
                logger.info("Article cache hit for %s", source_name)
                cached.attrs["total_in_feed"] = cached.attrs.get(
                    "total_in_feed",
                    len(cached),
                )
                cached.attrs["max_articles"] = limit
                cached.attrs["from_cache"] = True
                return cached
        except Exception as exc:
            logger.debug("Article cache unavailable: %s", exc)

    try:
        feed = RSSFeed(feed_url, source=source_name)
        df = feed.parse().reset_index(drop=True)
    except RSSFeedError as exc:
        logger.exception("RSS feed error for %s", feed_url)
        raise DataLoaderError(
            f"RSS недоступний: {exc}",
            source_name=source_name,
        ) from exc

    df = _ensure_article_columns(df)

    total_in_feed = len(df)
    if total_in_feed > limit:
        df = df.head(limit).reset_index(drop=True)

    df.attrs["total_in_feed"] = total_in_feed
    df.attrs["max_articles"] = limit
    df.attrs["from_cache"] = False

    links = [
        (index, str(df.at[index, "link"]))
        for index in df.index
        if df.at[index, "link"]
    ]

    try:
        scraped = scrape_links_parallel(links, scraper_name=scraper_name)
        for index, text in scraped.items():
            _apply_scrape_result(df, index, text)
    except Exception as exc:
        # Thread-pool failures should not abort the entire load; scrape sequentially.
        logger.exception("Parallel scraping failed, falling back to sequential: %s", exc)
        for index, link in links:
            try:
                from scraping import full_text

                _apply_scrape_result(
                    df,
                    index,
                    full_text(link, scraper_name=scraper_name),
                )
            except ScrapingError as scrape_exc:
                logger.warning("Scraping error for %s: %s", link, scrape_exc)
                _apply_scrape_result(df, index, "")
            except Exception as row_exc:
                logger.warning("Skipping article %s: %s", link, row_exc)
                _apply_scrape_result(df, index, "")

    scraped_count = int(df["scraped_ok"].sum()) if len(df) else 0
    logger.info(
        "Scrape stats for %s: %s/%s successful",
        source_name,
        scraped_count,
        len(df),
    )

    try:
        result = df.dropna(subset=["title"]).reset_index(drop=True)
    except KeyError as exc:
        logger.warning("Missing title column after load: %s", exc)
        result = df.reset_index(drop=True)

    if ARTICLE_CACHE_ENABLED:
        try:
            from article_cache import make_cache_key, store_articles

            store_articles(make_cache_key(source_name, feed_url, limit), source_name, result)
        except Exception as exc:
            logger.debug("Article cache store skipped: %s", exc)

    return result
