"""Site-specific CSS selectors for Ukrainian media outlets."""

import logging

from bs4 import BeautifulSoup
from soupsieve.util import SelectorSyntaxError

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# Ignore boilerplate regions before applying CSS selectors (HTML5 landmarks).
REMOVED_TAGS = ("script", "style", "nav", "footer", "header", "aside")
MIN_PARAGRAPH_LENGTH = 20


def _extract_by_selectors(html: bytes, selectors: list[str]) -> str:
    """
    Try ordered CSS selectors and return joined paragraph text.

    Selectors are tried top-to-bottom; the first selector that yields
    paragraphs longer than ``MIN_PARAGRAPH_LENGTH`` characters wins.
    """
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except (TypeError, ValueError) as exc:
        logger.debug("BeautifulSoup failed to parse HTML: %s", exc)
        return ""

    try:
        for tag in soup.find_all(list(REMOVED_TAGS)):
            tag.decompose()
    except (AttributeError, TypeError) as exc:
        logger.debug("DOM cleanup failed: %s", exc)

    for selector in selectors:
        try:
            nodes = soup.select(selector)
        except (NotImplementedError, ValueError, SelectorSyntaxError) as exc:
            logger.debug("Invalid CSS selector %r: %s", selector, exc)
            continue

        if not nodes:
            continue

        texts = [node.get_text(" ", strip=True) for node in nodes]
        texts = [text for text in texts if len(text) > MIN_PARAGRAPH_LENGTH]
        if texts:
            return "\n".join(texts)

    return ""


class SelectorScraper(BaseScraper):
    """Scraper driven by ordered CSS selector lists."""

    def __init__(self, selectors: list[str]) -> None:
        self.selectors = selectors

    def extract(self, html: bytes, url: str) -> str:
        """Extract article text; returns empty string on parse failure."""
        try:
            return _extract_by_selectors(html, self.selectors)
        except Exception as exc:
            logger.warning("Selector scrape failed for %s: %s", url, exc)
            return ""


NV_SCRAPER = SelectorScraper(
    [
        "article .article-body p",
        ".article-body p",
        "article p",
    ]
)

PRAVDA_SCRAPER = SelectorScraper(
    [
        ".post_text p",
        ".post__text p",
        "article .text p",
        "article p",
    ]
)

LIGA_SCRAPER = SelectorScraper(
    [
        ".article-text p",
        ".article__content p",
        "article p",
    ]
)

TSN_SCRAPER = SelectorScraper(
    [
        ".c-article__body p",
        ".article-body p",
        "article p",
    ]
)

UNIAN_SCRAPER = SelectorScraper(
    [
        ".article__content p",
        ".article-content p",
        "article p",
    ]
)

RBC_SCRAPER = SelectorScraper(
    [
        ".article-text p",
        ".article__text p",
        "article p",
    ]
)

INTERFAX_SCRAPER = SelectorScraper(
    [
        ".article-body p",
        ".article p",
        "article p",
    ]
)

RADIOSVOBODA_SCRAPER = SelectorScraper(
    [
        ".wsw p",
        ".article-body p",
        "article p",
    ]
)
