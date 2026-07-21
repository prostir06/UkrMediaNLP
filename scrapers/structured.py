"""Extract article body from JSON-LD structured data (schema.org)."""

import json
import logging
import re

from bs4 import BeautifulSoup

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

# Minimum body length avoids returning RSS-style short descriptions.
MIN_BODY_LENGTH = 80
ARTICLE_TYPES = frozenset({"NewsArticle", "Article", "BlogPosting"})


class StructuredDataScraper(BaseScraper):
    """
    Parse schema.org ``NewsArticle`` / ``Article`` JSON-LD blocks.

    Many Ukrainian news sites embed machine-readable metadata in
    ``<script type="application/ld+json">`` tags before CSS extraction.
    """

    def extract(self, html: bytes, url: str) -> str:
        """
        Scan all JSON-LD scripts and return the first valid article body.

        Returns an empty string when no suitable block is found or parsing fails.
        """
        if not html:
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")
        except (TypeError, ValueError) as exc:
            logger.debug("HTML parse failed for JSON-LD on %s: %s", url, exc)
            return ""

        try:
            for script in soup.find_all("script", type="application/ld+json"):
                raw = script.string or script.get_text()
                if not raw:
                    continue
                text = self._parse_json_ld(raw.strip())
                if text:
                    return text
        except (AttributeError, TypeError) as exc:
            logger.debug("JSON-LD DOM walk failed for %s: %s", url, exc)

        return ""

    @staticmethod
    def _parse_json_ld(raw: str) -> str:
        """
        Parse one JSON-LD payload and extract ``articleBody`` or ``description``.

        Supports top-level objects, arrays, and ``@graph`` containers.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.debug("Invalid JSON-LD payload: %s", exc)
            return ""

        items: list[dict] = []
        if isinstance(data, list):
            items.extend(item for item in data if isinstance(item, dict))
        elif isinstance(data, dict):
            graph = data.get("@graph")
            if isinstance(graph, list):
                items.extend(item for item in graph if isinstance(item, dict))
            else:
                items.append(data)

        for item in items:
            body = _extract_body_from_item(item)
            if body:
                return body
        return ""


def _extract_body_from_item(item: dict) -> str:
    """Return normalised article text from one schema.org object."""
    schema_type = item.get("@type", "")
    types = schema_type if isinstance(schema_type, list) else [schema_type]
    if not any(t in ARTICLE_TYPES for t in types):
        return ""

    body = item.get("articleBody") or item.get("description") or ""
    if not isinstance(body, str):
        return ""

    cleaned = re.sub(r"\s+", " ", body).strip()
    if len(cleaned) > MIN_BODY_LENGTH:
        return cleaned
    return ""
