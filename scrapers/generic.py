"""
Generic web scraping: find the DOM subtree with the most paragraph text.
"""

import logging

import pandas as pd
from bs4 import BeautifulSoup

from scrapers.base import BaseScraper

logger = logging.getLogger(__name__)


class GenericScraper(BaseScraper):
    """Heuristic paragraph-block extractor (same strategy as NLP_SyntA)."""

    def extract(self, html: bytes, url: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            body = soup.find("body")
            if body is None:
                return ""

            paragraphs = body.find_all("p")
            if not paragraphs:
                return ""

            rows = []
            for paragraph in paragraphs:
                text = paragraph.get_text(" ", strip=True)
                if len(text) < 30:
                    continue
                parents = [
                    f"{parent.name}#{parent.get('id', '')}.{'.'.join(parent.get('class', []) or [])}"
                    for parent in paragraph.parents
                    if parent is not None and getattr(parent, "name", None)
                ]
                parents.reverse()
                rows.append(
                    {
                        "parent_hierarchy": " -> ".join(parents),
                        "element_text": text,
                        "element_text_count": len(text),
                    }
                )

            if not rows:
                return ""

            blocks = pd.DataFrame(rows)
            grouped = (
                blocks.groupby("parent_hierarchy")["element_text_count"]
                .sum()
                .reset_index()
            )
            best = grouped.loc[grouped["element_text_count"].idxmax(), "parent_hierarchy"]
            texts = blocks.loc[blocks["parent_hierarchy"] == best, "element_text"]
            return "\n".join(texts.tolist())
        except Exception as exc:
            logger.warning("Generic scrape failed for %s: %s", url, exc)
            return ""
