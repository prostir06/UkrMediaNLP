"""
Rule-based sentiment baseline for Ukrainian news headlines.

This is intentionally lexicon-based (no transformers) so it works on
Streamlit Cloud free tier and as a comparison baseline next to COSMUS.

Labels match the UI vocabulary used by COSMUS:
``Позитивна`` / ``Негативна`` / ``Нейтральна`` / ``Змішана``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Stem-like markers; word-boundary regex keeps matches on token edges.
POSITIVE_MARKERS = (
    "перемога",
    "успіх",
    "зростання",
    "відновлення",
    "підтримка",
    "допомога",
    "угода",
    "прогрес",
    "рекорд",
    "покращення",
    "звільнення",
    "рятувальн",
    "мирн",
    "відбудов",
    "інвестиц",
)

NEGATIVE_MARKERS = (
    "війн",
    "атака",
    "обстріл",
    "загинул",
    "смерть",
    "криза",
    "вибух",
    "катастроф",
    "аварія",
    "скандал",
    "корупц",
    "санкці",
    "загроза",
    "руйнуван",
    "жертв",
    "окупац",
    "ракетн",
    "евакуац",
)


def _compile_marker_pattern(markers: tuple[str, ...]) -> re.Pattern[str]:
    """Build a case-insensitive word-boundary pattern for lexicon stems."""
    escaped = [re.escape(marker) for marker in markers if marker]
    # `(?<!\w)` / `(?!\w)` work better for Ukrainian stems than `\b`.
    body = "|".join(escaped)
    return re.compile(rf"(?<!\w)(?:{body})\w*", flags=re.UNICODE | re.IGNORECASE)


_POSITIVE_RE = _compile_marker_pattern(POSITIVE_MARKERS)
_NEGATIVE_RE = _compile_marker_pattern(NEGATIVE_MARKERS)


def _safe_lower(text: object) -> str:
    """Coerce arbitrary input to a lowercased string."""
    try:
        return str(text or "").lower()
    except Exception as exc:  # pragma: no cover - extremely defensive
        logger.debug("Cannot stringify headline: %s", exc)
        return ""


def _count_markers(text: str, pattern: re.Pattern[str]) -> int:
    try:
        return len(pattern.findall(text))
    except re.error as exc:
        logger.debug("Marker match failed: %s", exc)
        return 0


def classify_news_sentiment(text: str) -> str:
    """
    Lightweight lexicon baseline for news headlines.

    Returns Ukrainian labels compatible with COSMUS UI mapping.
    """
    try:
        lowered = _safe_lower(text)
        pos = _count_markers(lowered, _POSITIVE_RE)
        neg = _count_markers(lowered, _NEGATIVE_RE)

        if pos and neg:
            return "Змішана"
        if pos > neg:
            return "Позитивна"
        if neg > pos:
            return "Негативна"
        return "Нейтральна"
    except Exception as exc:
        logger.warning("News sentiment classification failed: %s", exc)
        return "Нейтральна"


def classify_news_sentiment_batch(texts: list[str] | None) -> list[str]:
    """Classify many headlines; empty/invalid input yields an empty list."""
    if texts is None or len(texts) == 0:
        return []

    try:
        return [classify_news_sentiment(text) for text in texts]
    except TypeError as exc:
        logger.warning("News sentiment batch failed: %s", exc)
        return []


def headline_polarity_score(text: str) -> float:
    """
    Simple polarity score in ``[-1, 1]`` for debugging / comparisons.

    ``0.0`` means neutral or unparseable input.
    """
    try:
        lowered = _safe_lower(text)
        tokens = re.findall(r"[\w']+", lowered, flags=re.UNICODE)
        if not tokens:
            return 0.0
        pos = _count_markers(lowered, _POSITIVE_RE)
        neg = _count_markers(lowered, _NEGATIVE_RE)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total
    except (TypeError, ValueError, re.error) as exc:
        logger.debug("Polarity score failed: %s", exc)
        return 0.0
