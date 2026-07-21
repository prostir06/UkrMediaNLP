"""Rule-based sentiment baseline for Ukrainian news headlines."""

from __future__ import annotations

import re

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
)


def classify_news_sentiment(text: str) -> str:
    """
    Lightweight lexicon baseline for news headlines.

    Returns Ukrainian labels compatible with COSMUS UI mapping.
    """
    lowered = (text or "").lower()
    pos = sum(1 for marker in POSITIVE_MARKERS if marker in lowered)
    neg = sum(1 for marker in NEGATIVE_MARKERS if marker in lowered)

    if pos and neg:
        return "Змішана"
    if pos > neg:
        return "Позитивна"
    if neg > pos:
        return "Негативна"
    return "Нейтральна"


def classify_news_sentiment_batch(texts: list[str]) -> list[str]:
    return [classify_news_sentiment(text) for text in texts]


def headline_polarity_score(text: str) -> float:
    """Simple score in [-1, 1] for debugging / comparisons."""
    lowered = (text or "").lower()
    tokens = re.findall(r"[\w']+", lowered, flags=re.UNICODE)
    if not tokens:
        return 0.0
    pos = sum(1 for marker in POSITIVE_MARKERS if marker in lowered)
    neg = sum(1 for marker in NEGATIVE_MARKERS if marker in lowered)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
