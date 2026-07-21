"""
Readability and text statistics for Ukrainian.

Standard English readability indices (Flesch, SMOG) are not available for
Ukrainian, so this module exposes language-agnostic structural metrics and a
simple vowel-based syllable estimate.
"""

import logging
import re
from typing import Iterable

from nlp.preprocessing import normalise_whitespace

logger = logging.getLogger(__name__)

UKRAINIAN_VOWELS = set("аеиіїоуяюєёыэАЕИІЇОУЯЮЄЁЫЭ")
WORD_PATTERN = re.compile(r"[\wЁА-яІіЇїЄєҐґ']+", re.UNICODE)
SENTENCE_SPLIT = re.compile(r"[.!?…]+")

METRIC_LABELS_UA = {
    "char_count": "Кількість символів",
    "word_count": "Кількість слів",
    "sentence_count": "Кількість речень",
    "avg_word_length": "Середня довжина слова",
    "avg_sentence_length": "Середня довжина речення (слів)",
    "syllable_count": "Орієнтовна кількість складів",
    "lexical_diversity": "Лексичне різноманіття (TTR)",
}


def _words(text: str) -> list[str]:
    """Tokenise Ukrainian/Latin words from plain text."""
    return WORD_PATTERN.findall(text)


def _sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation."""
    parts = SENTENCE_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def _syllable_estimate(word: str) -> int:
    """
    Approximate syllable count by counting Ukrainian vowel characters.

    Every word is assumed to contain at least one syllable.
    """
    count = sum(1 for char in word.lower() if char in UKRAINIAN_VOWELS)
    return max(count, 1)


def get_textstat_metrics(text: str) -> list[tuple[str, float | int | str]]:
    """
    Compute readability metrics for one text sample.

    Returns:
        Ordered list of ``(Ukrainian label, value)`` pairs.
    """
    try:
        text = normalise_whitespace(text)
        if not text:
            return []

        words = _words(text)
        sentences = _sentences(text)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        unique_words = len({word.lower() for word in words})
        syllables = sum(_syllable_estimate(word) for word in words)

        raw_metrics = {
            "char_count": len(text),
            "word_count": word_count,
            "sentence_count": len(sentences) if sentences else 0,
            "avg_word_length": (
                round(sum(len(word) for word in words) / word_count, 2)
                if word_count
                else 0
            ),
            "avg_sentence_length": (
                round(word_count / sentence_count, 2) if word_count else 0
            ),
            "syllable_count": syllables,
            "lexical_diversity": (
                round(unique_words / word_count, 3) if word_count else 0
            ),
        }

        return [
            (METRIC_LABELS_UA[key], value)
            for key, value in raw_metrics.items()
        ]
    except (TypeError, ValueError) as exc:
        logger.warning("Text statistics calculation failed: %s", exc)
        return []


def aggregate_corpus_metrics(
    texts: Iterable[str],
) -> list[tuple[str, float | int | str]]:
    """Compute metrics over the concatenation of many article bodies."""
    combined = normalise_whitespace(
        " ".join(str(item) for item in texts if str(item).strip()),
    )
    return get_textstat_metrics(combined)
