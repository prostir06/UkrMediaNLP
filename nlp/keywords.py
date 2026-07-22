"""
Keyphrase and keyword extraction for Ukrainian media articles.
"""

import logging
import re
from collections import Counter
from typing import Sequence

from nlp.text_utils import UKRAINIAN_TOKEN_PATTERN, single_token_stopwords

logger = logging.getLogger(__name__)


def extract_keywords(
    texts: Sequence[str],
    top_n: int = 15,
    min_word_len: int = 3,
) -> list[tuple[str, int]]:
    """
    Extract top keywords from Ukrainian text titles or content.

    Args:
        texts: Collection of Ukrainian headlines or article bodies.
        top_n: Maximum number of keywords to return.
        min_word_len: Minimum word length filter.

    Returns:
        List of (keyword, frequency) tuples.
    """
    if texts is None or len(texts) == 0:
        return []

    stopwords = single_token_stopwords()
    counter: Counter[str] = Counter()

    for text in texts:
        text_str = str(text or "").strip()
        if not text_str:
            continue

        words = re.findall(UKRAINIAN_TOKEN_PATTERN, text_str.lower(), flags=re.UNICODE)
        filtered = [
            word for word in words
            if len(word) >= min_word_len and word not in stopwords and not word.isdigit()
        ]
        counter.update(filtered)

    return counter.most_common(top_n)

