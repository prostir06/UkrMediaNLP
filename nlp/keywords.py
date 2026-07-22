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
    lemmatize: bool = True,
) -> list[tuple[str, int]]:
    """
    Extract top keywords from Ukrainian text titles or content.

    Args:
        texts: Collection of Ukrainian headlines or article bodies.
        top_n: Maximum number of keywords to return.
        min_word_len: Minimum word length filter.
        lemmatize: When True, lemmatize via spaCy before counting.

    Returns:
        List of (keyword, frequency) tuples.
    """
    if texts is None or len(texts) == 0:
        return []

    text_list = [str(text or "").strip() for text in texts]
    text_list = [text for text in text_list if text]
    if not text_list:
        return []

    if lemmatize:
        try:
            from nlp.model_registry import resolve_spacy_nlp
            from nlp.text_utils import lemmatize_texts

            nlp = resolve_spacy_nlp()
            text_list = lemmatize_texts(text_list, nlp)
        except Exception as exc:
            logger.debug("Keyword lemmatization skipped: %s", exc)

    stopwords = single_token_stopwords()
    counter: Counter[str] = Counter()

    for text in text_list:
        words = re.findall(UKRAINIAN_TOKEN_PATTERN, text.lower(), flags=re.UNICODE)
        filtered = [
            word
            for word in words
            if len(word) >= min_word_len and word not in stopwords and not word.isdigit()
        ]
        counter.update(filtered)

    return counter.most_common(top_n)
