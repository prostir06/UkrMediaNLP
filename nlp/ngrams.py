"""
N-gram frequency extraction for Ukrainian text.
"""

import logging
from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer

from nlp.text_utils import UKRAINIAN_TOKEN_PATTERN, as_text_list, single_token_stopwords

logger = logging.getLogger(__name__)


def _vectorize_ngrams(
    corpus: Iterable[str],
    ngram_range: tuple[int, int],
    top_n: int | None,
    lemmatize: bool = False,
) -> list[tuple[str, int]]:
    corpus_list = [str(item) for item in corpus if str(item).strip()]
    if not corpus_list:
        return []

    if lemmatize:
        try:
            from nlp.model_registry import resolve_spacy_nlp
            from nlp.text_utils import lemmatize_texts

            nlp = resolve_spacy_nlp()
            corpus_list = lemmatize_texts(corpus_list, nlp)
        except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Lemmatization skipped: %s", exc)

    try:
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=single_token_stopwords(),
            token_pattern=UKRAINIAN_TOKEN_PATTERN,
            lowercase=True,
        )
        matrix = vectorizer.fit_transform(corpus_list)
        totals = matrix.sum(axis=0)
        words_freq = [
            (word, int(totals[0, idx]))
            for word, idx in vectorizer.vocabulary_.items()
        ]
        words_freq.sort(key=lambda item: item[1], reverse=True)
        return words_freq[:top_n] if top_n else words_freq
    except ValueError as exc:
        logger.warning("N-gram extraction failed: %s", exc)
        return []


def get_top_n_words(corpus, n: int | None = 10) -> list[tuple[str, int]]:
    return _vectorize_ngrams(as_text_list(corpus), (1, 1), n, lemmatize=True)


def get_top_n_bigram(corpus, n: int | None = 10) -> list[tuple[str, int]]:
    return _vectorize_ngrams(as_text_list(corpus), (2, 2), n, lemmatize=True)


def get_top_n_trigram(corpus, n: int | None = 10) -> list[tuple[str, int]]:
    return _vectorize_ngrams(as_text_list(corpus), (3, 3), n, lemmatize=True)
