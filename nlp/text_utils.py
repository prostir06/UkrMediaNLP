"""
Shared text utilities for Ukrainian NLP pipelines.

Centralises stopword loading, token patterns, and lemmatization helpers so
vectorizers and spaCy pipelines share one configuration source.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

STOPWORDS_PATH = Path(__file__).resolve().parent.parent / "data" / "stopwords_uk.txt"

# Unicode-aware token pattern for sklearn CountVectorizer / TfidfVectorizer.
UKRAINIAN_TOKEN_PATTERN = r"(?u)\b[\wЁА-яІіЇїЄєҐґ']+\b"


def as_text_list(corpus) -> list[str]:
    """
    Convert Series, list, or iterable input to plain Python strings.

    ``NaN`` values in a pandas Series become empty strings.
    """
    try:
        if isinstance(corpus, pd.Series):
            return [str(item) for item in corpus.fillna("").tolist()]
        return [str(item) for item in corpus]
    except (TypeError, AttributeError) as exc:
        logger.warning("Cannot convert corpus to text list: %s", exc)
        return []


def normalise_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space."""
    try:
        return re.sub(r"\s+", " ", text or "").strip()
    except (TypeError, re.error) as exc:
        logger.debug("Whitespace normalisation failed: %s", exc)
        return str(text or "").strip()


@lru_cache(maxsize=1)
def load_stopwords() -> frozenset[str]:
    """
    Load Ukrainian stop words from the project file and spaCy.

    Lines starting with ``#`` in the file are treated as comments.
    """
    words: set[str] = set()

    try:
        if STOPWORDS_PATH.exists():
            for line in STOPWORDS_PATH.read_text(encoding="utf-8").splitlines():
                word = line.strip().lower()
                if word and not word.startswith("#"):
                    words.add(word)
    except OSError as exc:
        logger.warning("Cannot read stopwords file %s: %s", STOPWORDS_PATH, exc)

    try:
        from spacy.lang.uk.stop_words import STOP_WORDS

        words.update(word.lower() for word in STOP_WORDS)
    except ImportError:
        logger.debug("spaCy Ukrainian stop words unavailable")

    return frozenset(words)


def single_token_stopwords() -> list[str]:
    """Return stop words suitable for scikit-learn vectorizers (no spaces)."""
    return [word for word in load_stopwords() if " " not in word]


def lemmatize_texts(texts: list[str], nlp) -> list[str]:
    """
    Lemmatize a list of documents using a loaded spaCy pipeline.

    Tokens that are non-alphabetic or stop words are removed. Individual
    document failures fall back to the original text so one bad row does
    not abort the entire batch.
    """
    lemmatized: list[str] = []
    stopwords = load_stopwords()

    try:
        docs = nlp.pipe(texts, batch_size=32)
    except (AttributeError, TypeError) as exc:
        logger.warning("spaCy pipe unavailable, returning raw texts: %s", exc)
        return [str(text) for text in texts]

    for doc in docs:
        try:
            tokens = [
                token.lemma_.lower()
                for token in doc
                if token.is_alpha and token.lemma_.lower() not in stopwords
            ]
            lemmatized.append(" ".join(tokens))
        except (AttributeError, TypeError) as exc:
            logger.debug("Lemmatization skipped for one document: %s", exc)
            lemmatized.append("")

    return lemmatized
