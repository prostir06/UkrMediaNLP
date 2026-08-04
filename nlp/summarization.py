"""
Extractive text summarization for Ukrainian articles.
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from exceptions import NLPAnalysisError
from nlp.model_registry import resolve_spacy_nlp
from nlp.preprocessing import normalise_whitespace
from nlp.text_utils import UKRAINIAN_TOKEN_PATTERN, single_token_stopwords

logger = logging.getLogger(__name__)

LEXRANK_ITERATIONS = 20


def split_sentences(text: str, min_length: int = 25, nlp=None) -> list[str]:
    text = normalise_whitespace(text)
    if not text:
        return []

    try:
        if nlp is None:
            nlp = resolve_spacy_nlp()
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return [sentence for sentence in sentences if len(sentence) >= min_length]
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Sentence splitting failed: %s", exc)
        return []


def lexrank_summarize(text: str, sentence_count: int = 3, nlp=None) -> list[str]:
    sentences = split_sentences(text, nlp=nlp)
    if not sentences:
        return []
    if len(sentences) <= sentence_count:
        return sentences

    try:
        vectorizer = TfidfVectorizer(
            token_pattern=UKRAINIAN_TOKEN_PATTERN,
            stop_words=single_token_stopwords(),
        )
        matrix = vectorizer.fit_transform(sentences)
        similarity = cosine_similarity(matrix)

        scores = np.ones(len(sentences))
        for _ in range(LEXRANK_ITERATIONS):
            updated = similarity @ scores
            norm = np.linalg.norm(updated)
            scores = updated / norm if norm else updated

        ranked_indices = np.argsort(scores)[::-1][:sentence_count]
        ranked_indices = sorted(ranked_indices)
        return [sentences[i] for i in ranked_indices]
    except ValueError as exc:
        logger.warning("LexRank vectorisation failed: %s", exc)
        return sentences[:sentence_count]


def summarize_articles(
    df,
    sentence_count: int = 3,
    max_articles: int = 10,
    nlp=None,
) -> list[tuple[str, list[str]]]:
    """
    Summarize up to ``max_articles`` rows.

    Returns list of (title, summary_sentences) pairs. Uses one spaCy instance
    for the whole batch.
    """
    if nlp is None:
        nlp = resolve_spacy_nlp()

    results: list[tuple[str, list[str]]] = []
    articles = df.head(max_articles)
    contents = (
        articles["content"].fillna("").astype(str).tolist()
        if "content" in articles.columns
        else [""] * len(articles)
    )
    titles = (
        articles["title"].fillna("").astype(str).tolist()
        if "title" in articles.columns
        else ["Без заголовка"] * len(articles)
    )

    for title, content in zip(titles, contents, strict=False):
        body = content.strip()
        if not body:
            continue
        try:
            summary = lexrank_summarize(body, sentence_count=sentence_count, nlp=nlp)
            if not summary:
                continue
            results.append((title.strip() or "Без заголовка", summary))
        except NLPAnalysisError:
            raise
        except Exception as exc:
            logger.warning("Summarization failed: %s", exc)

    return results
