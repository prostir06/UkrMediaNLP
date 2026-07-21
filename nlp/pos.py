"""
Part-of-speech analysis with spaCy for Ukrainian texts.
"""

import logging
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from config import MAX_POS_ARTICLES, MAX_POS_CONTENT_CHARS
from exceptions import NLPAnalysisError
from nlp.model_registry import resolve_spacy_nlp
from nlp.preprocessing import POS_LABELS_UA

logger = logging.getLogger(__name__)


def extract_pos_batch(texts: list[str], nlp=None) -> list[list[str]]:
    """Return POS tags for each document using batched spaCy processing."""
    if nlp is None:
        nlp = resolve_spacy_nlp()

    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=32):
        tags = [
            token.pos_
            for token in doc
            if not token.is_space and not token.is_punct
        ]
        results.append(tags)
    return results


def sample_pos_corpus(content_series, max_articles: int = MAX_POS_ARTICLES) -> list[str]:
    """Limit POS analysis to a sample of truncated article bodies."""
    texts = []
    for text in content_series.head(max_articles):
        value = str(text or "").strip()
        if not value:
            continue
        texts.append(value[:MAX_POS_CONTENT_CHARS])
    return texts


def build_pos_figure(texts: list[str]):
    try:
        tag_lists = extract_pos_batch(texts)
        flat_tags = [tag for sublist in tag_lists for tag in sublist]

        if not flat_tags:
            return None

        counter = Counter(flat_tags)
        labels, counts = map(list, zip(*counter.most_common(7)))
        labels_ua = [POS_LABELS_UA.get(label, label) for label in labels]

        fig, ax = plt.subplots()
        sns.barplot(x=counts, y=labels_ua, ax=ax).set_title("Частини мови")
        return fig
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("POS chart failed")
        raise RuntimeError("POS chart failed") from exc
