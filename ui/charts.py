"""
Matplotlib chart builders for the Streamlit UI.

Pure compute (classification, NER extraction) stays in ``nlp/*``;
this module owns figure construction so NLP packages stay plot-light.

Error policy
------------
* Typed ``NLPAnalysisError`` is re-raised for the UI layer.
* Unexpected failures are logged and wrapped as ``RuntimeError`` (or
  return ``None`` for empty / soft-fail charts).
"""

from __future__ import annotations

import logging
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from exceptions import NLPAnalysisError
from nlp.preprocessing import NER_LABELS_UA, POS_LABELS_UA
from nlp.sentiment import SENTIMENT_COLORS

logger = logging.getLogger(__name__)


def build_ner_figure(texts, entity: str = "PER"):
    """
    Build a horizontal bar chart of the most common named entities.

    Returns:
        ``(figure, title)`` or ``(None, title)`` when no entities match.
    """
    from nlp.ner import extract_entities_batch

    try:
        text_list = [str(t or "") for t in texts]
        entity_lists = extract_entities_batch(text_list, entity)
        flat_entities = [item for sublist in entity_lists for item in sublist]

        counter = Counter(flat_entities)
        title = NER_LABELS_UA.get(entity, entity)
        if not counter:
            return None, title

        labels, counts = map(list, zip(*counter.most_common(10)))
        fig, ax = plt.subplots()
        sns.barplot(x=counts, y=labels, ax=ax).set_title(title)
        return fig, title
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.exception("NER chart failed")
        raise RuntimeError("NER chart failed") from exc


def build_pos_figure(texts: list[str]):
    """
    Build a bar chart of part-of-speech tag frequencies.

    Returns:
        Matplotlib figure, or ``None`` when the corpus has no POS tags.
    """
    from nlp.pos import extract_pos_batch

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


def build_sentiment_figure(texts, method: str = "cosmus"):
    """
    Build a bar chart of sentiment label counts.

    Args:
        texts: Headlines or short snippets.
        method: ``cosmus``, ``emotions``, or ``news_rules``.

    Returns:
        Matplotlib figure, or ``None`` when there are no labels to plot.

    Raises:
        ValueError: Unknown ``method``.
        NLPAnalysisError: Propagated from transformer-backed classifiers.
        RuntimeError: Unexpected plotting / classification failure.
    """
    from nlp.sentiment import classify_emotions_batch, classify_sentiment_batch

    try:
        if method == "cosmus":
            labels_list = classify_sentiment_batch(list(texts))
        elif method == "emotions":
            batch = classify_emotions_batch([str(t) for t in texts])
            labels_list = [dominant for _, dominant in batch]
        elif method == "news_rules":
            from nlp.news_sentiment import classify_news_sentiment_batch

            labels_list = classify_news_sentiment_batch([str(t) for t in texts])
        else:
            raise ValueError(f"Unknown sentiment method: {method}")

        import pandas as pd

        counts = pd.Series(labels_list).value_counts()
        if counts.empty:
            return None

        colors = [SENTIMENT_COLORS.get(label, "#3498db") for label in counts.index]
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
        ax.set_ylabel("Кількість")
        ax.tick_params(axis="x", rotation=25)
        return fig
    except (NLPAnalysisError, ValueError):
        raise
    except Exception as exc:
        logger.exception("Sentiment chart failed (method=%s)", method)
        raise RuntimeError("Sentiment chart failed") from exc


def build_emotion_figure(texts):
    """
    Build a horizontal bar chart of multi-label emotion counts.

    Soft-fails to ``None`` on unexpected inference errors (UI shows a warning).
    """
    from nlp.sentiment import classify_emotions_batch

    counter: Counter[str] = Counter()
    try:
        batch = classify_emotions_batch([str(t) for t in texts])
        for detected, _ in batch:
            counter.update(detected)
    except NLPAnalysisError:
        raise
    except Exception as exc:
        logger.warning("Emotion batch failed: %s", exc)
        return None

    if not counter:
        return None

    try:
        labels, counts = zip(*counter.most_common())
        fig, ax = plt.subplots()
        ax.barh(labels, counts, color="#8e44ad", edgecolor="white")
        ax.set_xlabel("Кількість згадувань")
        return fig
    except Exception as exc:
        logger.exception("Emotion chart plot failed")
        raise RuntimeError("Emotion chart failed") from exc
