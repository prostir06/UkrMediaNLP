"""
Streamlit UI renderers for NLP figures and summaries.

Keeps ``nlp/*`` free of Streamlit imports: compute modules return
``matplotlib.Figure`` / data structures; this module owns ``st.*`` calls.

Error policy: typed ``NLPAnalysisError`` → ``st.error``; unexpected errors
are logged and shown as a short Ukrainian message (no stack traces in UI).
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import streamlit as st

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)


def _safe_close(fig) -> None:
    """Close a matplotlib figure without raising."""
    try:
        if fig is not None:
            plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.debug("plt.close failed: %s", exc)


def plot_most_common_named_entity_barchart(texts, entity: str = "PER") -> None:
    """Render top named entities for one NER label."""
    from nlp.ner import build_ner_figure

    fig = None
    try:
        fig, title = build_ner_figure(texts, entity=entity)
        if fig is None:
            st.write(f"Сутності типу «{title}» не знайдено.")
            return
        st.pyplot(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except MemoryError:
        logger.exception("NER OOM")
        st.error("Недостатньо пам'яті для NER.")
    except Exception as exc:
        logger.exception("NER chart failed")
        st.error(f"Розпізнавання сутностей не вдалося: {exc}")
    finally:
        _safe_close(fig)


def plot_parts_of_speech_barchart(texts) -> None:
    """Render POS distribution for a sampled article corpus."""
    from config import MAX_POS_CONTENT_CHARS
    from nlp.pos import build_pos_figure, sample_pos_corpus

    fig = None
    try:
        if hasattr(texts, "head"):
            corpus = sample_pos_corpus(texts)
        else:
            corpus = [
                str(item)[:MAX_POS_CONTENT_CHARS]
                for item in texts
                if str(item).strip()
            ]

        fig = build_pos_figure(corpus)
        if fig is None:
            st.write("Частини мови не знайдено.")
            return
        st.pyplot(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except MemoryError:
        logger.exception("POS OOM")
        st.error("Недостатньо пам'яті для аналізу частин мови.")
    except Exception as exc:
        logger.exception("POS chart failed")
        st.error(f"Аналіз частин мови не вдався: {exc}")
    finally:
        _safe_close(fig)


def plot_sentiment_barchart(texts, method: str = "cosmus") -> None:
    """Render sentiment bar chart (``cosmus`` / ``emotions`` / ``news_rules``)."""
    from nlp.sentiment import build_sentiment_figure

    fig = None
    try:
        fig = build_sentiment_figure(texts, method=method)
        if fig is None:
            st.warning("Немає даних для аналізу тональності.")
            return
        st.pyplot(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except MemoryError:
        logger.exception("Sentiment OOM")
        st.error("Недостатньо пам'яті для аналізу тональності.")
    except Exception as exc:
        logger.exception("Sentiment chart failed")
        st.error(f"Аналіз тональності не вдався: {exc}")
    finally:
        _safe_close(fig)


def plot_emotion_distribution(texts) -> None:
    """Render multi-label emotion counts."""
    from nlp.sentiment import build_emotion_figure

    fig = None
    try:
        fig = build_emotion_figure(texts)
        if fig is None:
            st.warning("Емоції не виявлено.")
            return
        st.pyplot(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except MemoryError:
        logger.exception("Emotions OOM")
        st.error("Недостатньо пам'яті для аналізу емоцій.")
    except Exception as exc:
        logger.exception("Emotion chart failed")
        st.error(f"Аналіз емоцій не вдався: {exc}")
    finally:
        _safe_close(fig)


def display_topic_modeling(content, number_topics: int = 8, number_words: int = 6) -> None:
    """Show LDA topics as a bullet list."""
    from nlp.topics import run_topic_modeling

    try:
        topics = run_topic_modeling(content, number_topics, number_words)
    except NLPAnalysisError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        logger.exception("Topic modeling UI failed")
        st.error(f"Тематичне моделювання не вдалося: {exc}")
        return

    if not topics:
        st.warning(
            "Недостатньо тексту для тематичного моделювання. "
            "Потрібно щонайменше 3 статті з повним текстом."
        )
        return

    st.markdown("**Виявлені теми (LDA):**")
    for topic in topics:
        st.write(topic)


def run_text_summarization(df, sentence_count: int = 3, max_articles: int = 10) -> None:
    """Summarize article bodies with a single shared spaCy instance."""
    from nlp.model_registry import resolve_spacy_nlp
    from nlp.summarization import summarize_articles

    try:
        nlp = resolve_spacy_nlp()
        results = summarize_articles(
            df,
            sentence_count=sentence_count,
            max_articles=max_articles,
            nlp=nlp,
        )
    except NLPAnalysisError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        logger.exception("Summarization UI failed")
        st.error(f"Сумаризація не вдалася: {exc}")
        return

    if not results:
        st.warning("Жодну статтю не вдалося сумаризувати.")
        return

    for title, sentences in results:
        st.markdown(f"**{title}**")
        for sentence in sentences:
            st.write(sentence)
        st.divider()


def render_wordclouds(titles) -> None:
    """Show one word cloud by default (extra styles appear as tabs)."""
    from nlp.wordcloud_render import build_wordcloud_images

    try:
        images = build_wordcloud_images(titles, styles=None)
        if not images:
            st.warning("Немає тексту для побудови хмари слів.")
            return
        tab_labels = [f"Стиль {i + 1}" for i in range(len(images))]
        tabs = st.tabs(tab_labels) if len(images) > 1 else [st.container()]
        for tab, image in zip(tabs, images):
            with tab:
                st.image(image, width=700)
    except MemoryError:
        logger.exception("Wordcloud OOM")
        st.error("Недостатньо пам'яті для хмари слів.")
    except Exception as exc:
        logger.exception("Word cloud generation failed")
        st.error(f"Помилка побудови хмари слів: {exc}")
