"""
Streamlit UI renderers for NLP figures and summaries.

Keeps ``nlp/*`` free of Streamlit imports.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import streamlit as st

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)


def plot_most_common_named_entity_barchart(texts, entity: str = "PER") -> None:
    from nlp.ner import build_ner_figure

    try:
        fig, title = build_ner_figure(texts, entity=entity)
        if fig is None:
            st.write(f"Сутності типу «{title}» не знайдено.")
            return
        st.pyplot(fig)
        plt.close(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Розпізнавання сутностей не вдалося: {exc}")


def plot_parts_of_speech_barchart(texts) -> None:
    from config import MAX_POS_CONTENT_CHARS
    from nlp.pos import build_pos_figure, sample_pos_corpus

    try:
        if hasattr(texts, "head"):
            corpus = sample_pos_corpus(texts)
        else:
            corpus = [str(t)[:MAX_POS_CONTENT_CHARS] for t in texts if str(t).strip()]

        fig = build_pos_figure(corpus)
        if fig is None:
            st.write("Частини мови не знайдено.")
            return
        st.pyplot(fig)
        plt.close(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Аналіз частин мови не вдався: {exc}")


def plot_sentiment_barchart(texts, method: str = "cosmus") -> None:
    from nlp.sentiment import build_sentiment_figure

    try:
        fig = build_sentiment_figure(texts, method=method)
        if fig is None:
            st.warning("Немає даних для аналізу тональності.")
            return
        st.pyplot(fig)
        plt.close(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Sentiment chart failed")
        st.error(f"Аналіз тональності не вдався: {exc}")


def plot_emotion_distribution(texts) -> None:
    from nlp.sentiment import build_emotion_figure

    try:
        fig = build_emotion_figure(texts)
        if fig is None:
            st.warning("Емоції не виявлено.")
            return
        st.pyplot(fig)
        plt.close(fig)
    except NLPAnalysisError as exc:
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Emotion chart failed")
        st.error(f"Аналіз емоцій не вдався: {exc}")


def display_topic_modeling(content, number_topics: int = 8, number_words: int = 6) -> None:
    from nlp.topics import run_topic_modeling

    try:
        topics = run_topic_modeling(content, number_topics, number_words)
    except NLPAnalysisError as exc:
        st.error(str(exc))
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

    if not results:
        st.warning("Жодну статтю не вдалося сумаризувати.")
        return

    for title, sentences in results:
        st.markdown(f"**{title}**")
        for sentence in sentences:
            st.write(sentence)
        st.divider()


def render_wordclouds(titles) -> None:
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
    except Exception as exc:
        logger.exception("Word cloud generation failed")
        st.error(f"Помилка побудови хмари слів: {exc}")
