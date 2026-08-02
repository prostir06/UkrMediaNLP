"""Corpus topic-trends feature screen."""

import logging

import pandas as pd
import streamlit as st

from config import MAX_TREND_TERMS, get_cloud_light
from nlp.corpus import (
    aggregate_trends,
    aggregate_trends_by_source,
    parse_manual_terms,
    suggest_lda_labels,
    suggest_terms,
)
from ui.corpus_charts import build_source_trends_line, build_trends_line

logger = logging.getLogger(__name__)

def render_topic_trends() -> None:
    """Render topic trends for the loaded multi-source corpus."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        st.info("Спочатку завантажте корпус у бічній панелі.")
        return

    st.subheader("Тренди тем")
    automatic_terms = suggest_terms(corpus_df, 15)
    lda_terms: list[str] = []
    if not get_cloud_light():
        with st.expander("Поглиблено (LDA)"):
            try:
                lda_terms = suggest_lda_labels(corpus_df)
            except Exception as exc:
                logger.warning("LDA topic suggestions failed: %s", exc)
                lda_terms = []
            if not lda_terms:
                st.warning("Не вдалося запропонувати теми за допомогою LDA.")
            else:
                st.caption("LDA: " + ", ".join(lda_terms))

    manual_text = st.text_area(
        "Власні теми (по одній у рядку)",
        key="topic_trends_manual_terms",
    )
    manual_terms = parse_manual_terms(manual_text)

    candidates: list[str] = []
    seen: set[str] = set()
    for term in [*automatic_terms, *lda_terms, *manual_terms]:
        normalized = str(term).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            candidates.append(normalized)

    selected_terms = st.multiselect(
        "Теми для порівняння",
        candidates,
        max_selections=MAX_TREND_TERMS,
        key="topic_trends_terms",
    )
    frequency_label = st.radio(
        "Групування",
        ("День", "Тиждень"),
        horizontal=True,
        key="topic_trends_frequency",
    )
    frequency = "D" if frequency_label == "День" else "W-MON"

    if not selected_terms:
        st.caption("Оберіть хоча б одну тему.")
        return

    trend_figure = build_trends_line(
        aggregate_trends(corpus_df, list(selected_terms), freq=frequency)
    )
    if trend_figure is None:
        st.warning("Для обраних тем не знайдено даних із датами.")
    else:
        st.plotly_chart(trend_figure, use_container_width=True)

    comparison_term = st.selectbox(
        "Тема для порівняння медіа",
        selected_terms,
        key="topic_trends_source_term",
    )
    source_figure = build_source_trends_line(
        aggregate_trends_by_source(corpus_df, comparison_term, freq=frequency)
    )
    if source_figure is None:
        st.warning("Немає даних для порівняння медіа за цією темою.")
    else:
        st.plotly_chart(source_figure, use_container_width=True)
