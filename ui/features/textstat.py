"""Text statistics feature screen."""

import pandas as pd
import streamlit as st

from nlp.textstat_ua import aggregate_corpus_metrics, get_textstat_metrics


def render_text_stat(df: pd.DataFrame) -> None:
    st.subheader("Статистика тексту")
    st.markdown(
        "Базові метрики читабельності та структури тексту для українських статей. "
        "Індекси Flesch/SMOG недоступні — застосовано власні метрики."
    )

    content = df["content"].fillna("").astype(str)
    sample_text = next((text for text in content if text.strip()), "")
    if not sample_text:
        st.warning("Немає тексту статей для аналізу.")
        return

    st.markdown("**Перша стаття з повним текстом**")
    for name, value in get_textstat_metrics(sample_text):
        st.write(f"**{name}:** {value}")

    st.divider()
    st.markdown("**Увесь завантажений корпус**")
    for name, value in aggregate_corpus_metrics(content):
        st.write(f"**{name}:** {value}")
