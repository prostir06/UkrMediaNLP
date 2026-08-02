"""Text summarization feature screen."""

import pandas as pd
import streamlit as st

from config import MAX_SUMMARY_ARTICLES
from nlp_analysis import run_text_summarization
from ui.widgets import sample_size_slider


def render_summarization(df: pd.DataFrame) -> None:
    st.subheader("Сумаризація тексту")
    st.markdown(
        "Екстрактивна сумаризація на основі LexRank: "
        "найважливіші речення з повного тексту статті (spaCy + TF-IDF)."
    )
    sample_n = sample_size_slider(
        "Скільки статей сумаризувати",
        default=min(MAX_SUMMARY_ARTICLES, max(1, len(df))),
        max_value=max(1, len(df)),
        key="summary_sample",
    )
    run_text_summarization(df, sentence_count=3, max_articles=sample_n)
