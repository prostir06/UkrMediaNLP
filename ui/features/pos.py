"""Parts-of-speech feature screen."""

import pandas as pd
import streamlit as st

from config import MAX_POS_ARTICLES
from nlp_analysis import plot_parts_of_speech_barchart
from ui.widgets import sample_size_slider

POS_DESCRIPTIONS = [
    ("NOUN", "іменник", "Київ, уряд, закон"),
    ("VERB", "дієслово", "заявив, ухвалив, працює"),
    ("ADJ", "прикметник", "новий, важливий, економічний"),
    ("ADV", "прислівник", "швидко, сьогодні, дуже"),
    ("PROPN", "власна назва", "Україна, Зеленський, NATO"),
    ("ADP", "прийменник", "у, на, для, про"),
    ("CCONJ", "союз", "і, але, або"),
]


def render_pos(content: pd.Series) -> None:
    st.subheader("Частини мови")
    st.markdown("Розподіл частин мови у текстах статей (spaCy).")
    for tag, name_ua, example in POS_DESCRIPTIONS:
        st.markdown(f"- **{name_ua} ({tag})** — {example}")
    sample_n = sample_size_slider(
        "Скільки статей для POS",
        default=min(MAX_POS_ARTICLES, max(1, len(content))),
        max_value=max(1, len(content)),
        key="pos_sample",
    )
    plot_parts_of_speech_barchart(content.head(sample_n))
