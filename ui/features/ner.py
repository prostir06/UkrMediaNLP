"""Named-entity recognition feature screen."""

import pandas as pd
import streamlit as st

from nlp.preprocessing import NER_LABELS_UA
from nlp_analysis import plot_most_common_named_entity_barchart
from ui.widgets import sample_size_slider


def render_ner(df: pd.DataFrame, titles: pd.Series) -> None:
    st.subheader("Розпізнавання сутностей (NER)")
    st.markdown(
        "Named Entity Recognition — виділення іменованих сутностей у текстах статей "
        "(модель spaCy). За замовчуванням аналізується вибірка контенту."
    )
    entity = st.selectbox(
        "Тип сутності",
        options=list(NER_LABELS_UA.keys()),
        format_func=lambda key: NER_LABELS_UA[key],
    )
    source_mode = st.radio(
        "Джерело тексту",
        ["Контент статей", "Лише заголовки"],
        horizontal=True,
    )
    if source_mode == "Лише заголовки":
        texts = titles
    else:
        content = df["content"].fillna("").astype(str)
        nonempty = content[content.str.strip().astype(bool)]
        texts = nonempty if len(nonempty) else titles
        sample_n = sample_size_slider(
            "Скільки статей для NER",
            default=min(15, len(texts)),
            max_value=max(1, len(texts)),
            key="ner_sample",
        )
        texts = texts.head(sample_n)
    plot_most_common_named_entity_barchart(texts, entity=entity)
