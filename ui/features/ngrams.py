"""N-gram, keyword, and wordcloud feature screens."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import NGRAM_DESCRIPTION, WORDCLOUD_DESCRIPTION
from nlp_analysis import (
    extract_keywords,
    get_top_n_bigram,
    get_top_n_trigram,
    get_top_n_words,
    render_wordclouds,
)


def _render_ngram_table(
    titles: pd.Series,
    extractor,
    label: str,
    chart_type: str = "scatter",
) -> None:
    st.markdown(NGRAM_DESCRIPTION)
    common = extractor(titles, 10)
    if not common:
        st.warning(f"{label} не знайдено в заголовках.")
        return

    df_ngrams = pd.DataFrame(common, columns=[label, "Кількість"])
    st.table(df_ngrams)

    if chart_type == "bar":
        fig = px.bar(df_ngrams, x=label, y="Кількість", color="Кількість", height=500)
    else:
        fig = px.scatter(
            df_ngrams,
            x=label,
            y="Кількість",
            color="Кількість",
        )
        fig.update_layout(xaxis_title=label, yaxis_title="Кількість")
    st.plotly_chart(fig, use_container_width=True)


def render_unigrams(titles: pd.Series) -> None:
    st.subheader("Уніграми")
    _render_ngram_table(titles, get_top_n_words, "Слово")


def render_bigrams(titles: pd.Series) -> None:
    st.subheader("Біграми")
    _render_ngram_table(titles, get_top_n_bigram, "Біграма", chart_type="bar")


def render_trigrams(titles: pd.Series) -> None:
    st.subheader("Триграми")
    _render_ngram_table(titles, get_top_n_trigram, "Триграма")


def render_keywords(titles: pd.Series) -> None:
    st.subheader("Ключові слова")
    st.markdown("Виділення найбільш частотних лем у заголовках новин.")
    keywords = extract_keywords(titles, top_n=15, lemmatize=True)
    if not keywords:
        st.warning("Ключових слів не знайдено.")
        return
    df_kw = pd.DataFrame(keywords, columns=["Ключове слово", "Частота"])
    st.table(df_kw)
    fig = px.bar(df_kw, x="Ключове слово", y="Частота", color="Частота", height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_wordcloud(titles: pd.Series) -> None:
    st.subheader("Хмара слів")
    st.markdown(WORDCLOUD_DESCRIPTION)
    render_wordclouds(titles)
