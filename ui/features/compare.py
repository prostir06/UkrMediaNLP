"""Media comparison feature screen."""

import logging

import pandas as pd
import streamlit as st

from config import NEWS_SOURCES, source_category
from exceptions import DataLoaderError
from nlp.news_sentiment import classify_news_sentiment_batch
from nlp_analysis import get_top_n_words, preprocess
from ui.session_corpus import load_source

logger = logging.getLogger(__name__)

def render_compare_media(primary_source: str) -> None:
    st.subheader("Порівняння медіа")
    st.markdown("Порівняння топ-уніграм і новинної тональності двох джерел.")
    # Prefer peers from the same sidebar category (Новини / Спорт / …).
    primary_cat = source_category(primary_source)
    try:
        peers = [
            name
            for name, cfg in NEWS_SOURCES.items()
            if name != primary_source
            and isinstance(cfg, dict)
            and (primary_cat is None or cfg.get("category") == primary_cat)
        ]
    except (AttributeError, TypeError) as exc:
        logger.warning("Cannot build compare peers: %s", exc)
        peers = []
    if not peers:
        peers = [name for name in NEWS_SOURCES if name != primary_source]
    if not peers:
        st.warning("Немає іншого медіа для порівняння в цій категорії.")
        return

    other = st.selectbox("Друге медіа", peers)
    try:
        df_a = load_source(primary_source)
        df_b = load_source(other)
    except DataLoaderError as exc:
        st.error(str(exc))
        return

    titles_a = preprocess(df_a["title"])
    titles_b = preprocess(df_b["title"])
    top_a = dict(get_top_n_words(titles_a, 15))
    top_b = dict(get_top_n_words(titles_b, 15))
    shared = sorted(set(top_a) & set(top_b), key=lambda w: top_a[w] + top_b[w], reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{primary_source}**")
        st.table(pd.DataFrame(list(top_a.items())[:10], columns=["Слово", "N"]))
        labels_a = classify_news_sentiment_batch([str(t) for t in titles_a.head(30)])
        dist_a = pd.Series(labels_a).value_counts()
        st.markdown("Тональність (новини)")
        st.table(dist_a.rename("N"))
    with col2:
        st.markdown(f"**{other}**")
        st.table(pd.DataFrame(list(top_b.items())[:10], columns=["Слово", "N"]))
        labels_b = classify_news_sentiment_batch([str(t) for t in titles_b.head(30)])
        dist_b = pd.Series(labels_b).value_counts()
        st.markdown("Тональність (новини)")
        st.table(dist_b.rename("N"))

    if shared:
        st.markdown("**Спільні слова**")
        st.write(", ".join(shared[:20]))
    else:
        st.caption("Спільних топ-слів не знайдено.")
