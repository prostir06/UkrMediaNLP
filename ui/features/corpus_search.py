"""Corpus search feature screen."""

import logging

import pandas as pd
import streamlit as st

from exceptions import NLPAnalysisError
from nlp.corpus import search_corpus, search_corpus_semantic
from nlp.embeddings import embeddings_enabled
from ui.corpus_charts import build_source_hit_bar

logger = logging.getLogger(__name__)


def render_corpus_search() -> None:
    """Render search controls and results for the loaded multi-source corpus."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        st.info("Спочатку завантажте корпус у бічній панелі.")
        return

    st.subheader("Пошук у корпусі")
    query = st.text_input("Пошуковий запит", key="corpus_search_query")
    search_mode = st.radio(
        "Режим пошуку",
        ("Ключові слова", "Семантичний"),
        horizontal=True,
        key="corpus_search_mode",
    )
    semantic_mode = search_mode == "Семантичний"

    if semantic_mode and not embeddings_enabled():
        st.info(
            "Семантичний пошук вимкнено. Задайте ALLOW_EMBEDDINGS=1 "
            "(локально або в Docker) для цього режиму."
        )
        return

    field_mode = "Заголовках і текстах"
    whole_word = False
    use_lemmas = False

    if not semantic_mode:
        field_mode = st.radio(
            "Шукати в",
            ("Заголовках і текстах", "Лише заголовках", "Лише текстах"),
            horizontal=True,
            key="corpus_search_fields",
        )
        field_map = {
            "Заголовках і текстах": ("title", "content"),
            "Лише заголовках": ("title",),
            "Лише текстах": ("content",),
        }
        col_word, col_lemma = st.columns(2)
        with col_word:
            whole_word = st.checkbox(
                "Лише ціле слово",
                value=False,
                key="corpus_search_whole_word",
            )
        with col_lemma:
            use_lemmas = st.checkbox(
                "Враховувати леми",
                value=False,
                key="corpus_search_lemmas",
            )
    else:
        st.caption("Семантичний режим: ранжування за схожістю embeddings (hash encoder).")
        field_map = {
            "Заголовках і текстах": ("title", "content"),
        }

    if not str(query).strip():
        st.caption("Введіть слово або фразу для пошуку.")
        return

    try:
        if semantic_mode:
            results = search_corpus_semantic(
                corpus_df,
                str(query),
                min_score=0.05,
                top_k=100,
            )
        else:
            results = search_corpus(
                corpus_df,
                query=str(query),
                fields=field_map.get(field_mode, ("title", "content")),
                whole_word=bool(whole_word),
                use_lemmas=bool(use_lemmas),
            )
    except NLPAnalysisError as exc:
        logger.warning("Corpus search failed: %s", exc)
        st.error(str(exc))
        return
    except Exception as exc:
        logger.exception("Corpus search failed")
        st.error(f"Пошук у корпусі не вдався: {exc}")
        return

    source_count = results["source"].nunique() if "source" in results.columns else 0
    metric_hits, metric_sources = st.columns(2)
    metric_hits.metric("Знайдено статей", len(results))
    metric_sources.metric("Медіа зі знахідками", int(source_count))

    if results.empty:
        st.warning("За вашим запитом нічого не знайдено.")
        return

    if "source" in results.columns:
        figure = build_source_hit_bar(results["source"].fillna("").value_counts())
        if figure is not None:
            st.plotly_chart(figure, use_container_width=True)

    date_column = "published_dt" if "published_dt" in results.columns else "published"
    visible_columns = [
        column
        for column in ("title", "link", date_column, "source", "snippet", "relevance")
        if column in results.columns
    ]
    st.dataframe(
        results[visible_columns],
        use_container_width=True,
        hide_index=True,
    )
    try:
        csv_bytes = results[visible_columns].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Завантажити CSV",
            data=csv_bytes,
            file_name="corpus_search_hits.csv",
            mime="text/csv",
            key="corpus_search_csv",
        )
    except Exception as exc:
        logger.warning("Corpus search CSV export failed: %s", exc)
        st.caption("CSV-експорт тимчасово недоступний.")
