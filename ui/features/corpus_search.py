"""Corpus search feature screen (Streamlit).

HTML5 / CSS3 / StandardJS do not apply — Python UI module (PEP 8).
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from exceptions import NLPAnalysisError
from nlp.corpus import search_corpus, search_corpus_semantic
from nlp.embeddings import embeddings_enabled
from ui.corpus_charts import build_source_hit_bar

logger = logging.getLogger(__name__)

# Default semantic ranking thresholds (tuned for hash encoder similarities).
_SEMANTIC_MIN_SCORE = 0.05
_SEMANTIC_TOP_K = 100


def _load_corpus_from_session() -> pd.DataFrame | None:
    """Read corpus DataFrame from Streamlit session; None when missing/empty."""
    try:
        corpus_df = st.session_state.get("corpus_df")
    except Exception as exc:  # pragma: no cover - Streamlit session edge cases
        logger.warning("Cannot read corpus_df from session: %s", exc)
        return None
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        return None
    return corpus_df


def _run_search(
    corpus_df: pd.DataFrame,
    query: str,
    *,
    semantic_mode: bool,
    field_mode: str,
    whole_word: bool,
    use_lemmas: bool,
) -> pd.DataFrame:
    """
    Execute keyword or semantic search.

    Raises:
        NLPAnalysisError: Propagated from ``nlp.corpus`` search helpers.
    """
    if semantic_mode:
        return search_corpus_semantic(
            corpus_df,
            query,
            min_score=_SEMANTIC_MIN_SCORE,
            top_k=_SEMANTIC_TOP_K,
        )

    field_map = {
        "Заголовках і текстах": ("title", "content"),
        "Лише заголовках": ("title",),
        "Лише текстах": ("content",),
    }
    return search_corpus(
        corpus_df,
        query=query,
        fields=field_map.get(field_mode, ("title", "content")),
        whole_word=whole_word,
        use_lemmas=use_lemmas,
    )


def _report_search_error(exc: BaseException) -> None:
    """Show a user-facing Streamlit error for search failures."""
    if isinstance(exc, NLPAnalysisError):
        logger.warning("Corpus search failed: %s", exc)
        st.error(str(exc))
        return
    logger.exception("Corpus search failed")
    st.error(f"Пошук у корпусі не вдався: {exc}")


def _render_results_table(results: pd.DataFrame) -> list[str]:
    """Render metrics, optional chart, and dataframe; return visible columns."""
    source_count = results["source"].nunique() if "source" in results.columns else 0
    metric_hits, metric_sources = st.columns(2)
    metric_hits.metric("Знайдено статей", len(results))
    metric_sources.metric("Медіа зі знахідками", int(source_count))

    if results.empty:
        st.warning("За вашим запитом нічого не знайдено.")
        return []

    if "source" in results.columns:
        try:
            figure = build_source_hit_bar(results["source"].fillna("").value_counts())
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
        except Exception as exc:
            logger.warning("Corpus search chart failed: %s", exc)

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
    return visible_columns


def _render_csv_download(results: pd.DataFrame, visible_columns: list[str]) -> None:
    """Offer CSV export; soft-fail when encoding fails."""
    if not visible_columns:
        return
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


def render_corpus_search() -> None:
    """Render search controls and results for the loaded multi-source corpus."""
    corpus_df = _load_corpus_from_session()
    if corpus_df is None:
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

    if not str(query).strip():
        st.caption("Введіть слово або фразу для пошуку.")
        return

    try:
        results = _run_search(
            corpus_df,
            str(query),
            semantic_mode=semantic_mode,
            field_mode=field_mode,
            whole_word=whole_word,
            use_lemmas=use_lemmas,
        )
    except NLPAnalysisError as exc:
        _report_search_error(exc)
        return
    except Exception as exc:
        _report_search_error(exc)
        return

    visible = _render_results_table(results)
    _render_csv_download(results, visible)
