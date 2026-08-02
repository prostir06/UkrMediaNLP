"""Session helpers for loading articles and multi-source corpora."""

import logging

import pandas as pd
import streamlit as st

from cache import load_articles
from exceptions import DataLoaderError

logger = logging.getLogger(__name__)

def load_source(source_name: str) -> pd.DataFrame:
    """
    Load articles with a live scrape progress bar (N/M).

    Progress is only meaningful on cache miss; SQLite hits return immediately.
    The bar is always cleared in ``finally`` so a failed load does not leave
    a stuck widget on the page.
    """
    progress = None
    try:
        progress = st.progress(0, text="Завантаження статей...")

        def on_progress(done: int, total: int) -> None:
            """Update the Streamlit progress bar; ignore callback UI errors."""
            if total <= 0 or progress is None:
                return
            try:
                progress.progress(
                    min(done / total, 1.0),
                    text=f"Скрейпінг статей: {done}/{total}",
                )
            except Exception as exc:
                logger.debug("Progress UI update skipped: %s", exc)

        return load_articles(source_name, progress_callback=on_progress)
    except DataLoaderError:
        raise
    except Exception as exc:
        logger.exception("Failed to load source %s", source_name)
        raise DataLoaderError(
            f"Не вдалося завантажити статті: {exc}",
            source_name=source_name,
        ) from exc
    finally:
        if progress is not None:
            try:
                progress.empty()
            except Exception as exc:
                logger.debug("Progress cleanup failed: %s", exc)


def commit_corpus_load(
    corpus_df: pd.DataFrame,
    warnings: list[str],
    sources: list[str],
    category: str,
) -> bool:
    """Store a corpus load unless every requested source failed."""
    previous = st.session_state.get("corpus_df")
    has_previous = isinstance(previous, pd.DataFrame) and not previous.empty
    total_failure = bool(sources) and corpus_df.empty and len(warnings) >= len(sources)

    for warning in warnings:
        st.warning(f"Не вдалося завантажити джерело: {warning}")

    if total_failure and has_previous:
        st.error("Жодне джерело не завантажено. Попередній корпус збережено.")
        return False

    st.session_state["corpus_df"] = corpus_df
    st.session_state["corpus_sources"] = list(sources)
    st.session_state["corpus_category"] = category
    st.session_state["corpus_loaded_at"] = pd.Timestamp.now()

    if total_failure:
        st.error("Жодне джерело не завантажено.")
    elif corpus_df.empty:
        st.warning("Корпус порожній: статті за заданими умовами не знайдено.")
    else:
        st.success(f"Корпус завантажено: {len(corpus_df)} статей.")

    origin = corpus_df.attrs.get("corpus_origin", "live")
    if origin == "postgres" and not corpus_df.empty:
        st.caption(f"Джерело: Postgres ({len(corpus_df)} статей)")
    elif not corpus_df.empty:
        st.caption(f"Джерело: Live RSS ({len(corpus_df)} статей)")
        upsert_error = corpus_df.attrs.get("store_upsert_error")
        if upsert_error:
            st.warning(f"Не вдалося зберегти корпус у Postgres: {upsert_error}")
        elif corpus_df.attrs.get("store_upserted") is not None:
            st.caption(
                f"Збережено в store: {int(corpus_df.attrs['store_upserted'])} записів"
            )

    stats_list = corpus_df.attrs.get("scrape_stats_by_source") or []
    if isinstance(stats_list, list) and stats_list:
        parts = []
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            src = stats.get("source", "?")
            ok = int(stats.get("ok", 0))
            total = int(stats.get("total", 0))
            elapsed = stats.get("elapsed_ms")
            rate = f"{ok / total:.0%}" if total else "—"
            timing = f", {elapsed} мс" if elapsed is not None else ""
            parts.append(f"{src}: {ok}/{total} ({rate}){timing}")
        if parts:
            st.caption("Успішність скрейпінгу — " + "; ".join(parts))
    return True


def invalidate_stale_corpus(
    category: str,
    current_sources: list[str],
    all_category: bool,
) -> bool:
    """Soft-clear a loaded corpus that no longer matches sidebar controls."""
    corpus_df = st.session_state.get("corpus_df")
    if not isinstance(corpus_df, pd.DataFrame) or corpus_df.empty:
        return False

    stored_category = st.session_state.get("corpus_category")
    stored_sources = st.session_state.get("corpus_sources", [])
    category_changed = stored_category != category
    sources_changed = not all_category and sorted(stored_sources) != sorted(current_sources)
    if not category_changed and not sources_changed:
        return False

    st.session_state["corpus_df"] = corpus_df.iloc[0:0].copy()
    st.warning("Налаштування корпусу змінилися; попередній корпус більше не актуальний.")
    st.info("Завантажте корпус повторно для поточної категорії та вибраних медіа.")
    return True
