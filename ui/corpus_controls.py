"""Corpus loading orchestration and Streamlit sidebar controls."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import (
    MAX_CORPUS_ARTICLES_TOTAL,
    MAX_CORPUS_SOURCES,
    sources_for_category,
)
from nlp.corpus import filter_by_date, merge_source_frames

CORPUS_FUNCTIONS = frozenset({"Пошук у корпусі", "Тренди тем"})


def build_corpus_from_sources(
    sources: list[str],
    load_articles_fn,
    max_sources: int,
    max_rows: int,
    date_from,
    date_to,
    include_missing: bool,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load each source; collect warnings; merge, filter, and cap rows."""
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    for source in list(sources)[: max(0, max_sources)]:
        try:
            frame = load_articles_fn(source, progress_callback=progress_callback)
            if frame is not None and not frame.empty:
                frames.append(frame)
        except Exception as exc:
            warnings.append(f"{source}: {exc}")

    if not frames:
        return pd.DataFrame(), warnings

    merged = merge_source_frames(frames, max_rows=sum(len(frame) for frame in frames))
    filtered = filter_by_date(
        merged,
        date_from=date_from,
        date_to=date_to,
        include_missing=include_missing,
    )
    return merge_source_frames([filtered], max_rows=max_rows), warnings


def load_corpus_into_session(
    sources: list[str],
    date_from,
    date_to,
    include_missing: bool,
    category: str,
    load_articles_fn,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build a corpus with configured limits for storage by the caller."""
    df, warnings = build_corpus_from_sources(
        sources=sources,
        load_articles_fn=load_articles_fn,
        max_sources=MAX_CORPUS_SOURCES,
        max_rows=MAX_CORPUS_ARTICLES_TOTAL,
        date_from=date_from,
        date_to=date_to,
        include_missing=include_missing,
        progress_callback=progress_callback,
    )
    df.attrs["category"] = category
    return df, warnings


def render_corpus_sidebar(category: str, show_lda_toggle: bool = False) -> dict:
    """Render corpus controls and return their current values."""
    candidates = sources_for_category(category)
    all_category = st.sidebar.checkbox(
        "Вся категорія",
        value=False,
        key="corpus_all_category",
    )
    selected = st.sidebar.multiselect(
        "Медіа корпусу",
        candidates,
        default=candidates[:1],
        disabled=all_category,
        key="corpus_sources_ms",
    )
    date_from = st.sidebar.date_input(
        "Дата від",
        value=None,
        key="corpus_date_from",
    )
    date_to = st.sidebar.date_input(
        "Дата до",
        value=None,
        key="corpus_date_to",
    )
    include_missing = st.sidebar.checkbox(
        "Включити статті без дати",
        value=True,
        key="corpus_include_missing",
    )
    load_clicked = st.sidebar.button(
        "Завантажити / оновити корпус",
        key="corpus_load_btn",
    )

    # Reserved for the trends UI in a later task.
    _ = show_lda_toggle

    return {
        "sources": candidates if all_category else list(selected),
        "all_category": bool(all_category),
        "date_from": date_from,
        "date_to": date_to,
        "include_missing": bool(include_missing),
        "load_clicked": bool(load_clicked),
    }
