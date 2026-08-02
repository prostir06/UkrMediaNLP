"""Corpus loading orchestration and Streamlit sidebar controls."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from config import (
    CORPUS_LOAD_WORKERS,
    MAX_CORPUS_ARTICLES_TOTAL,
    MAX_CORPUS_SOURCES,
    sources_for_category,
)
from nlp.corpus import ensure_search_blobs, filter_by_date, merge_source_frames
from observability import log_step

logger = logging.getLogger(__name__)

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
    """Load each source (bounded concurrency); merge, filter, and cap rows."""
    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    scrape_stats: list[dict] = []
    selected = list(sources)[: max(0, max_sources)]

    def _load_one(source: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            # progress_callback is not used on worker threads (Streamlit is not
            # thread-safe); rate limiting still applies inside scrape helpers.
            frame = load_articles_fn(source, progress_callback=None)
            return source, frame, None
        except Exception as exc:
            return source, None, str(exc)

    with log_step(logger, step="corpus_load", source=",".join(selected[:3])):
        workers = max(1, min(int(CORPUS_LOAD_WORKERS), len(selected) or 1))
        if len(selected) <= 1 or workers == 1:
            ordered = [_load_one(source) for source in selected]
        else:
            ordered = []
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_load_one, source): source for source in selected
                }
                results: dict[str, tuple[str, pd.DataFrame | None, str | None]] = {}
                for future in as_completed(futures):
                    source = futures[future]
                    try:
                        results[source] = future.result()
                    except Exception as exc:
                        results[source] = (source, None, str(exc))
                ordered = [results[source] for source in selected]

        for source, frame, error in ordered:
            if error:
                warnings.append(f"{source}: {error}")
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
                stats = getattr(frame, "attrs", {}).get("scrape_stats")
                if isinstance(stats, dict):
                    scrape_stats.append(stats)

        if not frames:
            empty = pd.DataFrame()
            empty.attrs["scrape_stats_by_source"] = scrape_stats
            return empty, warnings

        merged = merge_source_frames(frames, max_rows=sum(len(frame) for frame in frames))
        filtered = filter_by_date(
            merged,
            date_from=date_from,
            date_to=date_to,
            include_missing=include_missing,
        )
        result = ensure_search_blobs(
            merge_source_frames([filtered], max_rows=max_rows)
        )
        result.attrs["scrape_stats_by_source"] = scrape_stats
        return result, warnings


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


def render_corpus_sidebar(category: str) -> dict:
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

    return {
        "sources": candidates if all_category else list(selected),
        "all_category": bool(all_category),
        "date_from": date_from,
        "date_to": date_to,
        "include_missing": bool(include_missing),
        "load_clicked": bool(load_clicked),
    }
