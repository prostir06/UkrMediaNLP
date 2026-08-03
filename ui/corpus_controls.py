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

# Optional durable store (requires sqlalchemy; absent on minimal installs).
try:
    from corpus_store import (
        is_store_configured,
        load_corpus_from_store,
        session_scope,
        upsert_articles,
    )
except ImportError:  # pragma: no cover - exercised when deps are stripped
    logger.warning("corpus_store package unavailable; session-only corpus mode")

    def is_store_configured() -> bool:
        return False

    def load_corpus_from_store(*_args, **_kwargs):
        return None

    def session_scope(*_args, **_kwargs):
        raise RuntimeError("corpus_store unavailable")

    def upsert_articles(*_args, **_kwargs) -> int:
        return 0

CORPUS_FUNCTIONS = frozenset({"Пошук у корпусі", "Тренди тем"})


def _try_load_from_store(
    sources: list[str],
    date_from,
    date_to,
    include_missing: bool,
    category: str,
    max_rows: int,
) -> pd.DataFrame | None:
    """
    Prefer Postgres when ``DATABASE_URL`` is set and the query returns rows.

    Soft-fails (returns ``None``) on any store / blob error so the caller can
    fall back to live RSS without crashing the Streamlit page.
    """
    try:
        if not is_store_configured():
            return None
    except Exception as exc:
        logger.warning("corpus store config check failed: %s", exc)
        return None

    try:
        with session_scope() as session:
            stored = load_corpus_from_store(
                session,
                sources=list(sources),
                date_from=date_from,
                date_to=date_to,
                categories=[category] if category else None,
                include_missing_dates=include_missing,
            )
    except Exception as exc:
        logger.warning("corpus store read failed: %s", exc)
        return None

    if stored is None or getattr(stored, "empty", True):
        return None

    try:
        capped = ensure_search_blobs(
            merge_source_frames([stored], max_rows=max_rows)
        )
        capped.attrs["corpus_origin"] = "postgres"
        return capped
    except Exception as exc:
        logger.warning("corpus store post-process failed: %s", exc)
        return None


def _try_upsert_to_store(df: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Upsert live corpus rows into Postgres when configured.

    Returns:
        ``(count, None)`` on success, ``(None, error_message)`` on failure,
        ``(None, None)`` when the store is offline or *df* is empty.
    """
    try:
        if not is_store_configured() or df is None or getattr(df, "empty", True):
            return None, None
    except Exception as exc:
        return None, str(exc)

    try:
        with session_scope() as session:
            count = upsert_articles(session, df)
        return count, None
    except Exception as exc:
        logger.warning("corpus store upsert failed: %s", exc)
        return None, str(exc)


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
    *,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prefer Postgres store when configured; else live RSS, then upsert.

    When ``force_refresh`` is True, skip the store read and always scrape live
    (still upserts into the store when configured).
    """
    if not force_refresh:
        stored = _try_load_from_store(
            sources=sources,
            date_from=date_from,
            date_to=date_to,
            include_missing=include_missing,
            category=category,
            max_rows=MAX_CORPUS_ARTICLES_TOTAL,
        )
        if stored is not None:
            stored.attrs["category"] = category
            return stored, []

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
    df.attrs["corpus_origin"] = "live"
    if force_refresh:
        df.attrs["corpus_force_refresh"] = True
    upserted, upsert_error = _try_upsert_to_store(df)
    if upserted is not None:
        df.attrs["store_upserted"] = upserted
    if upsert_error:
        df.attrs["store_upsert_error"] = upsert_error
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
    force_refresh = st.sidebar.checkbox(
        "Примусово з RSS (ігнорувати store)",
        value=False,
        key="corpus_force_rss",
        help="Завантажити свіжі статті з RSS навіть якщо Postgres уже має дані.",
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
        "force_refresh": bool(force_refresh),
        "load_clicked": bool(load_clicked),
    }
