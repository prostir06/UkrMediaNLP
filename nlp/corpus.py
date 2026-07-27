"""Multi-source corpus helpers: dates, search, trends (no Streamlit)."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parse_published(value: object) -> pd.Timestamp:
    """Parse RSS/published field to Timestamp; return NaT on failure."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        return pd.to_datetime(text, utc=False, errors="coerce")
    except (TypeError, ValueError) as exc:
        logger.debug("parse_published failed for %r: %s", value, exc)
        return pd.NaT


def ensure_published_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``published_dt`` column."""
    out = df.copy()
    try:
        if "published" not in out.columns:
            out["published_dt"] = pd.NaT
            return out
        out["published_dt"] = out["published"].map(parse_published)
    except Exception as exc:
        logger.warning("ensure_published_dt failed: %s", exc)
        out["published_dt"] = pd.NaT
    return out


def filter_by_date(
    df: pd.DataFrame,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
    include_missing: bool = False,
) -> pd.DataFrame:
    """Filter rows by ``published_dt`` inclusive day bounds."""
    if df.empty:
        return df.copy()
    work = ensure_published_dt(df) if "published_dt" not in df.columns else df.copy()
    try:
        mask_missing = work["published_dt"].isna()
        mask_ok = ~mask_missing
        if date_from is not None:
            start = pd.Timestamp(date_from).normalize()
            mask_ok &= work["published_dt"] >= start
        if date_to is not None:
            end = pd.Timestamp(date_to).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask_ok &= work["published_dt"] <= end
        if include_missing:
            return work.loc[mask_ok | mask_missing].copy()
        return work.loc[mask_ok].copy()
    except Exception as exc:
        logger.warning("filter_by_date failed: %s", exc)
        return work.copy()


def cap_corpus(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep newest ``max_rows`` articles by ``published_dt``."""
    if max_rows <= 0 or df.empty:
        return df.iloc[0:0].copy()
    work = ensure_published_dt(df) if "published_dt" not in df.columns else df.copy()
    try:
        return work.sort_values("published_dt", ascending=False, na_position="last").head(max_rows)
    except Exception as exc:
        logger.warning("cap_corpus failed: %s", exc)
        return work.head(max_rows)
