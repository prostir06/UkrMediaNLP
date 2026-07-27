"""Plotly charts for corpus search and trends (no Streamlit)."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)


def build_source_hit_bar(counts: pd.Series):
    try:
        if counts is None or len(counts) == 0:
            return None
        frame = counts.rename("N").reset_index()
        frame.columns = ["Медіа", "N"]
        return px.bar(frame, x="Медіа", y="N", title="Знахідки за медіа")
    except Exception as exc:
        logger.warning("build_source_hit_bar failed: %s", exc)
        return None


def build_trends_line(trends: pd.DataFrame):
    try:
        if trends is None or trends.empty:
            return None
        return px.line(
            trends,
            x="bucket",
            y="count",
            color="term",
            markers=True,
            title="Тренди тем",
            labels={"bucket": "Дата", "count": "Статей", "term": "Тема"},
        )
    except Exception as exc:
        logger.warning("build_trends_line failed: %s", exc)
        return None


def build_source_trends_line(trends: pd.DataFrame):
    try:
        if trends is None or trends.empty:
            return None
        return px.line(
            trends,
            x="bucket",
            y="count",
            color="source",
            markers=True,
            title="Порівняння медіа (одна тема)",
            labels={"bucket": "Дата", "count": "Статей", "source": "Медіа"},
        )
    except Exception as exc:
        logger.warning("build_source_trends_line failed: %s", exc)
        return None
