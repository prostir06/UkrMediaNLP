"""Shared Streamlit widgets for NLP feature screens."""

import logging

import streamlit as st

logger = logging.getLogger(__name__)


def sample_size_slider(label: str, default: int, max_value: int, key: str) -> int:
    """
    Clamp and render a Streamlit sample-size slider.

    Guards against empty corpora (``max_value < 1``) and invalid defaults so
    Streamlit never receives an out-of-range ``value``.
    """
    try:
        upper = max(1, int(max_value))
        value = min(max(1, int(default)), upper)
        return int(st.slider(label, min_value=1, max_value=upper, value=value, key=key))
    except (TypeError, ValueError) as exc:
        logger.warning("Sample size slider fallback (%s): %s", key, exc)
        return max(1, int(default) if str(default).isdigit() else 1)
    except Exception as exc:
        logger.exception("Unexpected slider failure for key=%s", key)
        raise RuntimeError(f"Не вдалося показати слайдер: {exc}") from exc
