"""Process-wide runtime defaults for Hugging Face / torch stability."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _transformers_available() -> bool:
    """Return True when the transformers package is importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def _truthy_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def get_cloud_light() -> bool:
    """
    Resolve light-UI mode (hide RoBERTa / emotions).

    Light (stable) when:
    * ``LIGHT_CLOUD`` env/secret is set, or
    * ``transformers`` is missing, or
    * ``ALLOW_HEAVY_NLP`` is not enabled (default).

    Opt in with ``ALLOW_HEAVY_NLP=1`` (env or Streamlit secrets). Streamlit is
    imported lazily here so ``config`` stays free of UI imports.
    """
    if _truthy_flag(os.environ.get("LIGHT_CLOUD")):
        return True
    if not _transformers_available():
        return True

    allow_heavy = _truthy_flag(os.environ.get("ALLOW_HEAVY_NLP"))
    try:
        import streamlit as st

        if _truthy_flag(st.secrets.get("LIGHT_CLOUD", "")):
            return True
        allow_heavy = allow_heavy or _truthy_flag(
            st.secrets.get("ALLOW_HEAVY_NLP", ""),
        )
    except Exception:
        pass

    return not allow_heavy


def apply_runtime_env() -> None:
    """
    Set safe Hugging Face / torch defaults for Cloud and local runs.

    Idempotent via ``setdefault`` — existing process env wins.
    Call this **before** importing transformers / huggingface_hub.
    """
    try:
        os.environ.setdefault("HF_HUB_DISABLE_DISK_LOCK", "1")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("QUANTIZE_CPU", "0")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        # Heavy transformers (RoBERTa / emotions) crash Streamlit on OOM.
        # Opt in explicitly: ALLOW_HEAVY_NLP=1
        os.environ.setdefault("ALLOW_HEAVY_NLP", "0")
    except Exception as exc:  # pragma: no cover
        logger.warning("Cannot set runtime env defaults: %s", exc)
