"""Process-wide runtime defaults for Hugging Face / torch stability."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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
