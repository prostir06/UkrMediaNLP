"""Structured step logging for scrape and corpus pipelines."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def log_step(
    logger: logging.Logger,
    step: str,
    source: str = "",
) -> Iterator[None]:
    """
    Log ``step`` / ``source`` / ``status`` / ``elapsed_ms`` around a block.

    Re-raises any exception after emitting ``status=error``.
    """
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "step=%s source=%s status=%s elapsed_ms=%s",
            step,
            source or "-",
            status,
            elapsed_ms,
        )
