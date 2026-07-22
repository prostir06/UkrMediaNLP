"""Entry point for Streamlit Community Cloud / local ``streamlit run``.

Must call ``main()`` at module level — Streamlit executes this file as a script
and expects the app to register widgets during that run.
"""

from __future__ import annotations

import logging
import sys

from runtime_env import apply_runtime_env

logger = logging.getLogger(__name__)

apply_runtime_env()


def _run() -> None:
    """Import and start the Streamlit UI with a clear failure path."""
    try:
        from app import main

        main()
    except MemoryError:
        logger.exception("Streamlit entry OOM")
        print(
            "Недостатньо пам'яті для запуску UkrMediaNLP. "
            "Закрийте інші програми або використайте requirements-cloud.txt.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except Exception as exc:
        logger.exception("Streamlit entry failed")
        print(f"Не вдалося запустити UkrMediaNLP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


_run()
