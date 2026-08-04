"""
Batch ingest CLI: scrape → upsert → purge (90-day retention).

Usage examples::

    python -m corpus_store.ingest --all --dry-run
    python -m corpus_store.ingest --category Новини
    docker compose --profile ingest run --rm ingest

Requires ``DATABASE_URL`` except for ``--dry-run``.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from media_sources import MEDIA_CATEGORIES, NEWS_SOURCES, sources_for_category
from corpus_store.db import is_store_configured, session_scope
from corpus_store.repository import RETENTION_DAYS, purge_older_than, upsert_articles
from data_loader import fetch_articles

logger = logging.getLogger(__name__)


def _resolve_sources(category: str | None, all_sources: bool) -> list[str]:
    """
    Resolve the list of ``NEWS_SOURCES`` keys to ingest.

    ``--all`` wins over category. Unknown categories yield an empty list
    (caller treats that as a configuration error).
    """
    try:
        if all_sources:
            return list(NEWS_SOURCES.keys())
        if category:
            return list(sources_for_category(category))
    except Exception as exc:
        logger.error("_resolve_sources failed: %s", exc)
        return []
    return []


def run_migrate() -> None:
    """
    Run ``alembic upgrade head`` (requires ``DATABASE_URL`` in the environment).

    Raises:
        RuntimeError: When Alembic exits non-zero or cannot be started.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            check=False,
        )
    except OSError as exc:
        logger.exception("run_migrate: cannot start alembic")
        raise RuntimeError(f"alembic upgrade head failed to start: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError("alembic upgrade head failed")


def ingest_sources(
    sources: list[str],
    *,
    dry_run: bool = False,
    purge: bool = True,
) -> dict[str, int]:
    """
    Fetch each source and upsert into the store.

    Per-source failures are counted in ``errors`` and do not abort the batch.
    Purge runs once at the end when ``purge`` is True.

    Returns:
        Summary dict: ``sources``, ``articles``, ``purged``, ``errors``.

    Raises:
        RuntimeError: When the store is not configured (non-dry-run).
    """
    summary = {"sources": 0, "articles": 0, "purged": 0, "errors": 0}
    if not sources:
        return summary

    if dry_run:
        logger.info(
            "dry-run: would ingest %s sources: %s",
            len(sources),
            sources,
        )
        summary["sources"] = len(sources)
        return summary

    if not is_store_configured():
        raise RuntimeError("DATABASE_URL is not configured")

    try:
        with session_scope() as session:
            for name in sources:
                summary["sources"] += 1
                try:
                    config = NEWS_SOURCES[name]
                    frame = fetch_articles(
                        source_name=name,
                        feed_url=config["rss_url"],
                        scraper_name=config.get("scraper", "generic"),
                    )
                    if frame is None or getattr(frame, "empty", True):
                        continue
                    summary["articles"] += int(upsert_articles(session, frame))
                except KeyError as exc:
                    summary["errors"] += 1
                    logger.warning("ingest: unknown / bad config for %s: %s", name, exc)
                except Exception as exc:
                    # Soft-fail one source so the rest of the batch continues.
                    summary["errors"] += 1
                    logger.warning("ingest failed for %s: %s", name, exc)

            if purge:
                try:
                    summary["purged"] = int(
                        purge_older_than(session, days=RETENTION_DAYS)
                    )
                except Exception as exc:
                    summary["errors"] += 1
                    logger.warning("ingest: retention purge failed: %s", exc)
    except Exception:
        logger.exception("ingest_sources: session aborted")
        raise

    return summary


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.

    Returns:
        ``0`` on success, ``1`` when some sources failed, ``2`` on config error.
    """
    parser = argparse.ArgumentParser(
        description="Ingest Ukrainian media into Postgres corpus",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--category",
        choices=list(MEDIA_CATEGORIES),
        help="Ingest all sources in a sidebar category",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Ingest every NEWS_SOURCES entry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sources only; no DB writes",
    )
    parser.add_argument(
        "--skip-migrate",
        action="store_true",
        help="Do not run alembic upgrade head before ingest",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Skip 90-day retention purge after ingest",
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed help / error; preserve its code.
        return int(exc.code) if isinstance(exc.code, int) else 2

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    sources = _resolve_sources(args.category, args.all)
    if not sources:
        logger.error("No sources resolved")
        return 2

    if not args.dry_run:
        if not is_store_configured():
            logger.error("DATABASE_URL is required")
            return 2
        if not args.skip_migrate:
            try:
                run_migrate()
            except RuntimeError as exc:
                logger.error("%s", exc)
                return 2

    try:
        summary = ingest_sources(
            sources,
            dry_run=args.dry_run,
            purge=not args.no_purge,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:
        logger.exception("ingest aborted: %s", exc)
        return 2

    logger.info(
        "ingest done sources=%s articles=%s purged=%s errors=%s",
        summary["sources"],
        summary["articles"],
        summary["purged"],
        summary["errors"],
    )
    return 1 if summary["errors"] and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
