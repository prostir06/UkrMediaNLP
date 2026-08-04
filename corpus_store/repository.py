"""
Repository helpers for durable article upsert / query / purge.

Design notes
------------
* Deduplication key is ``url_hash`` = SHA-256 of ``str(url).strip()``.
* Upsert never replaces a known ``published_at`` with NULL
  (``coalesce(excluded, existing)``).
* Unit tests may use SQLite; production uses Postgres. The insert dialect
  is selected from ``session.get_bind().dialect.name``.
* Never log full article ``content`` (public news, but still noisy / large).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from sqlalchemy import and_, delete, func, or_, select, true
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from corpus_store.models import Article

logger = logging.getLogger(__name__)

# Spec default: keep roughly three months of history.
RETENTION_DAYS = 90

# Columns returned by ``load_corpus_from_store`` (aligned with RSS frames).
_CORPUS_COLUMNS = [
    "title",
    "link",
    "description",
    "published",
    "category",
    "content",
    "source",
    "scraped_ok",
    "search_blob",
]


def _build_search_blob(title: object, content: object, description: object = "") -> str:
    """Build the same title+content blob used by ``nlp.corpus.ensure_search_blobs``."""
    title_text = str(title or "").strip()
    content_text = str(content or "").strip()
    if not content_text:
        content_text = str(description or "").strip()
    return f"{title_text}\n{content_text}".strip()


def canonical_url_hash(url: str) -> str:
    """
    Return SHA-256 hex digest of the stripped canonical URL.

    Empty / None inputs hash as the empty string after strip so callers can
    still compare digests without raising.
    """
    try:
        canonical = str(url or "").strip()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception as exc:
        # Extremely unlikely (encode failures); surface as empty-safe digest.
        logger.warning("canonical_url_hash failed for %r: %s", url, exc)
        return hashlib.sha256(b"").hexdigest()


def _as_utc(value: object) -> datetime | None:
    """
    Coerce timestamps / date strings to timezone-aware UTC.

    Returns ``None`` for missing, NaN, or unparseable values so upsert can
    leave an existing ``published_at`` untouched.
    """
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        ts = parsed.to_pydatetime()
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception as exc:
        logger.debug("_as_utc could not parse %r: %s", value, exc)
        return None


def _row_to_article_dict(row: object, scraped_at: datetime) -> dict | None:
    """
    Map one DataFrame row (Series or mapping) to an Article column dict.

    Returns ``None`` when the row has no usable URL (cannot dedupe).
    """
    try:
        if isinstance(row, pd.Series):
            get = row.get
        elif isinstance(row, dict):
            get = row.get
        else:
            return None
        url = str(get("link", "") or "").strip()
        if not url:
            return None
        digest = canonical_url_hash(url)
        title = str(get("title", "") or "").strip() or "(без заголовка)"
        content = str(get("content", "") or "")
        description = str(get("description", "") or "")
        published = _as_utc(get("published_dt", get("published")))
        existing_blob = str(get("search_blob", "") or "").strip()
        return {
            "url_hash": digest,
            "url": url,
            "source": str(get("source", "") or ""),
            "category": str(get("category", "") or ""),
            "title": title,
            "content": content,
            "search_blob": existing_blob
            or _build_search_blob(title, content, description),
            "published_at": published,
            "scraped_at": scraped_at,
            "scraped_ok": bool(get("scraped_ok", False)),
        }
    except Exception as exc:
        logger.warning("_row_to_article_dict skipped row: %s", exc)
        return None


def upsert_articles(session: Session, df: pd.DataFrame) -> int:
    """
    Upsert article rows from a corpus DataFrame.

    Last occurrence of each ``url_hash`` inside *df* wins (batch dedupe).
    On conflict the DB row is updated; ``published_at`` is preserved when
    the incoming value is NULL.

    Returns:
        Number of unique ``url_hash`` values written (0 on soft failure).

    Raises:
        SQLAlchemyError: Propagated when the database statement fails after
            a non-empty payload was prepared (caller decides rollback).
    """
    if df is None or getattr(df, "empty", True):
        return 0

    try:
        work = df.copy()
    except Exception as exc:
        logger.warning("upsert_articles: cannot copy DataFrame: %s", exc)
        return 0

    if "link" not in work.columns:
        logger.warning("upsert_articles: missing link column")
        return 0

    # Last-write-wins map keyed by url_hash (avoids conflicting multi-row
    # INSERT batches that SQLite does not self-resolve).
    by_hash: dict[str, dict] = {}
    now = datetime.now(timezone.utc)
    try:
        for record in work.to_dict(orient="records"):
            payload = _row_to_article_dict(record, now)
            if payload is None:
                continue
            by_hash[payload["url_hash"]] = payload
    except Exception as exc:
        logger.exception("upsert_articles: row iteration failed: %s", exc)
        return 0

    rows = list(by_hash.values())
    if not rows:
        return 0

    try:
        bind = session.get_bind()
        dialect = bind.dialect.name if bind is not None else "sqlite"
        insert = pg_insert if dialect == "postgresql" else sqlite_insert
        stmt = insert(Article).values(rows)
        update_cols = {
            "url": stmt.excluded.url,
            "source": stmt.excluded.source,
            "category": stmt.excluded.category,
            "title": stmt.excluded.title,
            "content": stmt.excluded.content,
            "search_blob": stmt.excluded.search_blob,
            "scraped_at": stmt.excluded.scraped_at,
            "scraped_ok": stmt.excluded.scraped_ok,
            # Keep previous publish time when the new scrape has no date.
            "published_at": func.coalesce(
                stmt.excluded.published_at,
                Article.published_at,
            ),
        }
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["url_hash"],
                set_=update_cols,
            )
        )
    except SQLAlchemyError:
        logger.exception(
            "upsert_articles: DB write failed (rows=%s dialect=%s)",
            len(rows),
            dialect if "dialect" in locals() else "?",
        )
        raise

    return len(rows)


def load_corpus_from_store(
    session: Session,
    sources: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    categories: list[str] | None = None,
    *,
    include_missing_dates: bool = True,
) -> pd.DataFrame:
    """
    Load matching articles into a DataFrame compatible with corpus helpers.

    Date filters use ``published_at``. When ``include_missing_dates`` is True,
    rows with NULL publish time are kept alongside in-range rows.

    Returns:
        Empty DataFrame with the standard corpus columns on soft failure
        (bad filters). SQLAlchemy errors propagate to the caller.
    """
    empty = pd.DataFrame(columns=_CORPUS_COLUMNS)
    clauses: list[Any] = []
    try:
        if sources:
            clauses.append(Article.source.in_(list(sources)))
        if categories:
            clauses.append(Article.category.in_(list(categories)))

        start = _as_utc(date_from) if date_from is not None else None
        end = _as_utc(date_to) if date_to is not None else None
        if start is not None or end is not None:
            date_ok: list[Any] = []
            if start is not None:
                date_ok.append(Article.published_at >= start)
            if end is not None:
                # Inclusive calendar day: keep everything before next midnight.
                date_ok.append(Article.published_at < end + timedelta(days=1))
            in_range = and_(*date_ok) if date_ok else true()
            if include_missing_dates:
                clauses.append(or_(Article.published_at.is_(None), in_range))
            else:
                clauses.append(in_range)

        stmt = select(Article)
        if clauses:
            stmt = stmt.where(and_(*clauses))
        stmt = stmt.order_by(Article.published_at.desc().nullslast())
        articles = list(session.scalars(stmt))
    except SQLAlchemyError:
        logger.exception("load_corpus_from_store: query failed")
        raise
    except Exception as exc:
        logger.warning("load_corpus_from_store: filter build failed: %s", exc)
        return empty

    if not articles:
        return empty

    records = []
    for article in articles:
        try:
            published = article.published_at
            blob = getattr(article, "search_blob", None) or ""
            if not str(blob).strip():
                blob = _build_search_blob(article.title, article.content or "")
            records.append(
                {
                    "title": article.title,
                    "link": article.url,
                    "description": "",
                    "published": published.isoformat() if published else "",
                    "category": article.category,
                    "content": article.content or "",
                    "source": article.source,
                    "scraped_ok": bool(article.scraped_ok),
                    "search_blob": str(blob),
                }
            )
        except Exception as exc:
            logger.warning("load_corpus_from_store: skip bad ORM row: %s", exc)

    try:
        return pd.DataFrame.from_records(records) if records else empty
    except Exception as exc:
        logger.warning("load_corpus_from_store: DataFrame build failed: %s", exc)
        return empty


def purge_older_than(
    session: Session,
    *,
    days: int = RETENTION_DAYS,
    now: datetime | None = None,
) -> int:
    """
    Delete articles older than *days* by ``coalesce(published_at, scraped_at)``.

    Returns:
        Number of deleted rows, or 0 when *days* <= 0 / on soft prep failure.

    Raises:
        SQLAlchemyError: When the DELETE statement fails.
    """
    if days <= 0:
        return 0

    try:
        moment = now or datetime.now(timezone.utc)
        cutoff = moment - timedelta(days=days)
        effective = func.coalesce(Article.published_at, Article.scraped_at)
        result = session.execute(
            delete(Article).where(effective < cutoff)
        )
        return int(getattr(result, "rowcount", 0) or 0)
    except SQLAlchemyError:
        logger.exception("purge_older_than: delete failed (days=%s)", days)
        raise
    except Exception as exc:
        logger.warning("purge_older_than: unexpected error: %s", exc)
        return 0
