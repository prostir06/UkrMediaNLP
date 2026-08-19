"""
Database engine and session helpers for the durable corpus store.

The store is optional: when ``DATABASE_URL`` is unset the UI keeps the
session-only corpus behaviour. Engines are cached process-wide so Streamlit
reruns reuse one connection pool.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Process-wide cache (Streamlit reruns stay in the same interpreter).
_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def is_store_configured() -> bool:
    """
    Return True when ``DATABASE_URL`` is set to a non-empty value.

    Whitespace-only values are treated as unset so misconfigured env files
    do not silently open an invalid engine.
    """
    try:
        url = os.environ.get("DATABASE_URL", "").strip()
    except (TypeError, AttributeError) as exc:  # pragma: no cover - env access is rarely broken
        logger.warning("is_store_configured: cannot read DATABASE_URL: %s", exc)
        return False
    return bool(url)


def get_database_url() -> str | None:
    """Return configured database URL or ``None`` when the store is offline."""
    try:
        url = os.environ.get("DATABASE_URL", "").strip()
    except (TypeError, AttributeError) as exc:  # pragma: no cover
        logger.warning("get_database_url failed: %s", exc)
        return None
    return url or None


def get_engine(url: str | None = None, *, force_new: bool = False) -> Engine:
    """
    Return a process-wide SQLAlchemy engine.

    Args:
        url: Optional explicit URL (used by tests). Defaults to ``DATABASE_URL``.
        force_new: Always create a fresh engine (tests / one-off scripts).

    Raises:
        RuntimeError: When no URL is available or the engine cannot be created.
    """
    global _ENGINE, _SESSION_FACTORY
    resolved = (url or get_database_url() or "").strip()
    if not resolved:
        raise RuntimeError("DATABASE_URL is not configured")

    # Reuse the cached engine unless the caller forced a new one or passed
    # an explicit URL (isolated session for tests).
    if _ENGINE is not None and not force_new and url is None:
        return _ENGINE

    try:
        engine = create_engine(resolved, pool_pre_ping=True)
    except (SQLAlchemyError, ValueError, TypeError) as exc:
        logger.exception("get_engine: create_engine failed")
        raise RuntimeError(f"Cannot create database engine: {exc}") from exc

    if url is None:
        _ENGINE = engine
        _SESSION_FACTORY = sessionmaker(bind=engine, expire_on_commit=False)
    return engine


def reset_engine() -> None:
    """
    Drop the cached engine.

    Safe to call when no engine exists. Dispose errors are logged but not
    raised so test teardown never masks the real assertion failure.
    """
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        try:
            _ENGINE.dispose()
        except Exception as exc:
            logger.warning("reset_engine: dispose failed: %s", exc)
    _ENGINE = None
    _SESSION_FACTORY = None


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """
    Yield a short-lived session; commit on success, rollback on error.

    Always closes the session in ``finally`` so connection-pool slots are
    returned even when the caller raises.
    """
    try:
        engine = get_engine(url=url, force_new=url is not None)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()
    except Exception:
        logger.exception("session_scope: failed to open session")
        raise

    try:
        yield session
        session.commit()
    except Exception:
        try:
            session.rollback()
        except Exception as rollback_exc:
            logger.warning("session_scope: rollback failed: %s", rollback_exc)
        raise
    finally:
        try:
            session.close()
        except Exception as close_exc:
            logger.warning("session_scope: close failed: %s", close_exc)
