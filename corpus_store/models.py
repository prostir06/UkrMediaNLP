"""
SQLAlchemy model for durable articles.

Table ``articles`` mirrors the durable-corpus design spec:
dedupe on ``url_hash``, 90-day retention driven by
``coalesce(published_at, scraped_at)``.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ``corpus_store`` ORM models."""


class Article(Base):
    """
    One scraped news article, uniquely identified by ``url_hash``.

    ``url_hash`` is SHA-256 of the canonical URL (stripped). Indexes support
    filtered loads by source/category/date and retention deletes by scrape time.
    """

    __tablename__ = "articles"
    __table_args__ = (
        # Composite index for "sources in range" queries used by the UI.
        Index("ix_articles_source_published", "source", "published_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # Full article body for corpus search; may be empty string when scrape failed.
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )
    scraped_ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
