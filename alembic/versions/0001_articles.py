"""create articles table

Revision ID: 0001_articles
Revises:
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001_articles"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("url_hash", sa.String(length=64), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scraped_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scraped_ok", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("url_hash", name="uq_articles_url_hash"),
    )
    op.create_index("ix_articles_source", "articles", ["source"], unique=False)
    op.create_index("ix_articles_category", "articles", ["category"], unique=False)
    op.create_index("ix_articles_published_at", "articles", ["published_at"], unique=False)
    op.create_index("ix_articles_scraped_at", "articles", ["scraped_at"], unique=False)
    op.create_index("ix_articles_source_published", "articles", ["source", "published_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_articles_source_published", table_name="articles")
    op.drop_index("ix_articles_scraped_at", table_name="articles")
    op.drop_index("ix_articles_published_at", table_name="articles")
    op.drop_index("ix_articles_category", table_name="articles")
    op.drop_index("ix_articles_source", table_name="articles")
    op.drop_table("articles")
