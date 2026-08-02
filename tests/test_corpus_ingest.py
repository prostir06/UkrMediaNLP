"""
Unit tests for corpus ingest CLI helpers.

Focus: dry-run, per-source soft-fail, migrate/config exit codes (PEP 8).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from corpus_store import ingest as ingest_mod


def test_resolve_sources_all(monkeypatch):
    monkeypatch.setattr(ingest_mod, "NEWS_SOURCES", {"A": {}, "B": {}})
    assert ingest_mod._resolve_sources(None, True) == ["A", "B"]


def test_resolve_sources_category(monkeypatch):
    monkeypatch.setattr(
        ingest_mod,
        "sources_for_category",
        lambda category: ["NV"] if category == "Новини" else [],
    )
    assert ingest_mod._resolve_sources("Новини", False) == ["NV"]
    assert ingest_mod._resolve_sources(None, False) == []


def test_resolve_sources_handles_exception(monkeypatch):
    def boom(_category):
        raise RuntimeError("bad catalog")

    monkeypatch.setattr(ingest_mod, "sources_for_category", boom)
    assert ingest_mod._resolve_sources("Новини", False) == []


def test_ingest_dry_run_skips_db(monkeypatch):
    calls = []

    def boom(*_args, **_kwargs):
        calls.append("fetch")
        raise AssertionError("should not fetch on dry-run")

    monkeypatch.setattr(ingest_mod, "fetch_articles", boom)
    summary = ingest_mod.ingest_sources(["NV"], dry_run=True)
    assert summary["sources"] == 1
    assert summary["articles"] == 0
    assert calls == []


def test_ingest_requires_database_url(monkeypatch):
    monkeypatch.setattr(ingest_mod, "is_store_configured", lambda: False)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        ingest_mod.ingest_sources(["NV"], dry_run=False)


def test_ingest_soft_fails_one_source(monkeypatch):
    monkeypatch.setattr(ingest_mod, "is_store_configured", lambda: True)

    class _FakeSession:
        pass

    class _Scope:
        def __enter__(self):
            return _FakeSession()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(ingest_mod, "session_scope", lambda: _Scope())
    monkeypatch.setattr(
        ingest_mod,
        "NEWS_SOURCES",
        {
            "Good": {"rss_url": "http://x", "scraper": "generic"},
            "Bad": {"rss_url": "http://y", "scraper": "generic"},
        },
    )

    def fake_fetch(source_name, feed_url, scraper_name):
        if source_name == "Bad":
            raise RuntimeError("scrape failed")
        return pd.DataFrame(
            {
                "title": ["t"],
                "link": ["https://example.com/g"],
                "content": ["c"],
                "source": ["Good"],
                "category": ["Новини"],
                "published": ["2024-01-01"],
                "scraped_ok": [True],
            }
        )

    monkeypatch.setattr(ingest_mod, "fetch_articles", fake_fetch)
    monkeypatch.setattr(ingest_mod, "upsert_articles", lambda _s, _df: 1)
    monkeypatch.setattr(ingest_mod, "purge_older_than", lambda _s, days=90: 3)

    summary = ingest_mod.ingest_sources(["Good", "Bad"], dry_run=False, purge=True)
    assert summary["sources"] == 2
    assert summary["articles"] == 1
    assert summary["errors"] == 1
    assert summary["purged"] == 3


def test_ingest_purge_failure_counts_as_error(monkeypatch):
    monkeypatch.setattr(ingest_mod, "is_store_configured", lambda: True)

    class _Scope:
        def __enter__(self):
            return MagicMock()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(ingest_mod, "session_scope", lambda: _Scope())
    monkeypatch.setattr(ingest_mod, "NEWS_SOURCES", {})
    monkeypatch.setattr(
        ingest_mod,
        "purge_older_than",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("purge")),
    )
    summary = ingest_mod.ingest_sources(["X"], dry_run=False, purge=True)
    # Unknown source KeyError counts + purge error.
    assert summary["errors"] >= 1


def test_run_migrate_raises_on_nonzero(monkeypatch):
    monkeypatch.setattr(
        ingest_mod.subprocess,
        "run",
        lambda *_a, **_k: MagicMock(returncode=1),
    )
    with pytest.raises(RuntimeError, match="alembic"):
        ingest_mod.run_migrate()


def test_main_dry_run_exit_zero(monkeypatch):
    monkeypatch.setattr(
        ingest_mod,
        "_resolve_sources",
        lambda category, all_sources: ["NV"],
    )
    monkeypatch.setattr(
        ingest_mod,
        "ingest_sources",
        lambda sources, dry_run=False, purge=True: {
            "sources": 1,
            "articles": 0,
            "purged": 0,
            "errors": 0,
        },
    )
    assert ingest_mod.main(["--all", "--dry-run"]) == 0


def test_main_returns_2_without_sources(monkeypatch):
    monkeypatch.setattr(ingest_mod, "_resolve_sources", lambda *_a, **_k: [])
    assert ingest_mod.main(["--all", "--dry-run"]) == 2


def test_main_returns_1_on_source_errors(monkeypatch):
    monkeypatch.setattr(ingest_mod, "_resolve_sources", lambda *_a, **_k: ["NV"])
    monkeypatch.setattr(ingest_mod, "is_store_configured", lambda: True)
    monkeypatch.setattr(ingest_mod, "run_migrate", lambda: None)
    monkeypatch.setattr(
        ingest_mod,
        "ingest_sources",
        lambda *_a, **_k: {
            "sources": 1,
            "articles": 0,
            "purged": 0,
            "errors": 2,
        },
    )
    assert ingest_mod.main(["--all", "--skip-migrate"]) == 1
