# Durable Corpus Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist scraped articles in Postgres with upsert, 90-day retention, UI read/write, optional ingest job, and rate-normalized media compare.

**Architecture:** `corpus_store` package (SQLAlchemy model + repository) behind `DATABASE_URL`; Alembic migrations; compose `postgres` + profile `ingest`. UI upserts after live load and prefers store for search/trends with live fallback. SQLite `article_cache` unchanged.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.x, Alembic, psycopg3, Postgres 16, pytest, Docker Compose, Streamlit.

**Spec:** `docs/superpowers/specs/2026-08-02-durable-corpus-design.md`

## Global Constraints

- Postgres only for durable store (not SQLite)
- Retention: `coalesce(published_at, scraped_at) < now() - 90 days`
- Upsert on `url_hash` (SHA-256 of stripped URL); never wipe `published_at` with NULL
- Without `DATABASE_URL`: warn + session-only behavior (today’s UX)
- Do not log full `content`
- Keep Streamlit out of `corpus_store/*`
- TDD for repository; commit per phase

## File map

| File | Role |
|------|------|
| `corpus_store/__init__.py` | Public API re-exports |
| `corpus_store/db.py` | Engine/session from `DATABASE_URL` |
| `corpus_store/models.py` | `Article` ORM |
| `corpus_store/repository.py` | `url_hash`, upsert, query, purge |
| `corpus_store/ingest.py` | CLI batch ingest + dry-run |
| `alembic/` + `alembic.ini` | Migrations |
| `docker-compose.yml` | `postgres` + profile `ingest` |
| `requirements.txt` / `requirements-dev.txt` | sqlalchemy, alembic, psycopg |
| `ui/corpus_controls.py`, `ui/session_corpus.py`, features | Wire load/read/CSV |
| `nlp/corpus.py` | Rate helper for compare/trends |
| `tests/test_corpus_store.py` | Repository tests (SQLite in-memory OK for unit; optional Postgres) |

---

### Task 1: Dependencies + DB session bootstrap

**Files:**
- Modify: `requirements.txt`, `requirements-dev.txt`
- Create: `corpus_store/db.py`, `corpus_store/__init__.py`
- Test: `tests/test_corpus_store.py`

- [x] Add `sqlalchemy>=2.0,<3`, `psycopg[binary]>=3.1,<4`, `alembic>=1.13,<2`
- [x] `get_engine()` / `session_scope()`; `is_store_configured()` → bool from `DATABASE_URL`
- [x] Test: `is_store_configured` false when env unset
- [ ] Commit

### Task 2: Model + repository (url_hash, upsert, query, purge)

**Files:**
- Create: `corpus_store/models.py`, `corpus_store/repository.py`
- Test: `tests/test_corpus_store.py`

- [x] Failing tests: hash stability; upsert dedupe; purge with coalesce; query by sources/dates
- [x] Implement `Article` + repository using SQLAlchemy; unit tests may use `sqlite+pysqlite:///:memory:` with JSON/compat types if needed, or Postgres URL from env `TEST_DATABASE_URL`
- [x] Prefer **Postgres dialect in CI**; locally allow skip if no DB (`pytest.importorskip` / marker `postgres`)
- [x] For Windows/dev without Postgres: implement repository against SQLAlchemy and run tests with Postgres service when available; add in-memory SQLite tests only for pure helpers (`canonical_url_hash`)
- [ ] Commit

### Task 3: Alembic + compose Postgres

**Files:**
- Create: `alembic.ini`, `alembic/env.py`, `alembic/versions/*_articles.py`
- Modify: `docker-compose.yml`, `README.md`, `.gitignore` (`.env`)

- [x] Initial migration matching spec schema + indexes
- [x] Compose `postgres:16-alpine`, volume, healthcheck, `DATABASE_URL` on app
- [x] Document migrate + connection string
- [ ] Commit

### Task 4: Wire UI upsert + store read + captions + CSV

**Files:**
- Modify: `ui/corpus_controls.py`, `ui/session_corpus.py` / `app.py` commit helpers, `ui/features/corpus_search.py`
- Test: controls/helpers with mocked repository

- [x] After successful corpus build: `upsert_articles` if configured
- [x] Prefer `load_corpus_from_store` when sources selected and store has rows
- [x] Captions + search CSV download
- [ ] Commit

### Task 5: Ingest CLI + retention + profile

**Files:**
- Create: `corpus_store/ingest.py`, optionally `scripts/run_corpus_ingest.sh`
- Modify: `docker-compose.yml` profile `ingest`

- [x] CLI: `--category` / `--all`, `--dry-run`, migrate + purge
- [x] Compose service using same image, command ingest
- [ ] Commit

### Task 6: Rate normalization

**Files:**
- Modify: `nlp/corpus.py` (`aggregate_trends_by_source` or helper), `ui/features/compare.py` / corpus trends
- Test: `tests/test_corpus.py`

- [x] Formula: `rate = hits / max(articles_in_bucket_for_source, 1)`
- [x] Wire chart/table to show rate (keep raw count column optional)
- [ ] Commit

## Verify

```bash
alembic upgrade head
pytest -m "not slow" -q
docker compose --profile ingest config
ruff check corpus_store tests/test_corpus_store.py
```
