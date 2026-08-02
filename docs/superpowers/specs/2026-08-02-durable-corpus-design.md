# Durable Corpus Store — Design

**Date:** 2026-08-02  
**Status:** Draft for review  
**Sprint:** 7+ (UkrMediaNLP roadmap)

## Problem

Corpus search and topic trends today use a **session snapshot**: RSS + scrape into `st.session_state["corpus_df"]`. The SQLite `article_cache` only TTL-caches scrape payloads per source/feed cap — it is **not** a historical article store. Trends cannot span sessions; reloads re-scrape; media compare uses raw counts.

## Goals (v1)

1. Persist articles in **Postgres** with dedupe on canonical URL.
2. **90-day** retention.
3. **UI upsert** on corpus load + optional **scheduled ingest**.
4. Search/trends **prefer the store**; fall back to live RSS if empty/unavailable.
5. Export search hits as **CSV**; normalize media compare by **rate** (hits per articles-per-day).

## Non-goals (v1)

- Embeddings / semantic search  
- Multi-tenant auth  
- Precomputed lemma / `search_blob` columns in DB (session-side blobs OK after load)  
- Replacing SQLite scrape TTL cache  
- Parquet export  

## Decisions

| Topic | Choice |
|-------|--------|
| Storage | Postgres 16 |
| Access layer | SQLAlchemy 2.x + Alembic |
| Retention | 90 days |
| Ingest | UI write path + compose profile `ingest` |
| Payload | title, url, source, category, content, published_at, scraped_at, scraped_ok |
| Offline DB | Warn; keep current session-only behavior |

## Schema

Table `articles`:

| Column | Type | Notes |
|--------|------|-------|
| `id` | `BIGSERIAL` PK | |
| `url_hash` | `CHAR(64)` UNIQUE NOT NULL | SHA-256 of canonical URL |
| `url` | `TEXT` NOT NULL | Canonical `link` |
| `source` | `TEXT` NOT NULL | `NEWS_SOURCES` key |
| `category` | `TEXT` NOT NULL | Sidebar category |
| `title` | `TEXT` NOT NULL | |
| `content` | `TEXT` | Full scrape; may be empty |
| `published_at` | `TIMESTAMPTZ` NULL | |
| `scraped_at` | `TIMESTAMPTZ` NOT NULL DEFAULT `now()` | |
| `scraped_ok` | `BOOLEAN` NOT NULL DEFAULT false | |

Indexes:

- `UNIQUE (url_hash)`
- `(source, published_at DESC)`
- `(category, published_at DESC)`
- `(scraped_at)` for retention deletes

**Upsert:** `ON CONFLICT (url_hash) DO UPDATE` set `title`, `content`, `scraped_at`, `scraped_ok`, `source`, `category`; update `published_at` only when the incoming value is non-NULL (do not wipe a known publish time with NULL).

**Canonical URL:** `str(link).strip()` (no fragment stripping required in v1; document if added later).

## Architecture

```
UI corpus load ──► load_articles / build_corpus_from_sources
                         │
                         ├─► upsert_articles(df) ──► Postgres
                         └─► session corpus_df (+ ensure_search_blobs)

Search / Trends ──► load_corpus_from_store(filters)
                         │ empty / no DATABASE_URL
                         └─► live RSS path (existing) then upsert

ingest service ──► walk NEWS_SOURCES ──► scrape ──► upsert ──► purge_older_than(90d)
```

Modules (proposed):

- `corpus_store/models.py` — SQLAlchemy model  
- `corpus_store/db.py` — engine / session from `DATABASE_URL`  
- `corpus_store/repository.py` — upsert, query by sources/dates, purge  
- `corpus_store/ingest.py` — batch ingest CLI entry  
- `alembic/` — migrations  
- Wire: `ui/corpus_controls.py`, `ui/features/corpus_*.py`, `docker-compose.yml`

Existing `article_cache` (SQLite TTL) stays as scrape accelerator only.

## Config / ops

- `DATABASE_URL` — e.g. `postgresql+psycopg://ukrmedia:ukrmedia@postgres:5432/ukrmedia`  
- Compose: `postgres` service + volume; app `depends_on` healthy; profile `ingest` for scheduled/manual job  
- Migrations: `alembic upgrade head` (document in README; ingest entrypoint runs migrate)  
- Retention purge: on ingest run + optional throttled call when app starts if DB configured  

## UI behavior

- Caption after load: `Джерело: Postgres (N статей)` vs `Live RSS (store offline/empty)`  
- Failed upsert: `st.warning`, do not fail the session corpus  
- Search results: CSV download  
- Media compare / trend-by-source: prefer **rate** = hits / max(articles_in_bucket, 1) (or articles-per-day for that source); document exact formula in implementation plan  

## PII / security

- Stored text is public news content; still: do not log full `content`  
- Never commit DB dumps or `.env` with credentials  
- SSRF rules unchanged for scrape URLs  

## Testing

- Unit: url_hash stability, upsert conflict, purge boundary, query filters (use Postgres in CI via service container or `pytest` + testcontainers / compose service)  
- Migration smoke: `alembic upgrade head`  
- Empty-store UX: UI path falls back to live load  
- Ingest dry-run flag: parse sources, no write  

## Phased delivery

| Phase | Deliverable |
|-------|-------------|
| 7.1 | Schema + Alembic + repository upsert/query/purge + compose Postgres |
| 7.2 | Wire UI load/read + captions + CSV export |
| 7.3 | Ingest CLI/profile + retention schedule hook |
| 7.4 | Trends/compare rate normalization from store |

## Success criteria

- Reloading Streamlit still finds articles ingested earlier (within 90 days)  
- Duplicate URL does not create a second row  
- Rows older than 90 days are removed by purge  
- Without `DATABASE_URL`, app behaves as today  

## Open points (resolve in plan if needed)

- CI: GitHub Actions Postgres service vs skip store tests on Windows-only runners  
- Exact rate formula for zero-article buckets  
- Whether `published_at` or `scraped_at` drives retention when both exist → **v1: `coalesce(published_at, scraped_at) < now() - 90d`**
