# S8 — Corpus refresh + store reliability

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or executing-plans. Parent: `2026-08-03-post-s7-roadmap.md`.

**Goal:** Force RSS corpus refresh when store is preferred; catch Postgres dialect + migrate failures in CI/Docker.

**Architecture:** Checkbox `force_refresh` skips `load_corpus_from_store`; live path always upserts. CI adds Postgres 16 service for dialect upsert smoke. Entrypoint fails hard on alembic when `DATABASE_URL` is set.

**Tech Stack:** Streamlit sidebar, pytest `postgres` marker, GitHub Actions services, Alembic.

## Global Constraints

- Keep Streamlit out of `corpus_store/*`
- Soft-fail upsert still must not crash session corpus
- Without `TEST_DATABASE_URL`, postgres-marked tests skip

---

### Task 1: Force-refresh UI + load path

**Files:** `ui/corpus_controls.py`, `app.py`, `tests/test_corpus_controls.py`

- [x] Checkbox `corpus_force_rss`
- [x] `load_corpus_into_session(..., force_refresh=False)`
- [x] Tests prefer vs force

### Task 2: Postgres CI dialect smoke

**Files:** `.github/workflows/tests.yml`, `pytest.ini`, `tests/test_corpus_store_postgres.py`

- [x] Postgres service + `TEST_DATABASE_URL`
- [x] Marker `postgres`; upsert/load smoke

### Task 3: Docker alembic hard-fail

**Files:** `scripts/docker_entrypoint.sh`

- [x] `alembic upgrade head` without `|| echo` when `DATABASE_URL` set
