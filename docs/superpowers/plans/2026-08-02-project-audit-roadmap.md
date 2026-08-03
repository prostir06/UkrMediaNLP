# UkrMediaNLP Project Audit & Development Roadmap

> **SUPERSEDED (2026-08-03):** See [`2026-08-03-post-s7-roadmap.md`](2026-08-03-post-s7-roadmap.md).  
> This document describes the pre-S7 baseline (`app.py` ~873, cov 60%, open SSRF P0s). S1–S7 are complete; do not implement from this file.

> **For agentic workers (historical):** Use this as the parent roadmap. Each sprint below can become its own implementation plan via `writing-plans`. Do not implement everything in one PR.

**Goal:** Harden security, reduce architectural hotspots, mature the corpus feature from RSS snapshot MVP to durable analytics, and raise operational quality.

**Current baseline:** `main@5b22538` · 29 RSS sources · ~259 unit tests · Streamlit Cloud light + Docker full · corpus search/trends shipped.

**Architecture:** Streamlit UI → cache/data_loader → rss/scraping → nlp/* → ui charts/renderers. Core NLP/ingestion stay Streamlit-free.

**Tech stack:** Python 3.11/3.12, Streamlit, pandas, spaCy, scikit-learn, optional transformers/torch, SQLite article cache, pytest, ruff, Docker Compose.

## Global Constraints

- Prefer pure `nlp/*` / ingestion changes with unit tests; keep Streamlit out of core modules
- TDD for security and corpus performance work
- Do not enable `ALLOW_HEAVY_NLP=1` as Cloud free-tier default
- No embeddings / semantic search until durable corpus store exists
- Commit frequently; keep PRs sprint-sized

## Snapshot findings

### Strengths
- Clear layering; `test_nlp_no_streamlit` guards NLP purity
- SSRF allowlist + private IP baseline, rate limit, retries, size caps
- SQLite TTL article cache; light vs full requirements split
- Corpus MVP with hybrid terms, trends, media compare + tests
- CI: py3.11/3.12, ruff, cloud-deps, docker smoke

### Critical gaps
1. **P0** `url_utils.get_allowed_domains` uses `parts[-2:]` → for `pravda.com.ua` also allows `com.ua` (suffix match)
2. **P0** `scraping._execute_http_get` / RSS fetch use `allow_redirects=True`; final URL checked only after hop
3. **P1** `app.py` ~873 lines — orchestration + all feature screens
4. **P1** Corpus “trends” are current-feed snapshots in `session_state`, not historical store
5. **P1** Corpus search uses `iterrows` + optional per-row spaCy; serial multi-source load
6. **P1** Broad `except Exception` soft-fails (~115) collapse errors into empty UI
7. **P1** Docker “full” still defaults `ALLOW_HEAVY_NLP=0`; `mem_limit: 2g`
8. **P1** Coverage gate 60% omits `app.py` and `ui/*`

---

### Sprint 1: SSRF hardening (P0)

**Files:**
- Modify: `url_utils.py`, `scraping.py`, `rss.py`
- Test: `tests/test_url_utils.py`, new redirect tests

**Deliverables:**
1. Replace `parts[-2:]` registrable-domain heuristic with **explicit allowlist** (full host + known parents from `extra`, optionally `publicsuffix2` later). Never auto-add `com.ua`.
2. Follow redirects manually: validate each `Location` with `is_allowed_url` before the next request; cap hop count.
3. Tests: `evil.com.ua` blocked; open-redirect chain to private IP / unknown host blocked; happy-path same-site redirect allowed.
4. Document threat model in `url_utils` module docstring.

**Verify:** `pytest tests/test_url_utils.py tests/test_rss.py tests/test_scraping.py -q` · ruff

**Done when:** P0 items closed; adversarial cases green.

---

### Sprint 2: Soft-fail policy + observability (P1)

**Files:**
- Modify: `nlp/corpus.py`, `data_loader.py`, selective `nlp/sentiment.py`, `app.py`
- Optionally: `exceptions.py` (result types)

**Deliverables:**
1. Distinguish **intentional empty** vs **failure**: log + typed error / UI warning, not silent empty chart.
2. Corpus search/trends: on unexpected error raise `NLPAnalysisError` or return `(df, error)` consumed by UI.
3. Add structured log fields: `source`, `step`, `elapsed_ms` for scrape and corpus load.
4. Metrics stub: scrape success rate per source (log or simple counter dict returned to UI caption).

**Verify:** unit tests for failure paths; no new silent empties in corpus aggregators.

---

### Sprint 3: Split `app.py` into feature modules (P1)

**Files:**
- Create: `ui/features/` (`snapshot.py`, `ngrams.py`, `sentiment_ui.py`, `corpus_search.py`, `corpus_trends.py`, `compare.py`, …)
- Modify: `app.py` → thin router + `main()`
- Move helpers: `_load_source`, corpus session helpers near features or `ui/session_corpus.py`

**Deliverables:**
1. `app.py` under ~250 lines (config, sidebar, dispatch only).
2. Each feature file owns one `render_*` + local widgets.
3. No behavior change; snapshot tests / existing helpers still pass.
4. Update README structure tree.

**Verify:** full `pytest -m "not slow"` · manual smoke of 3 functions including corpus.

---

### Sprint 4: Corpus search performance (P1)

**Files:**
- Modify: `nlp/corpus.py`, `ui/corpus_controls.py`
- Test: `tests/test_corpus.py`

**Deliverables:**
1. Precompute `search_blob` / optional lemma blob columns once after load (vectorized or batch spaCy).
2. Replace `iterrows` search with vectorized `str.contains` / prebuilt masks where possible.
3. Bounded concurrent source loading (e.g. 2–3 workers) respecting rate limiter.
4. Bench note in PR: N articles × M terms before/after (local only).

**Verify:** existing corpus tests + new perf regression guard (timeout or row count).

---

### Sprint 5: CI, Docker, health (P1)

**Files:**
- Modify: `.coveragerc`, `.github/workflows/tests.yml`, `docker-compose.yml`, `Dockerfile`, `config.SCRAPE_SAMPLE_URLS`, `README.md`

**Deliverables:**
1. Gradually include `ui/*` in coverage; raise fail-under 60→65→70 with omit list shrinking.
2. Compose profile `full-nlp` with `ALLOW_HEAVY_NLP=1` and higher `mem_limit` (e.g. 4g); document.
3. Fail build if spaCy model install fails (remove silent `|| true` or make explicit).
4. Expand scraper health sample URLs toward all 29 sources (or generate from `NEWS_SOURCES`).
5. Fix README test count (~228 → actual).

**Verify:** CI green locally mirrored commands; `docker compose --profile full-nlp config`.

---

### Sprint 6: Config / sentiment decomposition (P2)

**Files:**
- Create: e.g. `config/sources.yaml` or `sources.py` + loader
- Split: `nlp/sentiment_models.py`, `nlp/resource_guard.py`, `nlp/sentiment_inference.py`
- Remove: unused `NLP_FUNCTIONS`, noop `show_lda_toggle` (or wire it)

**Deliverables:**
1. Typed source schema (`TypedDict` / dataclass) + validation already in `validate_news_sources_schema`.
2. Sentiment file &lt; ~250 lines each.
3. Deduplicate HF env defaults (`runtime_env` single source of truth).

**Verify:** sentiment unit tests + config tests; no Streamlit import in new nlp modules.

---

### Sprint 7+: Durable corpus product (P1 product)

**Design prerequisite:** short brainstorm/spec for storage (SQLite vs Postgres), retention, PII.

**Deliverables (phased):**
1. Persistent articles table (canonical URL hash, source, published_dt, title, content, scraped_at).
2. Deduplicate on ingest; retention policy (e.g. 90 days).
3. Scheduled job / GitHub Action / cron container for incremental fetch.
4. Trends read from store, not only live session RSS.
5. Export CSV/Parquet of query results.
6. Normalize media compare by articles-per-day (rate, not raw count).

**Verify:** migration tests; empty-store UX; backfill script dry-run.

---

### Later backlog (P2 / research)

- Labeled Ukrainian news sentiment evaluation set + regression fixtures
- Entity-over-time charts (NER trends)
- Query language / morphological index (beyond spaCy batch)
- Dependency pinning + image SBOM scan
- mypy/pyright optional CI job
- Remove `nlp_analysis.py` facade once imports cleaned

---

## Suggested execution order

| Order | Sprint | Why first |
|------:|--------|-----------|
| 1 | SSRF | Blocks safe public deploy |
| 2 | Soft-fail policy | Makes later debugging possible |
| 3 | Split app.py | Unblocks parallel feature work |
| 4 | Corpus perf | UX pain on multi-source |
| 5 | CI/Docker/health | Prevents regressions |
| 6 | Config/sentiment split | Maintainability |
| 7+ | Durable corpus | Product differentiation |

## Out of scope for this roadmap document

- Implementing the sprints (create per-sprint plans when starting)
- Immediate `main` push to origin (still ahead of remote unless pushed)
- Cloud free-tier heavy transformers

## Success metrics (6 months)

- P0 SSRF closed with adversarial tests in CI
- `app.py` &lt; 300 LOC
- Corpus load of 5 sources interactive &lt; ~2–3 min typical (network-bound) with progress clarity
- Coverage gate ≥ 70% including UI helpers
- Optional: N days of retained articles for ≥1 category with scheduled ingest
