# Sprint 2: Soft-fail Policy + Observability

> **For agentic workers:** Implement task-by-task with TDD. Checkboxes track progress.

**Goal:** Distinguish intentional empty results from failures; add structured step timing logs and scrape success attrs for UI.

**Architecture:** Corpus aggregators raise `NLPAnalysisError` on unexpected errors (UI already catches). `observability.log_step` emits `step` / `source` / `elapsed_ms`. `fetch_articles` stores `scrape_stats` on `DataFrame.attrs`.

**Tech Stack:** Python logging, pandas attrs, existing `exceptions.NLPAnalysisError`.

## Global Constraints

- Keep Streamlit out of `nlp/*` and `data_loader.py`
- TDD for failure-path unit tests
- Prefer raise over silent empty for search/trends aggregators
- Commit at end of sprint

---

### Task 1: Corpus failure vs empty

- [ ] Failing tests: `search_corpus` / `aggregate_trends*` raise `NLPAnalysisError` when inner step blows up; empty query still returns empty DF
- [ ] Replace bare `except Exception: return empty` in those three functions with `raise NLPAnalysisError(..., step=...) from exc`
- [ ] Keep intentional empties (no query / no dates / no terms)

### Task 2: Structured `log_step`

- [ ] Add `observability.py` with `log_step` context manager
- [ ] Use in `fetch_articles` (scrape) and `build_corpus_from_sources` (corpus_load)
- [ ] Unit-test log message fields via `caplog`

### Task 3: Scrape metrics stub

- [ ] Set `df.attrs["scrape_stats"] = {source, ok, total, elapsed_ms}`
- [ ] Collect per-source stats on corpus merge attrs
- [ ] Caption in snapshot / corpus commit when stats present
- [ ] Selective: `classify_sentiment_batch` raises `NLPAnalysisError` instead of all-Neutral soft-fail

### Verify

`pytest tests/test_corpus.py tests/test_data_loader.py tests/test_observability.py tests/test_sentiment*.py -q` · ruff · commit
