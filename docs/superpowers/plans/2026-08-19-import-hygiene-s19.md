# S19 — Import Hygiene & Façade Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Parent:** [`2026-08-19-post-s18-roadmap.md`](2026-08-19-post-s18-roadmap.md)

**Goal:** Make `config.py` caps/env-only (no Streamlit at import), drop `nlp_analysis.py`, and import registry helpers from `media_sources`.

**Architecture:** `media_sources.py` remains the canonical NEWS_SOURCES registry. `runtime_env.py` (or a small `cloud_mode.py`) owns `get_cloud_light`. `config.py` keeps NLP function lists and numeric caps. AST tests replace façade guards.

**Tech Stack:** Python 3.11/3.12, pytest, ruff, existing AST tests in `tests/test_nlp_no_streamlit.py`.

## Global Constraints

- Keep Streamlit out of `nlp/*` and `corpus_store/*`
- Do not change NLP/search behaviour
- TDD: failing AST/import tests first
- Cloud defaults unchanged (`ALLOW_HEAVY_NLP=0`)
- Caps (`MAX_ARTICLES`, etc.) stay in `config.py`

---

## File map

| File | Change |
|------|--------|
| `tests/test_nlp_no_streamlit.py` | Assert `config.py` has no top-level streamlit; drop `nlp_analysis` module list |
| `tests/test_config.py` | Import `get_cloud_light` from new module |
| `runtime_env.py` or `cloud_mode.py` | Host `get_cloud_light` |
| `config.py` | Remove Streamlit; optional re-export documented or removed from `__all__` |
| `app.py`, `ui/features/corpus_trends.py` | Import `get_cloud_light` from new home |
| `cache.py` | `from media_sources import get_source_config` |
| `nlp_analysis.py` | Delete |
| `README.md` | Drop façade from tree; test count ~368 |

---

## Task 1: Failing tests for import graph

**Files:** `tests/test_nlp_no_streamlit.py`

- [ ] **Step 1:** Add `test_config_source_has_no_streamlit_import` — AST-walk `config.py`; fail if `import streamlit` or `from streamlit`.

- [ ] **Step 2:** Change `test_app_does_not_import_nlp_analysis_facade` to also assert `nlp_analysis.py` **does not exist** (or skip until Task 4). Prefer: keep façade test until delete, then replace with `test_nlp_analysis_module_removed`.

- [ ] **Step 3:** Run — expect FAIL on Streamlit-in-config:

```bash
pytest tests/test_nlp_no_streamlit.py tests/test_config.py -q --tb=short
```

- [ ] **Step 4:** Commit: `test: fail if config imports streamlit`

---

## Task 2: Move `get_cloud_light`

**Files:** `runtime_env.py`, `config.py`, `app.py`, `ui/features/corpus_trends.py`, `tests/test_config.py`, `tests/test_app_helpers.py`

- [ ] **Step 1:** Move `_truthy_flag`, `_transformers_available`, `get_cloud_light` to `runtime_env.py` (already applies env; secrets stay try/except).

- [ ] **Step 2:** `config.py` may re-export `get_cloud_light` **without** importing streamlit itself — re-export from `runtime_env` is OK for one release.

- [ ] **Step 3:** Update call sites to `from runtime_env import get_cloud_light` (`app.py`, `corpus_trends.py`).

- [ ] **Step 4:** Update tests that patch `config.get_cloud_light` to patch `runtime_env` or the feature module.

- [ ] **Step 5:** Run `pytest tests/test_config.py tests/test_nlp_no_streamlit.py tests/test_app_helpers.py -q`

- [ ] **Step 6:** Commit: `refactor: move get_cloud_light off config Streamlit import`

---

## Task 3: Cache / registry imports

**Files:** `cache.py`

- [ ] **Step 1:** `from media_sources import get_source_config` instead of `config`.

- [ ] **Step 2:** Run `pytest tests/test_cache.py -q` (or whatever cache tests exist).

- [ ] **Step 3:** Commit: `refactor: cache imports get_source_config from media_sources`

---

## Task 4: Remove `nlp_analysis.py`

**Files:** `nlp_analysis.py`, `tests/test_nlp_no_streamlit.py`, `README.md`

- [ ] **Step 1:** Grep repo for `nlp_analysis` (excluding this plan).

- [ ] **Step 2:** Delete `nlp_analysis.py`; update tests: remove from import-without-streamlit list; add `test_nlp_analysis_file_absent`.

- [ ] **Step 3:** README project tree: remove façade line.

- [ ] **Step 4:** `pytest tests/test_nlp_no_streamlit.py -q` + `ruff check .`

- [ ] **Step 5:** Commit: `refactor: remove deprecated nlp_analysis façade`

---

## Task 5: Docs and verification

**Files:** `README.md`, `docs/superpowers/plans/2026-08-19-post-s18-roadmap.md`

- [ ] **Step 1:** README tests ~368.

- [ ] **Step 2:**

```bash
pytest -m "not slow and not postgres" -q
ruff check .
python -m mypy
```

- [ ] **Step 3:** Mark S19 Done in post-s18 roadmap.

- [ ] **Step 4:** Commit: `docs: S19 import hygiene complete`

---

## Done when

- [ ] `config.py` AST has no streamlit
- [ ] `nlp_analysis.py` gone
- [ ] `cache.py` uses `media_sources.get_source_config`
- [ ] Caps still imported from `config` in UI (intentional)
- [ ] CI green

## Out of scope (S19)

- Reducing `except Exception` counts (S20)
- Scheduled ingest (S21)
- Lockfile (S23)
