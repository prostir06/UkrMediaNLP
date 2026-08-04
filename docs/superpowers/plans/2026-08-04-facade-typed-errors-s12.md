# S12 — Typed errors + façade finish

> **Parent:** `2026-08-04-post-s11-roadmap.md`

**Goal:** Prefer typed errors on hot paths; finish façade cleanup so `app` does not need `nlp_analysis`, and ingest/SSRF import `media_sources`.

**Done when:**
- `app.py` imports `preprocess_texts` from `nlp.preprocessing`
- `nlp_analysis.py` remains a thin deprecated re-export only
- `url_utils`, `corpus_store.ingest`, `scraper_health_check` import registry from `media_sources`
- Hot-path handlers catch `DataLoaderError` / `NLPAnalysisError` before broad `Exception`
