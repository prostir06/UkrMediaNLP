# Sprint 4: Corpus Search Performance

> **For agentic workers:** TDD. Preserve search/trends semantics from existing tests.

**Goal:** Faster corpus search and multi-source load without behavior change.

**Architecture:** Precompute `search_blob_*` columns once; vectorized `str.contains` masks; optional batch lemma column; `ThreadPoolExecutor` (2–3) for source loads.

**Tech Stack:** pandas vectorized string ops, concurrent.futures, existing spaCy lemmatize helper.

## Tasks

1. `ensure_search_blobs` / optional `ensure_lemma_blobs`; wire after corpus merge
2. Rewrite `search_corpus` + `_term_hit_mask` without `iterrows`/`apply`
3. Concurrent `build_corpus_from_sources` with `CORPUS_LOAD_WORKERS`
4. Perf guard test (N rows finishes under timeout); update failure-path monkeypatch
5. pytest + ruff + commit
