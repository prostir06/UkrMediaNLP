# S18 — Semantic Search UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Parent:** [`2026-08-10-post-s17-roadmap.md`](2026-08-10-post-s17-roadmap.md)

**Goal:** Add a semantic search mode to corpus search that ranks articles by embedding cosine similarity when `ALLOW_EMBEDDINGS=1`, while keeping keyword search as default.

**Architecture:** Pure ranking lives in `nlp/corpus.py` + existing `nlp/embeddings.py` (hash encoder). Streamlit screen `ui/features/corpus_search.py` adds a mode radio and delegates to the new function. No DB schema changes in S18 — vectors computed in-memory per search (YAGNI until corpus sizes force caching).

**Tech Stack:** pandas, existing `nlp.embeddings`, Streamlit UI, pytest.

## Global Constraints

- Keep Streamlit out of `nlp/*` and `corpus_store/*`
- `ALLOW_EMBEDDINGS=0` by default (Cloud light)
- Do not add torch / sentence-transformers dependency in S18
- Catch `NLPAnalysisError` in UI; soft message when embeddings disabled
- Maintain cov ≥70% / cloud ≥50%
- Ukrainian UI strings for new controls

---

## File map

| File | Change |
|------|--------|
| `nlp/corpus.py` | Add `search_corpus_semantic()` |
| `nlp/embeddings.py` | Optional: `rank_by_similarity()` helper (only if keeps corpus.py thin) |
| `ui/features/corpus_search.py` | Mode radio, delegate semantic path, captions |
| `tests/test_corpus.py` | Unit tests for semantic ranking |
| `tests/test_feature_renders.py` | UI smoke for semantic mode (mocked) |
| `README.md` | Document semantic mode + `ALLOW_EMBEDDINGS` |

---

## Task 1: Failing tests for semantic ranking

**Files:** `tests/test_corpus.py`

- [ ] **Step 1:** Add test `test_search_corpus_semantic_ranks_similar_higher` with monkeypatched `ALLOW_EMBEDDINGS=1`:

```python
def test_search_corpus_semantic_ranks_similar_higher(monkeypatch):
    monkeypatch.setenv("ALLOW_EMBEDDINGS", "1")
    from nlp.corpus import search_corpus_semantic

    df = pd.DataFrame(
        {
            "title": ["Київ футбол", "Біржові котирування", "футбол у Києві"],
            "content": ["", "", ""],
            "source": ["A", "B", "C"],
            "published": ["2024-01-01"] * 3,
        }
    )
    out = search_corpus_semantic(df, "Київ футбол", min_score=0.0, top_k=3)
    assert len(out) >= 2
    titles = out["title"].tolist()
    assert titles[0] in {"Київ футбол", "футбол у Києві"}
    assert "Біржові котирування" not in titles[:1]
```

- [ ] **Step 2:** Add test `test_search_corpus_semantic_disabled_raises`:

```python
def test_search_corpus_semantic_disabled_raises(monkeypatch):
    monkeypatch.delenv("ALLOW_EMBEDDINGS", raising=False)
    from exceptions import NLPAnalysisError
    from nlp.corpus import search_corpus_semantic

    df = pd.DataFrame({"title": ["x"], "content": [""], "source": ["A"], "published": [""]})
    with pytest.raises(NLPAnalysisError, match="вимкнені"):
        search_corpus_semantic(df, "x")
```

- [ ] **Step 3:** Run tests — expect FAIL (function missing):

```bash
pytest tests/test_corpus.py -k semantic -v
```

- [ ] **Step 4:** Commit: `test: add failing semantic corpus search tests`

---

## Task 2: Implement `search_corpus_semantic`

**Files:** `nlp/corpus.py`

- [ ] **Step 1:** Import `embed_texts`, `cosine_similarity`, `embeddings_enabled` from `nlp.embeddings` and `NLPAnalysisError` from `exceptions`.

- [ ] **Step 2:** Add function (minimal):

```python
def search_corpus_semantic(
    df: pd.DataFrame,
    query: str,
    *,
    min_score: float = 0.0,
    top_k: int = 50,
    text_column: str = "search_blob",
) -> pd.DataFrame:
    """Rank rows by embedding cosine similarity to *query*."""
    q = (query or "").strip()
    if not q or df.empty:
        return df.iloc[0:0].copy()
    work = ensure_search_blobs(df)
    blob_col = text_column if text_column in work.columns else "search_blob"
    texts = work[blob_col].fillna("").astype(str).tolist()
    query_vec = embed_texts([q])[0]
    scores = [
        cosine_similarity(query_vec, doc_vec)
        for doc_vec in embed_texts(texts)
    ]
    out = work.copy()
    out["relevance"] = scores
    out = out.loc[out["relevance"] >= min_score]
    out = out.sort_values("relevance", ascending=False).head(top_k)
    out["snippet"] = [
        make_snippet(str(t), q) for t in out[blob_col].fillna("").astype(str)
    ]
    return out.reset_index(drop=True)
```

- [ ] **Step 3:** Wrap unexpected failures in `NLPAnalysisError(step="search_corpus_semantic")` — re-raise `NLPAnalysisError` from embeddings as-is.

- [ ] **Step 4:** Run tests:

```bash
pytest tests/test_corpus.py -k semantic -v
```

- [ ] **Step 5:** Commit: `feat: add search_corpus_semantic ranking`

---

## Task 3: Wire UI mode toggle

**Files:** `ui/features/corpus_search.py`

- [ ] **Step 1:** Add radio after subheader:

```python
search_mode = st.radio(
    "Режим пошуку",
    ("Ключові слова", "Семантичний"),
    horizontal=True,
    key="corpus_search_mode",
)
```

- [ ] **Step 2:** When mode is semantic and embeddings disabled, show `st.info(...)` and return early (do not call embed).

- [ ] **Step 3:** Branch call:

```python
if search_mode == "Семантичний":
    from nlp.corpus import search_corpus_semantic
    results = search_corpus_semantic(corpus_df, str(query), min_score=0.05, top_k=100)
else:
    results = search_corpus(...)
```

- [ ] **Step 4:** Hide lemma/whole-word checkboxes when semantic (not applicable) OR disable with caption.

- [ ] **Step 5:** Catch `NLPAnalysisError` instead of broad `Exception` for search path.

- [ ] **Step 6:** Commit: `feat: corpus search semantic mode in UI`

---

## Task 4: UI smoke test

**Files:** `tests/test_feature_renders.py`

- [ ] **Step 1:** Add `test_render_corpus_search_semantic_mode` — monkeypatch `embeddings_enabled` → True, mock `search_corpus_semantic`, set radio to return "Семантичний".

- [ ] **Step 2:** Run:

```bash
pytest tests/test_feature_renders.py -k corpus_search -v
```

- [ ] **Step 3:** Commit: `test: semantic corpus search UI smoke`

---

## Task 5: Docs & verification

**Files:** `README.md`

- [ ] **Step 1:** Under corpus search / config table, note: semantic mode requires `ALLOW_EMBEDDINGS=1` locally or Docker.

- [ ] **Step 2:** Full verification:

```bash
pytest -m "not slow and not postgres" --cov=. --cov-fail-under=70
ruff check .
python -m mypy
```

- [ ] **Step 3:** Commit: `docs: document semantic corpus search mode`

---

## Done when

- [ ] Keyword search unchanged (default mode)
- [ ] Semantic mode ranks similar UA headlines higher in unit test
- [ ] UI shows info when `ALLOW_EMBEDDINGS=0`
- [ ] No Streamlit imports added under `nlp/`
- [ ] CI green

## Out of scope (S18)

- Persisting embedding vectors in Postgres
- sentence-transformers / multilingual-e5 backend
- pgvector / FAISS index
- Semantic mode in corpus trends
