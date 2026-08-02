# Sprint 3: Split app.py into feature modules

> **For agentic workers:** Mechanical move; no behavior change. Update tests' monkeypatch paths.

**Goal:** `app.py` ≤ ~250 LOC as sidebar + dispatch; each feature owns `render_*`.

**Architecture:** `ui/widgets.py`, `ui/session_corpus.py`, `ui/features/*.py`; `app.py` re-exports for compatibility and hosts `main` / `load_data` / sidebar.

**Tech Stack:** Streamlit UI modules only.

## Layout

| Module | Contents |
|--------|----------|
| `ui/widgets.py` | `sample_size_slider` |
| `ui/session_corpus.py` | `load_source`, `commit_corpus_load`, `invalidate_stale_corpus` |
| `ui/features/intro.py` | `render_intro` |
| `ui/features/snapshot.py` | `render_snapshot` |
| `ui/features/ngrams.py` | n-grams, keywords, wordcloud |
| `ui/features/textstat.py` | `render_text_stat` |
| `ui/features/ner.py` | `render_ner` |
| `ui/features/pos.py` | `render_pos` |
| `ui/features/sentiment_ui.py` | COSMUS / emotions / news |
| `ui/features/compare.py` | `render_compare_media` |
| `ui/features/topics.py` | LDA |
| `ui/features/summarization.py` | summarization |
| `ui/features/corpus_search.py` | corpus search |
| `ui/features/corpus_trends.py` | topic trends |

## Tasks

- [ ] Create modules with moved bodies
- [ ] Thin `app.py` + re-exports (`_load_source` aliases)
- [ ] Fix tests monkeypatch paths
- [ ] README tree; `pytest -m "not slow"`; ruff; commit
