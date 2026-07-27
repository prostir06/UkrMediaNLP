# Corpus Search & Topic Trends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two Streamlit NLP functions — «Пошук у корпусі» and «Тренди тем» — on a shared multi-media corpus within one sidebar category.

**Architecture:** Pure logic lives in `nlp/corpus.py` (parse dates, filter, search, suggestions, trend aggregation, merge). Streamlit-only controls live in `ui/corpus_controls.py`. Plotly chart builders live in `ui/corpus_charts.py`. `app.py` owns session_state, loading via existing `load_articles`, and two render handlers. No new SQLite table; reuse per-source article cache.

**Tech Stack:** Python 3.11+, pandas, plotly, spaCy (via existing `get_top_n_words` / `run_topic_modeling`), Streamlit, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-07-27-corpus-search-trends-design.md`

## Global Constraints

- Function labels exactly: `Пошук у корпусі`, `Тренди тем`
- Config defaults: `MAX_CORPUS_SOURCES=10`, `MAX_CORPUS_ARTICLES_TOTAL=300`, `MAX_TREND_TERMS=8`
- Both labels in `NLP_FUNCTIONS_FULL` and `NLP_FUNCTIONS_LIGHT`; LDA expander only when `not get_cloud_light()`
- No embeddings; no cross-category corpus; no PDF export; no second SQLite corpus store
- Unit tests: no live network; mock `load_articles` where needed
- TDD: failing test → implement → pass → commit per task
- Keep try/except on I/O and NLP soft-fails; ruff clean (`E`, `F`, `I`)

## File map

| File | Responsibility |
|------|----------------|
| `config.py` | Caps + NLP function list entries |
| `nlp/corpus.py` | Pure corpus ops |
| `ui/corpus_charts.py` | Plotly figures (no Streamlit) |
| `ui/corpus_controls.py` | Sidebar corpus widgets + load button |
| `app.py` | session_state, handlers, wiring |
| `tests/test_corpus.py` | Core corpus unit tests |
| `tests/test_corpus_suggest.py` | Suggestions unit tests |
| `tests/test_corpus_charts.py` | Chart builders |
| `tests/test_config.py` | Function names / caps |
| `tests/test_corpus_controls.py` | Controls with mocked `st` |
| `README.md` | Short feature note |

---

### Task 1: Config constants and NLP function names

**Files:**
- Modify: `config.py` (near `MAX_POS_ARTICLES` and `NLP_FUNCTIONS_*`)
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `MAX_CORPUS_SOURCES: int`, `MAX_CORPUS_ARTICLES_TOTAL: int`, `MAX_TREND_TERMS: int`; list membership for both new labels

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_corpus_config_and_functions():
    from config import (
        MAX_CORPUS_ARTICLES_TOTAL,
        MAX_CORPUS_SOURCES,
        MAX_TREND_TERMS,
        NLP_FUNCTIONS_FULL,
        NLP_FUNCTIONS_LIGHT,
    )

    assert MAX_CORPUS_SOURCES == 10
    assert MAX_CORPUS_ARTICLES_TOTAL == 300
    assert MAX_TREND_TERMS == 8
    assert "Пошук у корпусі" in NLP_FUNCTIONS_FULL
    assert "Тренди тем" in NLP_FUNCTIONS_FULL
    assert "Пошук у корпусі" in NLP_FUNCTIONS_LIGHT
    assert "Тренди тем" in NLP_FUNCTIONS_LIGHT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_corpus_config_and_functions -v`  
Expected: FAIL (ImportError or AssertionError — names/constants missing)

- [ ] **Step 3: Write minimal implementation**

In `config.py`, append after `MAX_POS_ARTICLES`:

```python
MAX_CORPUS_SOURCES = _env_int("MAX_CORPUS_SOURCES", 10)
MAX_CORPUS_ARTICLES_TOTAL = _env_int("MAX_CORPUS_ARTICLES_TOTAL", 300)
MAX_TREND_TERMS = _env_int("MAX_TREND_TERMS", 8)
```

Insert into both `NLP_FUNCTIONS_FULL` and `NLP_FUNCTIONS_LIGHT` (after `"Порівняння медіа"`):

```python
    "Пошук у корпусі",
    "Тренди тем",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_corpus_config_and_functions -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add corpus search/trends config and NLP function names"
```

---

### Task 2: Date parsing and filtering (`nlp/corpus.py`)

**Files:**
- Create: `nlp/corpus.py`
- Create: `tests/test_corpus.py`

**Interfaces:**
- Produces:
  - `parse_published(value: object) -> pd.Timestamp | pd.NaT`
  - `ensure_published_dt(df: pd.DataFrame) -> pd.DataFrame` (adds `published_dt`)
  - `filter_by_date(df, date_from, date_to, include_missing: bool = False) -> pd.DataFrame`
  - `cap_corpus(df, max_rows: int) -> pd.DataFrame` (sort by `published_dt` desc, head)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus.py
import pandas as pd

from nlp.corpus import cap_corpus, ensure_published_dt, filter_by_date, parse_published


def test_parse_published_iso_and_empty():
    assert parse_published("2024-01-15T12:00:00") == pd.Timestamp("2024-01-15 12:00:00")
    assert pd.isna(parse_published(""))
    assert pd.isna(parse_published(None))


def test_filter_by_date_and_cap():
    df = pd.DataFrame(
        {
            "title": ["a", "b", "c"],
            "published": ["2024-01-01", "2024-01-10", ""],
            "source": ["A", "A", "B"],
        }
    )
    df = ensure_published_dt(df)
    filtered = filter_by_date(
        df,
        date_from=pd.Timestamp("2024-01-05"),
        date_to=pd.Timestamp("2024-01-31"),
        include_missing=False,
    )
    assert list(filtered["title"]) == ["b"]
    with_missing = filter_by_date(
        df,
        date_from=pd.Timestamp("2024-01-05"),
        date_to=pd.Timestamp("2024-01-31"),
        include_missing=True,
    )
    assert set(with_missing["title"]) == {"b", "c"}
    capped = cap_corpus(df, max_rows=1)
    assert len(capped) == 1
    assert capped.iloc[0]["title"] == "b"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corpus.py::test_parse_published_iso_and_empty tests/test_corpus.py::test_filter_by_date_and_cap -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# nlp/corpus.py
"""Multi-source corpus helpers: dates, search, trends (no Streamlit)."""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def parse_published(value: object) -> pd.Timestamp:
    """Parse RSS/published field to Timestamp; return NaT on failure."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        return pd.to_datetime(text, utc=False, errors="coerce")
    except (TypeError, ValueError) as exc:
        logger.debug("parse_published failed for %r: %s", value, exc)
        return pd.NaT


def ensure_published_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``published_dt`` column."""
    out = df.copy()
    try:
        if "published" not in out.columns:
            out["published_dt"] = pd.NaT
            return out
        out["published_dt"] = out["published"].map(parse_published)
    except Exception as exc:
        logger.warning("ensure_published_dt failed: %s", exc)
        out["published_dt"] = pd.NaT
    return out


def filter_by_date(
    df: pd.DataFrame,
    date_from: pd.Timestamp | None,
    date_to: pd.Timestamp | None,
    include_missing: bool = False,
) -> pd.DataFrame:
    """Filter rows by ``published_dt`` inclusive day bounds."""
    if df.empty:
        return df.copy()
    work = ensure_published_dt(df) if "published_dt" not in df.columns else df.copy()
    try:
        mask_missing = work["published_dt"].isna()
        mask_ok = ~mask_missing
        if date_from is not None:
            start = pd.Timestamp(date_from).normalize()
            mask_ok &= work["published_dt"] >= start
        if date_to is not None:
            end = pd.Timestamp(date_to).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask_ok &= work["published_dt"] <= end
        if include_missing:
            return work.loc[mask_ok | mask_missing].copy()
        return work.loc[mask_ok].copy()
    except Exception as exc:
        logger.warning("filter_by_date failed: %s", exc)
        return work.copy()


def cap_corpus(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep newest ``max_rows`` articles by ``published_dt``."""
    if max_rows <= 0 or df.empty:
        return df.iloc[0:0].copy()
    work = ensure_published_dt(df) if "published_dt" not in df.columns else df.copy()
    try:
        return work.sort_values("published_dt", ascending=False, na_position="last").head(max_rows)
    except Exception as exc:
        logger.warning("cap_corpus failed: %s", exc)
        return work.head(max_rows)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corpus.py::test_parse_published_iso_and_empty tests/test_corpus.py::test_filter_by_date_and_cap -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlp/corpus.py tests/test_corpus.py
git commit -m "feat: add corpus date parse/filter helpers"
```

---

### Task 3: Search and snippets

**Files:**
- Modify: `nlp/corpus.py`
- Modify: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `ensure_published_dt`
- Produces:
  - `article_search_text(row, fields: Iterable[str]) -> str`
  - `make_snippet(text: str, query: str, width: int = 120) -> str`
  - `row_matches(text: str, query: str, whole_word: bool, use_lemmas: bool = False) -> bool`
  - `search_corpus(df, query, fields=("title","content"), whole_word=False, use_lemmas=False) -> pd.DataFrame`  
    Output columns include at least: original cols + `snippet`, `relevance` (int). Sorted by `published_dt` desc then `relevance` desc.

- [ ] **Step 1: Write the failing tests**

```python
def test_search_corpus_phrase_and_whole_word():
    from nlp.corpus import search_corpus

    df = pd.DataFrame(
        {
            "title": ["Перемога збірної України", "Економіка зростає"],
            "content": ["матч завершився перемогою", "ринок акцій"],
            "description": ["", ""],
            "published": ["2024-02-01", "2024-02-02"],
            "source": ["A", "B"],
            "link": ["https://a", "https://b"],
        }
    )
    hits = search_corpus(df, "перемога", fields=("title", "content"), whole_word=False)
    assert len(hits) == 1
    assert "перемог" in hits.iloc[0]["snippet"].lower() or "Перемог" in hits.iloc[0]["title"]

    whole = search_corpus(df, "зро", fields=("title",), whole_word=True)
    assert len(whole) == 0
    phrase = search_corpus(df, "збірної України", fields=("title",), whole_word=False)
    assert len(phrase) == 1


def test_search_empty_query_returns_empty():
    from nlp.corpus import search_corpus

    df = pd.DataFrame({"title": ["a"], "content": ["b"], "description": [""], "published": ["2024-01-01"], "source": ["A"], "link": ["u"]})
    assert search_corpus(df, "   ").empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corpus.py::test_search_corpus_phrase_and_whole_word tests/test_corpus.py::test_search_empty_query_returns_empty -v`  
Expected: FAIL (functions missing)

- [ ] **Step 3: Write minimal implementation**

Append to `nlp/corpus.py`:

```python
def article_search_text(row: pd.Series, fields: Iterable[str]) -> str:
    parts: list[str] = []
    for field in fields:
        try:
            value = row.get(field, "")
        except Exception:
            value = ""
        text = str(value or "").strip()
        if not text and field == "content":
            text = str(row.get("description", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def make_snippet(text: str, query: str, width: int = 120) -> str:
    lowered = text.lower()
    q = query.lower().strip()
    if not q:
        return text[:width]
    idx = lowered.find(q)
    if idx < 0:
        return text[:width]
    start = max(0, idx - width // 3)
    end = min(len(text), idx + len(q) + width // 2)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def row_matches(text: str, query: str, whole_word: bool, use_lemmas: bool = False) -> bool:
    q = query.strip()
    if not q:
        return False
    hay = text.lower()
    needle = q.lower()
    if use_lemmas:
        try:
            from nlp.model_registry import resolve_spacy_nlp
            from nlp.text_utils import lemmatize_texts

            nlp = resolve_spacy_nlp()
            hay = lemmatize_texts([text], nlp)[0].lower()
            needle = lemmatize_texts([q], nlp)[0].lower()
        except Exception as exc:
            logger.debug("lemma match fallback: %s", exc)
    if whole_word:
        pattern = re.compile(rf"(?iu)\b{re.escape(needle)}\b")
        return pattern.search(hay) is not None
    return needle in hay


def search_corpus(
    df: pd.DataFrame,
    query: str,
    fields: Iterable[str] = ("title", "content"),
    whole_word: bool = False,
    use_lemmas: bool = False,
) -> pd.DataFrame:
    """Return matching rows with ``snippet`` and ``relevance``."""
    q = (query or "").strip()
    if not q or df.empty:
        return df.iloc[0:0].copy()
    work = ensure_published_dt(df)
    field_list = tuple(fields) or ("title", "content")
    rows = []
    try:
        for _, row in work.iterrows():
            title = str(row.get("title", "") or "")
            blob = article_search_text(row, field_list)
            if not row_matches(blob, q, whole_word=whole_word, use_lemmas=use_lemmas):
                continue
            title_hit = 2 if row_matches(title, q, whole_word=whole_word, use_lemmas=use_lemmas) else 0
            content_hit = 1 if row_matches(blob, q, whole_word=whole_word, use_lemmas=use_lemmas) else 0
            relevance = title_hit + content_hit
            item = row.to_dict()
            item["snippet"] = make_snippet(blob, q)
            item["relevance"] = relevance
            rows.append(item)
    except Exception as exc:
        logger.exception("search_corpus failed: %s", exc)
        return work.iloc[0:0].copy()
    if not rows:
        return work.iloc[0:0].copy()
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["published_dt", "relevance"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corpus.py -k search -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlp/corpus.py tests/test_corpus.py
git commit -m "feat: add corpus keyword search with snippets"
```

---

### Task 4: Term suggestions and trend aggregation

**Files:**
- Modify: `nlp/corpus.py`
- Create: `tests/test_corpus_suggest.py`
- Modify: `tests/test_corpus.py`

**Interfaces:**
- Produces:
  - `suggest_terms(df, n: int = 15) -> list[str]` — uses `get_top_n_words` on titles (+ first 200 chars of content sample)
  - `suggest_lda_labels(df, number_topics: int = 5) -> list[str]` — wraps `run_topic_modeling`; returns `[]` on soft fail
  - `parse_manual_terms(text: str) -> list[str]` — non-empty unique lines, strip, max `MAX_TREND_TERMS` later at UI
  - `aggregate_trends(df, terms: list[str], freq: str = "D", fields=("title","content"), whole_word=False) -> pd.DataFrame` columns `bucket`, `term`, `count`
  - `aggregate_trends_by_source(df, term: str, freq: str = "D", ...) -> pd.DataFrame` columns `bucket`, `source`, `count`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corpus_suggest.py
import pandas as pd

from nlp.corpus import parse_manual_terms, suggest_terms


def test_parse_manual_terms():
    assert parse_manual_terms(" футбол \n\nЗбірна\nфутбол ") == ["футбол", "Збірна"]


def test_suggest_terms_from_titles(monkeypatch):
    monkeypatch.setattr(
        "nlp.corpus.get_top_n_words",
        lambda corpus, n=10: [("футбол", 3), ("матч", 2)][:n],
    )
    df = pd.DataFrame({"title": ["a", "b"], "content": ["c", "d"]})
    assert suggest_terms(df, n=2) == ["футбол", "матч"]
```

```python
# append to tests/test_corpus.py
def test_aggregate_trends_day_and_by_source():
    from nlp.corpus import aggregate_trends, aggregate_trends_by_source

    df = pd.DataFrame(
        {
            "title": ["футбол сьогодні", "футбол вчора", "теніс"],
            "content": ["гра", "гра", "сет"],
            "description": ["", "", ""],
            "published": ["2024-03-01", "2024-03-02", "2024-03-02"],
            "source": ["A", "B", "A"],
            "link": ["u1", "u2", "u3"],
        }
    )
    trends = aggregate_trends(df, ["футбол"], freq="D")
    assert trends["count"].sum() == 2
    assert set(trends.columns) >= {"bucket", "term", "count"}
    by_src = aggregate_trends_by_source(df, "футбол", freq="D")
    assert set(by_src.columns) >= {"bucket", "source", "count"}
    assert by_src["count"].sum() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_corpus_suggest.py tests/test_corpus.py::test_aggregate_trends_day_and_by_source -v`  
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

Append to `nlp/corpus.py` (import `get_top_n_words` lazily inside `suggest_terms` to keep module light):

```python
def parse_manual_terms(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in (text or "").splitlines():
        term = line.strip()
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out


def suggest_terms(df: pd.DataFrame, n: int = 15) -> list[str]:
    if df is None or df.empty or n <= 0:
        return []
    try:
        from nlp.ngrams import get_top_n_words
        from nlp.preprocessing import preprocess_texts

        titles = preprocess_texts(df.get("title", pd.Series(dtype=str)))
        contents = df.get("content", pd.Series(dtype=str)).fillna("").astype(str).str.slice(0, 200)
        corpus = (titles.fillna("") + " " + contents).tolist()
        return [word for word, _count in get_top_n_words(corpus, n=n)]
    except Exception as exc:
        logger.warning("suggest_terms failed: %s", exc)
        return []


def suggest_lda_labels(df: pd.DataFrame, number_topics: int = 5) -> list[str]:
    if df is None or df.empty:
        return []
    try:
        from nlp.topics import run_topic_modeling

        content = df.get("content", pd.Series(dtype=str)).fillna("").astype(str)
        labels = run_topic_modeling(content, number_topics=number_topics, number_words=5)
        return list(labels or [])
    except Exception as exc:
        logger.warning("suggest_lda_labels soft-failed: %s", exc)
        return []


def _term_hit_mask(
    df: pd.DataFrame,
    term: str,
    fields: Iterable[str],
    whole_word: bool,
) -> pd.Series:
    field_list = tuple(fields) or ("title", "content")

    def _hit(row: pd.Series) -> bool:
        return row_matches(article_search_text(row, field_list), term, whole_word=whole_word)

    return df.apply(_hit, axis=1)


def aggregate_trends(
    df: pd.DataFrame,
    terms: list[str],
    freq: str = "D",
    fields: Iterable[str] = ("title", "content"),
    whole_word: bool = False,
) -> pd.DataFrame:
    work = ensure_published_dt(df)
    work = work.dropna(subset=["published_dt"])
    if work.empty or not terms:
        return pd.DataFrame(columns=["bucket", "term", "count"])
    rows: list[dict] = []
    try:
        work = work.copy()
        work["bucket"] = work["published_dt"].dt.to_period("W" if freq.upper().startswith("W") else "D").dt.start_time
        for term in terms:
            mask = _term_hit_mask(work, term, fields, whole_word)
            counts = work.loc[mask].groupby("bucket").size()
            for bucket, count in counts.items():
                rows.append({"bucket": bucket, "term": term, "count": int(count)})
    except Exception as exc:
        logger.exception("aggregate_trends failed: %s", exc)
        return pd.DataFrame(columns=["bucket", "term", "count"])
    return pd.DataFrame(rows).sort_values(["bucket", "term"]).reset_index(drop=True)


def aggregate_trends_by_source(
    df: pd.DataFrame,
    term: str,
    freq: str = "D",
    fields: Iterable[str] = ("title", "content"),
    whole_word: bool = False,
) -> pd.DataFrame:
    work = ensure_published_dt(df)
    work = work.dropna(subset=["published_dt"])
    if work.empty or not str(term).strip():
        return pd.DataFrame(columns=["bucket", "source", "count"])
    try:
        work = work.copy()
        work["bucket"] = work["published_dt"].dt.to_period("W" if freq.upper().startswith("W") else "D").dt.start_time
        mask = _term_hit_mask(work, term, fields, whole_word)
        counts = work.loc[mask].groupby(["bucket", "source"]).size().reset_index(name="count")
        return counts.sort_values(["bucket", "source"]).reset_index(drop=True)
    except Exception as exc:
        logger.exception("aggregate_trends_by_source failed: %s", exc)
        return pd.DataFrame(columns=["bucket", "source", "count"])
```

Note: for weekly buckets use period `"W-MON"` when `freq` is `"W-MON"`:

```python
period = "W-MON" if str(freq).upper().startswith("W") else "D"
work["bucket"] = work["published_dt"].dt.to_period(period).dt.start_time
```

Use the same in both aggregate functions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_corpus_suggest.py tests/test_corpus.py::test_aggregate_trends_day_and_by_source -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlp/corpus.py tests/test_corpus.py tests/test_corpus_suggest.py
git commit -m "feat: add corpus term suggestions and trend aggregation"
```

---

### Task 5: Merge multi-source frames (pure)

**Files:**
- Modify: `nlp/corpus.py`
- Modify: `tests/test_corpus.py`

**Interfaces:**
- Produces: `merge_source_frames(frames: list[pd.DataFrame], max_rows: int) -> pd.DataFrame`  
  Ensures `source` column exists, concat, `ensure_published_dt`, `cap_corpus`.

- [ ] **Step 1: Write the failing test**

```python
def test_merge_source_frames():
    from nlp.corpus import merge_source_frames

    a = pd.DataFrame({"title": ["t1"], "published": ["2024-05-01"], "content": ["x"], "source": ["A"]})
    b = pd.DataFrame({"title": ["t2"], "published": ["2024-05-02"], "content": ["y"], "source": ["B"]})
    merged = merge_source_frames([a, b], max_rows=10)
    assert len(merged) == 2
    assert set(merged["source"]) == {"A", "B"}
    assert "published_dt" in merged.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus.py::test_merge_source_frames -v`  
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def merge_source_frames(frames: list[pd.DataFrame], max_rows: int) -> pd.DataFrame:
    clean: list[pd.DataFrame] = []
    for frame in frames:
        if frame is None or getattr(frame, "empty", True):
            continue
        try:
            part = frame.copy()
            if "source" not in part.columns:
                part["source"] = ""
            clean.append(part)
        except Exception as exc:
            logger.warning("skip bad frame: %s", exc)
    if not clean:
        return pd.DataFrame()
    try:
        merged = pd.concat(clean, ignore_index=True)
    except Exception as exc:
        logger.exception("concat failed: %s", exc)
        return pd.DataFrame()
    return cap_corpus(ensure_published_dt(merged), max_rows=max_rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus.py::test_merge_source_frames -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add nlp/corpus.py tests/test_corpus.py
git commit -m "feat: merge multi-source corpus frames with cap"
```

---

### Task 6: Plotly chart builders

**Files:**
- Create: `ui/corpus_charts.py`
- Create: `tests/test_corpus_charts.py`

**Interfaces:**
- Produces:
  - `build_source_hit_bar(counts: pd.Series) -> go.Figure | None`
  - `build_trends_line(trends: pd.DataFrame) -> go.Figure | None` (expects `bucket`,`term`,`count`)
  - `build_source_trends_line(trends: pd.DataFrame) -> go.Figure | None` (expects `bucket`,`source`,`count`)

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from ui.corpus_charts import build_source_hit_bar, build_source_trends_line, build_trends_line


def test_build_charts_empty_and_ok():
    assert build_source_hit_bar(pd.Series(dtype=int)) is None
    assert build_trends_line(pd.DataFrame(columns=["bucket", "term", "count"])) is None
    fig = build_trends_line(
        pd.DataFrame(
            {
                "bucket": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "term": ["футбол", "футбол"],
                "count": [1, 2],
            }
        )
    )
    assert fig is not None
    fig2 = build_source_trends_line(
        pd.DataFrame(
            {
                "bucket": [pd.Timestamp("2024-01-01")],
                "source": ["A"],
                "count": [3],
            }
        )
    )
    assert fig2 is not None
    bar = build_source_hit_bar(pd.Series({"A": 2, "B": 1}))
    assert bar is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus_charts.py -v`  
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# ui/corpus_charts.py
"""Plotly charts for corpus search and trends (no Streamlit)."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)


def build_source_hit_bar(counts: pd.Series):
    try:
        if counts is None or len(counts) == 0:
            return None
        frame = counts.rename("N").reset_index()
        frame.columns = ["Медіа", "N"]
        return px.bar(frame, x="Медіа", y="N", title="Знахідки за медіа")
    except Exception as exc:
        logger.warning("build_source_hit_bar failed: %s", exc)
        return None


def build_trends_line(trends: pd.DataFrame):
    try:
        if trends is None or trends.empty:
            return None
        return px.line(
            trends,
            x="bucket",
            y="count",
            color="term",
            markers=True,
            title="Тренди тем",
            labels={"bucket": "Дата", "count": "Статей", "term": "Тема"},
        )
    except Exception as exc:
        logger.warning("build_trends_line failed: %s", exc)
        return None


def build_source_trends_line(trends: pd.DataFrame):
    try:
        if trends is None or trends.empty:
            return None
        return px.line(
            trends,
            x="bucket",
            y="count",
            color="source",
            markers=True,
            title="Порівняння медіа (одна тема)",
            labels={"bucket": "Дата", "count": "Статей", "source": "Медіа"},
        )
    except Exception as exc:
        logger.warning("build_source_trends_line failed: %s", exc)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus_charts.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/corpus_charts.py tests/test_corpus_charts.py
git commit -m "feat: add Plotly chart builders for corpus trends"
```

---

### Task 7: Corpus sidebar controls + load into session_state

**Files:**
- Create: `ui/corpus_controls.py`
- Create: `tests/test_corpus_controls.py`
- Modify: `app.py` (helpers used by later tasks; may add `_ensure_corpus_loaded` here)

**Interfaces:**
- Produces:
  - `CORPUS_FUNCTIONS = frozenset({"Пошук у корпусі", "Тренди тем"})`
  - `render_corpus_sidebar(category: str, show_lda_toggle: bool = False) -> dict` with keys:  
    `sources: list[str]`, `all_category: bool`, `date_from`, `date_to`, `include_missing: bool`, `load_clicked: bool`
  - `load_corpus_into_session(sources: list[str], date_from, date_to, include_missing: bool, category: str, load_articles_fn, progress_callback=None) -> tuple[pd.DataFrame, list[str]]`  
    Returns `(df, warnings)`. On total failure returns `(empty or previous kept by caller, errors)`.

Pure load orchestration (testable without Streamlit):

```python
def build_corpus_from_sources(
    sources: list[str],
    load_articles_fn,
    max_sources: int,
    max_rows: int,
    date_from,
    date_to,
    include_missing: bool,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[str]]:
    """Load each source; collect warnings; merge+filter+cap."""
```

- [ ] **Step 1: Write the failing test for build_corpus_from_sources**

```python
# tests/test_corpus_controls.py
import pandas as pd

from ui.corpus_controls import build_corpus_from_sources


def test_build_corpus_partial_failure():
    def fake_load(name, progress_callback=None):
        if name == "Bad":
            raise RuntimeError("boom")
        return pd.DataFrame(
            {
                "title": [f"{name}-t"],
                "published": ["2024-06-01"],
                "content": ["body"],
                "source": [name],
                "link": ["u"],
                "description": [""],
            }
        )

    df, warnings = build_corpus_from_sources(
        ["Good", "Bad"],
        load_articles_fn=fake_load,
        max_sources=10,
        max_rows=50,
        date_from=None,
        date_to=None,
        include_missing=True,
    )
    assert len(df) == 1
    assert df.iloc[0]["source"] == "Good"
    assert any("Bad" in w for w in warnings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus_controls.py::test_build_corpus_partial_failure -v`  
Expected: FAIL

- [ ] **Step 3: Implement `ui/corpus_controls.py`**

Implement `build_corpus_from_sources` using `merge_source_frames`, `filter_by_date`, caps from config. Cap source list to `max_sources`. Catch per-source exceptions into `warnings`. If all fail, return empty DF + warnings.

Also implement `render_corpus_sidebar` with Streamlit widgets (multiselect, checkbox «Вся категорія», dates, include_missing, button). Keep widget keys stable: `corpus_all_category`, `corpus_sources_ms`, `corpus_date_from`, `corpus_date_to`, `corpus_include_missing`, `corpus_load_btn`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus_controls.py::test_build_corpus_partial_failure -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/corpus_controls.py tests/test_corpus_controls.py
git commit -m "feat: add corpus load orchestration and sidebar controls"
```

---

### Task 8: Wire «Пошук у корпусі» in `app.py`

**Files:**
- Modify: `app.py` (`main` / `load_data` / new `render_corpus_search`)
- Modify: `tests/test_app_helpers.py` (optional smoke with mocked st)

**Interfaces:**
- Consumes: `render_corpus_sidebar`, `build_corpus_from_sources`, `search_corpus`, `build_source_hit_bar`
- Session keys: `corpus_df`, `corpus_sources`, `corpus_category`, `corpus_loaded_at`, `corpus_date_from`, `corpus_date_to`

- [ ] **Step 1: Write failing helper test (optional but preferred)**

```python
def test_render_corpus_search_requires_corpus(mock_st, monkeypatch):
    from app import render_corpus_search

    monkeypatch.setattr("app.st.session_state", {}, raising=False)
    # Ensure empty corpus path shows info
    class SS(dict):
        pass
    ss = SS()
    mock_st.session_state = ss
    monkeypatch.setattr("app.st", mock_st)
    render_corpus_search()
    mock_st.info.assert_called()
```

If `session_state` mocking is awkward, skip this step and rely on manual verification in Step 4; still implement the handler.

- [ ] **Step 2: Implement wiring**

1. In `main()`, after selecting function: if function in `CORPUS_FUNCTIONS`, call `render_corpus_sidebar(selected_category, show_lda_toggle=False)`. On `load_clicked`, call `build_corpus_from_sources` with `load_articles`, store DF in `st.session_state`, show warnings.
2. In `load_data` / handler map, add:

```python
"Пошук у корпусі": lambda: render_corpus_search(),
```

Note: these two functions do **not** use single-source `_load_source`; they early-return in `load_data` like «Вступ» / «Порівняння медіа».

3. `render_corpus_search()`:
   - If no `corpus_df` → `st.info("Спочатку завантажте корпус…")` return
   - Inputs: query, fields radio, whole_word, use_lemmas
   - Call `search_corpus`
   - Show metrics, Plotly bar, dataframe with title/link/date/source/snippet

- [ ] **Step 3: Run unit suite for corpus + config**

Run: `pytest tests/test_corpus.py tests/test_corpus_suggest.py tests/test_corpus_charts.py tests/test_corpus_controls.py tests/test_config.py -q`  
Expected: all PASS

- [ ] **Step 4: Manual smoke (local)**

Run: `streamlit run streamlit_app.py`  
Pick category → «Пошук у корпусі» → select 2 media → load → search a known word → see table.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_helpers.py
git commit -m "feat: wire corpus search UI and session corpus load"
```

---

### Task 9: Wire «Тренди тем» in `app.py`

**Files:**
- Modify: `app.py`
- Modify: `ui/corpus_controls.py` only if LDA expander needs a flag from caller

**Interfaces:**
- Consumes: `suggest_terms`, `suggest_lda_labels`, `parse_manual_terms`, `aggregate_trends`, `aggregate_trends_by_source`, chart builders
- Produces: `render_topic_trends()`

- [ ] **Step 1: Implement `render_topic_trends`**

Behavior:
1. Require `st.session_state["corpus_df"]` non-empty.
2. Auto suggestions = `suggest_terms(df, 15)`.
3. If `not get_cloud_light()`: expander «Поглиблено (LDA)» → `suggest_lda_labels`; on empty/fail `st.warning`.
4. Manual `st.text_area` → `parse_manual_terms`.
5. Combined candidates → multiselect (max `MAX_TREND_TERMS`).
6. Freq radio: День / Тиждень (`D` / `W-MON`).
7. Plot `build_trends_line(aggregate_trends(...))`.
8. Select one term for media compare → `build_source_trends_line(aggregate_trends_by_source(...))`.

Add handler:

```python
"Тренди тем": lambda: render_topic_trends(),
```

Ensure corpus sidebar also renders for this function (same as Task 8).

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_corpus.py tests/test_corpus_suggest.py tests/test_corpus_charts.py -q`  
Expected: PASS

- [ ] **Step 3: Manual smoke**

Load corpus → «Тренди тем» → pick 2–3 suggested terms → see line chart → pick one term for media compare.

- [ ] **Step 4: Commit**

```bash
git add app.py ui/corpus_controls.py
git commit -m "feat: wire topic trends UI with hybrid term selection"
```

---

### Task 10: README + full verification

**Files:**
- Modify: `README.md` (Можливості + short subsection)

- [ ] **Step 1: Update README**

Add bullets under Можливості:

```markdown
- **Пошук у корпусі** — ключові слова/фрази по кількох медіа категорії
- **Тренди тем** — гібридні теми (авто + ручні) і порівняння медіа на графіку
```

- [ ] **Step 2: Full test + lint**

Run:

```bash
pytest -m "not slow" -q
ruff check nlp/corpus.py ui/corpus_charts.py ui/corpus_controls.py app.py config.py
```

Expected: all tests PASS; ruff clean.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document corpus search and topic trends"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Shared corpus + session_state | 7–8 |
| Multiselect + «вся категорія» | 7 |
| Explicit load button + partial failure | 7 |
| Date filter + include missing + caps | 2, 5, 7 |
| Search UI + snippet + bar by media | 3, 6, 8 |
| Hybrid terms + LDA soft path (full only) | 4, 9 |
| Trends line + media compare chart | 4, 6, 9 |
| Both NLP_FUNCTIONS lists | 1 |
| Unit tests, no network | 2–7 |
| README | 10 |

## Self-review notes

- No TBD/placeholder steps left.
- Names consistent: `aggregate_trends` / `aggregate_trends_by_source` / `build_corpus_from_sources`.
- Weekly freq standardized as `W-MON` in UI and aggregators.
- Lemma search is optional and soft-fails to substring match.
