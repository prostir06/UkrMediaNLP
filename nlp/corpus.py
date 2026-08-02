"""Multi-source corpus helpers: dates, search, trends (no Streamlit)."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import pandas as pd

from exceptions import NLPAnalysisError

logger = logging.getLogger(__name__)


def parse_published(value: object) -> pd.Timestamp:
    """Parse a date as timezone-naive UTC; return NaT on failure."""
    if value is None:
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        parsed = pd.to_datetime(text, utc=True, errors="coerce")
        return parsed.tz_localize(None)
    except (TypeError, ValueError) as exc:
        logger.debug("parse_published failed for %r: %s", value, exc)
        return pd.NaT


def ensure_published_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a timezone-naive UTC ``published_dt`` column."""
    out = df.copy()
    try:
        if "published" in out.columns:
            source = out["published"]
        elif "published_dt" in out.columns:
            source = out["published_dt"]
        else:
            out["published_dt"] = pd.NaT
            return out
        out["published_dt"] = source.map(parse_published)
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
    work = ensure_published_dt(df)
    try:
        mask_missing = work["published_dt"].isna()
        mask_ok = ~mask_missing
        if date_from is not None:
            start = parse_published(date_from).normalize()
            mask_ok &= work["published_dt"] >= start
        if date_to is not None:
            end = parse_published(date_to).normalize() + pd.Timedelta(days=1)
            mask_ok &= work["published_dt"] < end
        if include_missing:
            return work.loc[mask_ok | mask_missing].copy()
        return work.loc[mask_ok].copy()
    except Exception as exc:
        logger.warning("filter_by_date failed: %s", exc)
        return work.iloc[0:0].copy()


def cap_corpus(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep newest ``max_rows`` articles by ``published_dt``."""
    if max_rows <= 0 or df.empty:
        return df.iloc[0:0].copy()
    work = ensure_published_dt(df)
    try:
        return work.sort_values("published_dt", ascending=False, na_position="last").head(max_rows)
    except Exception as exc:
        logger.warning("cap_corpus failed: %s", exc)
        return work.head(max_rows)


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
        raise NLPAnalysisError(
            f"Пошук у корпусі не вдався: {exc}",
            step="search_corpus",
        ) from exc
    if not rows:
        return work.iloc[0:0].copy()
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["published_dt", "relevance"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


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
        return [
            re.sub(r"^\s*Тема\s+\d+\s*:\s*", "", str(label), flags=re.IGNORECASE)
            for label in (labels or [])
        ]
    except Exception as exc:
        logger.warning("suggest_lda_labels soft-failed: %s", exc)
        return []


def _to_bucket(series: pd.Series, freq: str) -> pd.Series:
    ts = pd.to_datetime(series)
    if str(freq).upper().startswith("W"):
        # Monday-start week
        return (ts - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.normalize()
    return ts.dt.normalize()


def _full_bucket_range(series: pd.Series, freq: str) -> pd.DatetimeIndex:
    step = "7D" if str(freq).upper().startswith("W") else "D"
    return pd.date_range(series.min(), series.max(), freq=step)


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
        work["bucket"] = _to_bucket(work["published_dt"], freq)
        buckets = _full_bucket_range(work["bucket"], freq)
        for term in terms:
            mask = _term_hit_mask(work, term, fields, whole_word)
            counts = work.loc[mask].groupby("bucket").size().reindex(buckets, fill_value=0)
            for bucket, count in counts.items():
                rows.append({"bucket": bucket, "term": term, "count": int(count)})
    except Exception as exc:
        logger.exception("aggregate_trends failed: %s", exc)
        raise NLPAnalysisError(
            f"Агрегація трендів не вдалася: {exc}",
            step="aggregate_trends",
        ) from exc
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
        work["bucket"] = _to_bucket(work["published_dt"], freq)
        mask = _term_hit_mask(work, term, fields, whole_word)
        buckets = _full_bucket_range(work["bucket"], freq)
        sources = pd.Index(work["source"].dropna().unique())
        full_index = pd.MultiIndex.from_product(
            [buckets, sources],
            names=["bucket", "source"],
        )
        counts = (
            work.loc[mask]
            .groupby(["bucket", "source"])
            .size()
            .reindex(full_index, fill_value=0)
            .reset_index(name="count")
        )
        return counts.sort_values(["bucket", "source"]).reset_index(drop=True)
    except Exception as exc:
        logger.exception("aggregate_trends_by_source failed: %s", exc)
        raise NLPAnalysisError(
            f"Порівняння трендів за джерелами не вдалося: {exc}",
            step="aggregate_trends_by_source",
        ) from exc
