import pandas as pd
import pytest

from nlp.corpus import cap_corpus, ensure_published_dt, filter_by_date, parse_published


def test_parse_published_iso_and_empty():
    assert parse_published("2024-01-15T12:00:00") == pd.Timestamp("2024-01-15 12:00:00")
    assert parse_published("2024-01-15T12:00:00+00:00") == pd.Timestamp(
        "2024-01-15 12:00:00"
    )
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


def test_filter_by_date_handles_mixed_timezone_values():
    df = pd.DataFrame(
        {
            "title": ["before", "inside", "after"],
            "published": [
                "2024-01-14T23:59:59+00:00",
                "2024-01-15T12:00:00+00:00",
                "2024-01-16T00:00:00+00:00",
            ],
        }
    )

    filtered = filter_by_date(
        df,
        date_from=pd.Timestamp("2024-01-15"),
        date_to=pd.Timestamp("2024-01-15"),
    )

    assert list(filtered["title"]) == ["inside"]


def test_cap_corpus_handles_timezone_aware_values():
    df = pd.DataFrame(
        {
            "title": ["old", "new"],
            "published_dt": [
                pd.Timestamp("2024-01-15T12:00:00+00:00"),
                pd.Timestamp("2024-01-16T12:00:00+02:00"),
            ],
        }
    )

    capped = cap_corpus(df, max_rows=1)

    assert list(capped["title"]) == ["new"]


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


def test_search_corpus_unexpected_error_raises(monkeypatch):
    from exceptions import NLPAnalysisError
    from nlp import corpus as corpus_mod

    df = pd.DataFrame(
        {
            "title": ["x"],
            "content": ["y"],
            "description": [""],
            "published": ["2024-01-01"],
            "source": ["A"],
            "link": ["u"],
        }
    )

    def boom(*args, **kwargs):
        raise RuntimeError("search blew up")

    monkeypatch.setattr(corpus_mod, "row_matches", boom)

    with pytest.raises(NLPAnalysisError) as exc_info:
        corpus_mod.search_corpus(df, "x")
    assert exc_info.value.step == "search_corpus"


def test_aggregate_trends_unexpected_error_raises(monkeypatch):
    from exceptions import NLPAnalysisError
    from nlp import corpus as corpus_mod

    df = pd.DataFrame(
        {
            "title": ["футбол"],
            "content": [""],
            "published": ["2024-03-01"],
            "source": ["A"],
        }
    )

    def boom(*args, **kwargs):
        raise RuntimeError("trend blew up")

    monkeypatch.setattr(corpus_mod, "_term_hit_mask", boom)

    with pytest.raises(NLPAnalysisError) as exc_info:
        corpus_mod.aggregate_trends(df, ["футбол"], freq="D")
    assert exc_info.value.step == "aggregate_trends"

    with pytest.raises(NLPAnalysisError) as exc_info:
        corpus_mod.aggregate_trends_by_source(df, "футбол", freq="D")
    assert exc_info.value.step == "aggregate_trends_by_source"


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


def test_aggregate_trends_zero_fills_days_between_hits():
    from nlp.corpus import aggregate_trends, aggregate_trends_by_source

    df = pd.DataFrame(
        {
            "title": ["футбол", "інша тема", "футбол"],
            "content": ["", "", ""],
            "published": ["2024-03-01", "2024-03-02", "2024-03-03"],
            "source": ["A", "B", "A"],
        }
    )

    trends = aggregate_trends(df, ["футбол"], freq="D")
    assert list(trends["count"]) == [1, 0, 1]
    assert list(trends["bucket"]) == list(pd.date_range("2024-03-01", "2024-03-03"))

    by_source = aggregate_trends_by_source(df, "футбол", freq="D")
    middle = by_source[by_source["bucket"] == pd.Timestamp("2024-03-02")]
    assert set(middle["source"]) == {"A", "B"}
    assert middle["count"].eq(0).all()


def test_aggregate_trends_weekly_w_mon():
    from nlp.corpus import aggregate_trends

    # 2024-03-04 Mon, 2024-03-06 Wed — same Monday-start week
    df = pd.DataFrame(
        {
            "title": ["футбол понеділок", "футбол середа"],
            "content": ["", ""],
            "description": ["", ""],
            "published": ["2024-03-04", "2024-03-06"],
            "source": ["A", "A"],
            "link": ["u1", "u2"],
        }
    )
    trends = aggregate_trends(df, ["футбол"], freq="W-MON")
    assert len(trends) == 1
    assert trends.iloc[0]["count"] == 2
    mon = pd.Timestamp("2024-03-04").normalize()
    assert pd.Timestamp(trends.iloc[0]["bucket"]).normalize() == mon


def test_merge_source_frames():
    from nlp.corpus import merge_source_frames

    a = pd.DataFrame({"title": ["t1"], "published": ["2024-05-01"], "content": ["x"], "source": ["A"]})
    b = pd.DataFrame({"title": ["t2"], "published": ["2024-05-02"], "content": ["y"], "source": ["B"]})
    merged = merge_source_frames([a, b], max_rows=10)
    assert len(merged) == 2
    assert set(merged["source"]) == {"A", "B"}
    assert "published_dt" in merged.columns
