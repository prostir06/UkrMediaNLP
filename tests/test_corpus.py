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
