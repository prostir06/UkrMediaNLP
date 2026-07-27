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
