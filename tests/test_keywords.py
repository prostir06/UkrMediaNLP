"""Unit tests for keyword extraction module."""

import pandas as pd

from nlp.keywords import extract_keywords


def test_extract_keywords_empty():
    assert extract_keywords([]) == []
    assert extract_keywords(["", None]) == []


def test_extract_keywords_basic():
    texts = [
        "Україна та Європейський Союз підписали нову угоду про співпрацю.",
        "Президент України відвідав саміт у Брюсселі.",
        "Нова економічна угода допоможе розвитку України.",
    ]
    keywords = extract_keywords(texts, top_n=5)
    words = [kw for kw, _ in keywords]

    assert "угода" in words or "угоду" in words or "україни" in words
    assert len(keywords) <= 5


def test_extract_keywords_pandas_series():
    series = pd.Series([
        "Уряд України ухвалив новий закон про економіку.",
        "Закон передбачає підтримку підприємців у регіонах.",
        "Новий економічний закон набув чинності.",
    ])
    keywords = extract_keywords(series, top_n=5)
    words = [kw for kw, _ in keywords]

    assert "закон" in words
    assert len(keywords) <= 5
