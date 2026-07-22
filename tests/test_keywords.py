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
    keywords = extract_keywords(texts, top_n=5, lemmatize=False)
    words = [kw for kw, _ in keywords]

    assert "угода" in words or "угоду" in words or "україни" in words
    assert len(keywords) <= 5


def test_extract_keywords_pandas_series():
    series = pd.Series([
        "Уряд України ухвалив новий закон про економіку.",
        "Закон передбачає підтримку підприємців у регіонах.",
        "Новий економічний закон набув чинності.",
    ])
    keywords = extract_keywords(series, top_n=5, lemmatize=False)
    words = [kw for kw, _ in keywords]

    assert "закон" in words
    assert len(keywords) <= 5


def test_extract_keywords_lemmatize_success(monkeypatch):
    class FakeNlp:
        def pipe(self, texts, batch_size=32):
            for text in texts:
                yield text

    monkeypatch.setattr(
        "nlp.model_registry.resolve_spacy_nlp",
        lambda: FakeNlp(),
    )
    monkeypatch.setattr(
        "nlp.text_utils.lemmatize_texts",
        lambda texts, nlp: ["закон економіка"] * len(texts),
    )
    keywords = extract_keywords(
        ["Закони про економіку", "Економічний закон"],
        top_n=3,
        lemmatize=True,
    )
    words = [kw for kw, _ in keywords]
    assert "закон" in words or "економіка" in words


def test_extract_keywords_lemmatize_fallback(monkeypatch):
    monkeypatch.setattr(
        "nlp.model_registry.resolve_spacy_nlp",
        lambda: (_ for _ in ()).throw(RuntimeError("no spaCy")),
    )
    keywords = extract_keywords(
        ["Уряд ухвалив закон", "Уряд підтримав закон"],
        top_n=5,
        lemmatize=True,
    )
    words = [kw for kw, _ in keywords]
    assert "уряд" in words or "закон" in words
