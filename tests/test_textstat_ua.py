"""Tests for Ukrainian text statistics."""

from nlp.textstat_ua import aggregate_corpus_metrics, get_textstat_metrics


def test_textstat_metrics_empty_text():
    assert get_textstat_metrics("") == []
    assert get_textstat_metrics("   ") == []


def test_textstat_metrics_basic_counts():
    text = "Уряд ухвалив рішення. Міністр зазначив важливість реформ."
    metrics = dict(get_textstat_metrics(text))
    assert metrics["Кількість слів"] >= 5
    assert metrics["Кількість речень"] == 2
    assert 0 < metrics["Лексичне різноманіття (TTR)"] <= 1


def test_aggregate_corpus_metrics():
    texts = [
        "Уряд ухвалив закон.",
        "Парламент підтримав рішення.",
    ]
    metrics = dict(aggregate_corpus_metrics(texts))
    assert metrics["Кількість слів"] >= 4


def test_textstat_handles_non_string_gracefully():
    metrics = get_textstat_metrics(None)  # type: ignore[arg-type]
    assert metrics == []
