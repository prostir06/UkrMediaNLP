"""Unit tests for news headline sentiment baseline."""

from nlp.news_sentiment import (
    classify_news_sentiment,
    classify_news_sentiment_batch,
    headline_polarity_score,
)


def test_positive_headline():
    assert classify_news_sentiment("Велика перемога та успіх команди") == "Позитивна"


def test_negative_headline():
    assert classify_news_sentiment("Обстріл міста та загроза катастрофи") == "Негативна"


def test_mixed_headline():
    assert classify_news_sentiment("Перемога під час війни") == "Змішана"


def test_positive_recovery_headline():
    assert classify_news_sentiment("Програма відновлення регіонів") == "Позитивна"


def test_neutral_headline():
    assert classify_news_sentiment("Уряд ухвалив новий закон") == "Нейтральна"


def test_batch_and_score():
    labels = classify_news_sentiment_batch(["успіх", "атака"])
    assert labels == ["Позитивна", "Негативна"]
    assert headline_polarity_score("атака") < 0


def test_classify_handles_none_and_empty():
    assert classify_news_sentiment(None) == "Нейтральна"  # type: ignore[arg-type]
    assert classify_news_sentiment("") == "Нейтральна"


def test_batch_handles_empty_input():
    assert classify_news_sentiment_batch(None) == []  # type: ignore[arg-type]
    assert classify_news_sentiment_batch([]) == []


def test_polarity_score_empty():
    assert headline_polarity_score("") == 0.0
    assert headline_polarity_score(None) == 0.0  # type: ignore[arg-type]


def test_word_boundary_avoids_embedded_stem():
    # Substring «війн» must not match inside an unrelated token without a boundary.
    # «відновлення» contains letters overlapping stems but should stay positive-only.
    assert classify_news_sentiment("Програма відновлення регіонів") == "Позитивна"
    # Explicit negative stem at word start still matches declined forms.
    assert classify_news_sentiment("Війна триває") == "Негативна"


def test_extended_markers():
    assert classify_news_sentiment("Нові інвестиції в регіон") == "Позитивна"
    assert classify_news_sentiment("Ракетний удар по місту") == "Негативна"
