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


def test_neutral_headline():
    assert classify_news_sentiment("Уряд ухвалив новий закон") == "Нейтральна"


def test_batch_and_score():
    labels = classify_news_sentiment_batch(["успіх", "атака"])
    assert labels == ["Позитивна", "Негативна"]
    assert headline_polarity_score("атака") < 0
