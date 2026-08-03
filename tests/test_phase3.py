"""Tests for Phase 3: LDA, summarization, sentiment helpers."""

import pandas as pd
import pytest

from nlp.sentiment_constants import _label_to_ua, _truncate
from nlp.summarization import lexrank_summarize, split_sentences
from nlp.topics import run_topic_modeling

spacy = pytest.importorskip("spacy", reason="spaCy required")


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("uk_core_news_sm")
    except OSError:
        pytest.skip("uk_core_news_sm not installed")


CORPUS = pd.Series(
    [
        "Уряд України ухвалив новий закон про енергетичну безпеку країни.",
        "Міністр енергетики представив план модернізації електромереж України.",
        "Парламент обговорив законопроект про відновлення інфраструктури енергетики.",
        "Експерти оцінили вплив енергетичних реформ на економіку України.",
        "Комісія схвалила фінансування проєктів відновлюваної енергетики.",
    ]
)


def test_truncate_limits_length():
    long_text = "а" * 5000
    assert len(_truncate(long_text)) == 2000


def test_label_to_ua_maps_known_labels():
    assert _label_to_ua("positive") == "Позитивна"
    assert _label_to_ua("negative") == "Негативна"
    assert _label_to_ua("LABEL_3") == "Позитивна"
    assert _label_to_ua("LABEL_2") == "Нейтральна"


def test_lda_returns_topics():
    topics = run_topic_modeling(CORPUS, number_topics=3, number_words=4)
    assert topics
    assert all(topic.startswith("Тема") for topic in topics)


def test_lda_empty_corpus():
    assert run_topic_modeling(pd.Series([])) == []


def test_split_sentences(nlp, monkeypatch):
    monkeypatch.setattr("nlp.summarization.resolve_spacy_nlp", lambda: nlp)
    text = "Уряд ухвалив рішення. Міністр зазначив важливість реформ."
    sentences = split_sentences(text, min_length=10)
    assert len(sentences) >= 2


def test_lexrank_summarize(nlp, monkeypatch):
    monkeypatch.setattr("nlp.summarization.resolve_spacy_nlp", lambda: nlp)
    text = (
        "Уряд України ухвалив важливе рішення щодо енергетики. "
        "Міністр енергетики пояснив деталі нової програми. "
        "Парламент підтримав законопроект більшістю голосів. "
        "Експерти позитивно оцінили запропоновані зміни."
    )
    summary = lexrank_summarize(text, sentence_count=2)
    assert 1 <= len(summary) <= 2


@pytest.mark.slow
def test_cosmus_sentiment_live():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    del transformers, torch

    from nlp.sentiment import classify_sentiment_cosmus

    label = classify_sentiment_cosmus("Чудові новини для України!")
    assert label in {"Позитивна", "Негативна", "Нейтральна", "Змішана"}
