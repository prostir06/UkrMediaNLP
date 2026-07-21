"""Tests for Ukrainian NLP helpers."""

import pandas as pd
import pytest

from nlp.ngrams import get_top_n_words
from nlp.preprocessing import preprocess_texts as preprocess
from nlp.text_utils import load_stopwords
from nlp.textstat_ua import get_textstat_metrics

spacy = pytest.importorskip("spacy", reason="spaCy required for model tests")


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("uk_core_news_sm")
    except OSError:
        pytest.skip("uk_core_news_sm not installed")


def test_load_stopwords_contains_common_words():
    stops = load_stopwords()
    assert "і" in stops
    assert "що" in stops
    assert len(stops) > 20


def test_preprocess_strips_html():
    series = pd.Series(['<p>Тест &amp; перевірка</p>', None])
    cleaned = preprocess(series)
    assert "<p>" not in cleaned.iloc[0]
    assert "Тест" in cleaned.iloc[0]
    assert cleaned.iloc[1] == ""


def test_ngrams_on_fixed_corpus():
    corpus = pd.Series(
        [
            "Уряд України ухвалив новий закон про енергетику",
            "Уряд України обговорив новий закон про оборону",
            "Парламент ухвалив закон про енергетику країни",
        ]
    )
    unigrams = get_top_n_words(corpus, n=5)
    assert unigrams
    words = {word for word, _ in unigrams}
    assert "закон" in words or "україни" in words


def test_textstat_metrics():
    text = "Уряд ухвалив рішення. Міністр зазначив важливість реформ."
    metrics = dict(get_textstat_metrics(text))
    assert metrics["Кількість слів"] >= 5
    assert metrics["Кількість речень"] == 2
    assert 0 < metrics["Лексичне різноманіття (TTR)"] <= 1


def test_spacy_ner_finds_person(nlp):
    doc = nlp("Президент України Володимир Зеленський виступив у Києві.")
    labels = {ent.label_ for ent in doc.ents}
    assert "PER" in labels or "LOC" in labels
