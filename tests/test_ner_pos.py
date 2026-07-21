"""Tests for NER/POS batch helpers without Streamlit."""

import pytest

from nlp.ner import extract_entities_batch
from nlp.pos import extract_pos_batch, sample_pos_corpus

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def nlp():
    try:
        return spacy.load("uk_core_news_sm")
    except OSError:
        pytest.skip("uk_core_news_sm not installed")


def test_extract_entities_batch(nlp):
    texts = ["Володимир Зеленський відвідав Київ."]
    entities = extract_entities_batch(texts, "PER", nlp=nlp)
    assert len(entities) == 1
    assert isinstance(entities[0], list)


def test_extract_pos_batch(nlp):
    tags = extract_pos_batch(["Уряд ухвалив закон."], nlp=nlp)
    assert tags and tags[0]
    assert "NOUN" in tags[0] or "PROPN" in tags[0] or "VERB" in tags[0]


def test_sample_pos_corpus():
    import pandas as pd

    series = pd.Series(["a" * 100, "", "b" * 100])
    sample = sample_pos_corpus(series, max_articles=2)
    assert len(sample) <= 2
