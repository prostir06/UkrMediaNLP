"""Unit tests for nlp.text_utils."""

import pandas as pd
import pytest

from nlp.text_utils import (
    as_text_list,
    lemmatize_texts,
    load_stopwords,
    normalise_whitespace,
    single_token_stopwords,
)


def test_as_text_list_from_series():
    series = pd.Series(["Alpha", None, "Beta"])
    assert as_text_list(series) == ["Alpha", "", "Beta"]


def test_as_text_list_from_list():
    assert as_text_list(["a", "b"]) == ["a", "b"]


def test_as_text_list_handles_invalid_input():
    assert as_text_list(None) == []


def test_normalise_whitespace():
    assert normalise_whitespace("  текст   другий  ") == "текст другий"


def test_load_stopwords_includes_news_boilerplate():
    words = load_stopwords()
    assert "повідомляє" in words
    assert "фото" in words


def test_single_token_stopwords_excludes_phrases():
    stopwords = single_token_stopwords()
    assert all(" " not in word for word in stopwords)


def test_lemmatize_texts_with_mock_nlp():
    class Token:
        def __init__(self, lemma, is_alpha=True):
            self.lemma_ = lemma
            self.is_alpha = is_alpha

    class Doc:
        def __init__(self, tokens):
            self._tokens = tokens

        def __iter__(self):
            return iter(self._tokens)

    class MockNlp:
        def pipe(self, texts, batch_size=32):
            for text in texts:
                yield Doc([Token("уряд"), Token("ухвалити")])

    result = lemmatize_texts(["Уряд ухвалив"], MockNlp())
    assert result == ["уряд ухвалити"]


def test_lemmatize_texts_falls_back_when_pipe_missing():
    class BrokenNlp:
        pass

    texts = ["raw text"]
    assert lemmatize_texts(texts, BrokenNlp()) == texts


def test_load_stopwords_handles_missing_file(monkeypatch, tmp_path):
    import nlp.text_utils as tu

    monkeypatch.setattr(tu, "STOPWORDS_PATH", tmp_path / "missing.txt")
    tu.load_stopwords.cache_clear()
    words = tu.load_stopwords()
    assert isinstance(words, frozenset)
    tu.load_stopwords.cache_clear()
