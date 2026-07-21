"""Unit tests for summarize_articles helper."""

import pandas as pd
import pytest

from nlp.summarization import summarize_articles


class _FakeDoc:
    def __init__(self, sents):
        self.sents = sents


class _FakeSent:
    def __init__(self, text):
        self.text = text


class _FakeNlp:
    def __call__(self, text):
        parts = [p.strip() for p in text.split(".") if p.strip()]
        return _FakeDoc([_FakeSent(p + ".") for p in parts])


def test_summarize_articles_reuses_nlp():
    df = pd.DataFrame(
        [
            {
                "title": "A",
                "content": (
                    "Перше речення досить довге для фільтра. "
                    "Друге речення також достатньо довге. "
                    "Третє речення завершує короткий текст статті."
                ),
            },
            {"title": "B", "content": ""},
        ]
    )
    results = summarize_articles(df, sentence_count=2, max_articles=2, nlp=_FakeNlp())
    assert len(results) == 1
    assert results[0][0] == "A"
    assert 1 <= len(results[0][1]) <= 2


def test_summarize_articles_empty_frame():
    df = pd.DataFrame(columns=["title", "content"])
    assert summarize_articles(df, nlp=_FakeNlp()) == []
