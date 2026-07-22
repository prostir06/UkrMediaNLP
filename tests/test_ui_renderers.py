"""Unit tests for ui.renderers with mocked Streamlit."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from exceptions import NLPAnalysisError


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    container = MagicMock()
    container.__enter__ = MagicMock(return_value=container)
    container.__exit__ = MagicMock(return_value=False)
    st.container.return_value = container
    st.tabs.return_value = [container]
    monkeypatch.setattr("ui.renderers.st", st)
    return st


def test_plot_sentiment_shows_warning_when_empty(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr(
        "ui.charts.build_sentiment_figure",
        lambda texts, method="cosmus": None,
    )
    renderers.plot_sentiment_barchart(["a"], method="news_rules")
    mock_st.warning.assert_called()


def test_plot_sentiment_shows_nlp_error(mock_st, monkeypatch):
    from ui import renderers

    def boom(*args, **kwargs):
        raise NLPAnalysisError("fail", step="cosmus_load")

    monkeypatch.setattr("ui.charts.build_sentiment_figure", boom)
    renderers.plot_sentiment_barchart(["a"])
    mock_st.error.assert_called()


def test_display_topic_modeling_empty(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr("nlp.topics.run_topic_modeling", lambda *a, **k: [])
    renderers.display_topic_modeling(pd.Series(["x", "y", "z"]))
    mock_st.warning.assert_called()


def test_display_topic_modeling_lists_topics(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr(
        "nlp.topics.run_topic_modeling",
        lambda *a, **k: ["Тема 1: уряд закон"],
    )
    renderers.display_topic_modeling(pd.Series(["a"] * 5))
    mock_st.markdown.assert_called()
    mock_st.write.assert_called()


def test_run_text_summarization_no_results(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr("nlp.model_registry.resolve_spacy_nlp", lambda: object())
    monkeypatch.setattr("nlp.summarization.summarize_articles", lambda *a, **k: [])
    renderers.run_text_summarization(pd.DataFrame({"title": [], "content": []}))
    mock_st.warning.assert_called()


def test_render_wordclouds_empty(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr(
        "nlp.wordcloud_render.build_wordcloud_images",
        lambda titles, styles=None: [],
    )
    renderers.render_wordclouds(pd.Series([""]))
    mock_st.warning.assert_called()


def test_plot_ner_handles_empty(mock_st, monkeypatch):
    from ui import renderers

    monkeypatch.setattr(
        "ui.charts.build_ner_figure",
        lambda texts, entity="PER": (None, "Особа"),
    )
    renderers.plot_most_common_named_entity_barchart(["тест"])
    mock_st.write.assert_called()
