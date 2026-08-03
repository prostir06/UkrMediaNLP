"""Smoke tests for feature render_* screens with mocked Streamlit."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ui.features import pos, textstat, topics

ST_TARGETS = (
    "ui.features.textstat.st",
    "ui.features.topics.st",
    "ui.features.pos.st",
)


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    for target in ST_TARGETS:
        monkeypatch.setattr(target, st)
    return st


def test_render_text_stat_shows_metrics(mock_st, monkeypatch):
    monkeypatch.setattr(
        textstat,
        "get_textstat_metrics",
        lambda text: [("Слів", 5), ("Символів", 20)],
    )
    monkeypatch.setattr(
        textstat,
        "aggregate_corpus_metrics",
        lambda content: [("Середня к-сть слів", 10)],
    )
    df = pd.DataFrame({"content": ["Стаття один.", "Стаття два."]})
    textstat.render_text_stat(df)
    mock_st.subheader.assert_called()
    mock_st.markdown.assert_called()
    assert mock_st.write.call_count >= 2


def test_render_topic_modeling_delegates(mock_st, monkeypatch):
    called = {}

    def fake_display(content):
        called["n"] = len(content)

    monkeypatch.setattr(topics, "display_topic_modeling", fake_display)
    topics.render_topic_modeling(pd.Series(["a", "b", "c"]))
    mock_st.subheader.assert_called()
    assert called["n"] == 3


def test_render_pos_uses_slider(mock_st, monkeypatch):
    monkeypatch.setattr(pos, "sample_size_slider", lambda *a, **k: 2)
    monkeypatch.setattr(pos, "plot_parts_of_speech_barchart", lambda *a, **k: None)
    content = pd.Series(["Текст один", "Текст два", "Текст три"])
    pos.render_pos(content)
    mock_st.subheader.assert_called()
