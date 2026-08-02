"""Unit tests for heavy-NLP UI session_state flow (emotions / COSMUS)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

ST_TARGETS = (
    "app.st",
    "ui.widgets.st",
    "ui.features.sentiment_ui.st",
)


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    cols = [MagicMock(), MagicMock()]
    st.columns.return_value = cols
    status = MagicMock()
    status.__enter__ = MagicMock(return_value=status)
    status.__exit__ = MagicMock(return_value=False)
    st.status.return_value = status
    st.button.return_value = False
    st.slider.return_value = 5
    st.session_state = {}
    for target in ST_TARGETS:
        monkeypatch.setattr(target, st)
    return st


def test_render_emotions_waits_for_button(mock_st, monkeypatch):
    from app import render_sentiment_emotions

    monkeypatch.setattr(
        "ui.features.sentiment_ui.sample_size_slider",
        lambda *a, **k: 5,
    )
    titles = pd.Series(["Заголовок один", "Заголовок два"])
    render_sentiment_emotions(titles)
    mock_st.caption.assert_called()
    assert "Натисніть кнопку" in str(mock_st.caption.call_args)


def test_render_emotions_runs_when_pending(mock_st, monkeypatch):
    from app import render_sentiment_emotions

    mock_st.session_state = {
        "emotions_pending": True,
        "emotions_sample_n": 2,
    }
    monkeypatch.setattr(
        "ui.features.sentiment_ui.sample_size_slider",
        lambda *a, **k: 10,
    )
    called = {"n": 0}

    def fake_plot(sample):
        called["n"] += 1
        assert len(sample) == 2

    monkeypatch.setattr("ui.features.sentiment_ui.plot_emotion_distribution", fake_plot)
    titles = pd.Series(["a", "b", "c", "d"])
    render_sentiment_emotions(titles)
    assert called["n"] == 1
    assert mock_st.session_state["emotions_done"] is True
    assert mock_st.session_state["emotions_pending"] is False


def test_render_emotions_reset_clears_state(mock_st, monkeypatch):
    from app import render_sentiment_emotions

    monkeypatch.setattr(
        "ui.features.sentiment_ui.sample_size_slider",
        lambda *a, **k: 3,
    )

    mock_st.button.side_effect = [False, True]
    mock_st.session_state = {
        "emotions_pending": True,
        "emotions_done": True,
        "emotions_sample_n": 3,
    }

    monkeypatch.setattr(
        "ui.features.sentiment_ui.plot_emotion_distribution",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not plot")),
    )
    render_sentiment_emotions(pd.Series(["x", "y", "z"]))
    assert "emotions_pending" not in mock_st.session_state
    assert "emotions_done" not in mock_st.session_state


def test_render_emotions_handles_plot_error(mock_st, monkeypatch):
    from app import render_sentiment_emotions

    mock_st.session_state = {"emotions_pending": True, "emotions_sample_n": 1}
    monkeypatch.setattr(
        "ui.features.sentiment_ui.sample_size_slider",
        lambda *a, **k: 1,
    )
    monkeypatch.setattr(
        "ui.features.sentiment_ui.plot_emotion_distribution",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    render_sentiment_emotions(pd.Series(["заголовок"]))
    mock_st.error.assert_called()
    assert mock_st.session_state["emotions_pending"] is False


def test_render_cosmus_uses_stored_sample_n(mock_st, monkeypatch):
    from app import render_sentiment_cosmus

    mock_st.session_state = {
        "cosmus_pending": True,
        "cosmus_sample_n": 2,
    }
    monkeypatch.setattr(
        "ui.features.sentiment_ui.sample_size_slider",
        lambda *a, **k: 9,
    )
    seen = {}

    def fake_plot(sample, method="cosmus"):
        seen["n"] = len(sample)
        seen["method"] = method

    monkeypatch.setattr("ui.features.sentiment_ui.plot_sentiment_barchart", fake_plot)
    render_sentiment_cosmus(pd.Series(["a", "b", "c", "d"]))
    assert seen["n"] == 2
    assert seen["method"] == "cosmus"
    assert mock_st.session_state["cosmus_done"] is True
