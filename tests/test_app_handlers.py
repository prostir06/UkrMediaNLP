"""Tests for Streamlit render helpers in app.py with mocked st."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from app import render_intro, render_snapshot
from config import MAX_ARTICLES


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    expander = MagicMock()
    expander.__enter__ = MagicMock(return_value=expander)
    expander.__exit__ = MagicMock(return_value=False)
    st.expander.return_value = expander
    cols = [MagicMock(), MagicMock(), MagicMock()]
    st.columns.return_value = cols
    monkeypatch.setattr("app.st", st)
    return st


def test_render_intro_calls_markdown(mock_st):
    render_intro("**Test intro**")
    mock_st.markdown.assert_called_once_with("**Test intro**")


def test_render_snapshot_shows_metrics_and_table(mock_st):
    df = pd.DataFrame(
        {
            "title": ["Новина 1", "Новина 2"],
            "published": ["2024-01-01", "2024-01-02"],
            "category": ["Політика", "Економіка"],
            "scraped_ok": [True, False],
            "link": ["https://example.com/1", "https://example.com/2"],
        }
    )
    df.attrs["total_in_feed"] = 100

    render_snapshot(df)

    mock_st.subheader.assert_called_once()
    mock_st.caption.assert_called()
    mock_st.dataframe.assert_called_once()
    mock_st.columns.assert_called_once_with(3)
    assert sum(col.metric.call_count for col in mock_st.columns.return_value) == 3
    mock_st.expander.assert_called_once()


def test_render_snapshot_skips_limit_caption_when_under_max(mock_st):
    df = pd.DataFrame(
        {
            "title": ["Новина"],
            "published": ["2024-01-01"],
            "category": ["Політика"],
            "scraped_ok": [True],
            "link": ["https://example.com/1"],
        }
    )
    df.attrs["total_in_feed"] = MAX_ARTICLES

    render_snapshot(df)

    for call in mock_st.caption.call_args_list:
        assert "MAX_ARTICLES" not in str(call)
