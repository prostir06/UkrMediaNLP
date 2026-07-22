"""Extra unit tests for app helpers (sliders, progress load, sentiment table)."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from exceptions import DataLoaderError


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    progress = MagicMock()
    st.progress.return_value = progress
    st.slider.return_value = 5
    monkeypatch.setattr("app.st", st)
    return st


def test_sample_size_slider_clamps(mock_st):
    from app import _sample_size_slider

    value = _sample_size_slider("N", default=100, max_value=3, key="k1")
    assert value == 5  # mocked slider return
    kwargs = mock_st.slider.call_args.kwargs
    assert kwargs["max_value"] == 3
    assert kwargs["value"] == 3  # clamped default


def test_sample_size_slider_handles_invalid_default(mock_st, monkeypatch):
    from app import _sample_size_slider

    mock_st.slider.side_effect = TypeError("bad")
    value = _sample_size_slider("N", default="oops", max_value=2, key="k2")
    assert value == 1


def test_load_source_passes_progress_and_clears(mock_st, monkeypatch):
    from app import _load_source

    captured = {}

    def fake_load(source_name, progress_callback=None):
        captured["source"] = source_name
        if progress_callback:
            progress_callback(1, 2)
            progress_callback(2, 2)
        return pd.DataFrame({"title": ["a"]})

    monkeypatch.setattr("app.load_articles", fake_load)
    df = _load_source("NV")
    assert captured["source"] == "NV"
    assert len(df) == 1
    mock_st.progress.return_value.empty.assert_called()


def test_load_source_wraps_unexpected_errors(mock_st, monkeypatch):
    from app import _load_source

    monkeypatch.setattr(
        "app.load_articles",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(DataLoaderError):
        _load_source("NV")
    mock_st.progress.return_value.empty.assert_called()


def test_render_sentiment_table(mock_st):
    from app import _render_sentiment_table

    titles = pd.Series(["A", "B"])
    _render_sentiment_table(titles, ["Позитивна", "Негативна"])
    mock_st.dataframe.assert_called_once()
    mock_st.download_button.assert_called_once()


def test_render_sentiment_table_empty(mock_st):
    from app import _render_sentiment_table

    _render_sentiment_table(pd.Series([], dtype=str), [])
    mock_st.warning.assert_called()


def test_select_sidebar_source_returns_media(mock_st, monkeypatch):
    from app import _select_sidebar_source

    mock_st.sidebar.selectbox.side_effect = ["Технології", "DOU"]
    monkeypatch.setattr(
        "app.sources_for_category",
        lambda cat: ["DOU", "AIN.UA"] if cat == "Технології" else [],
    )
    assert _select_sidebar_source() == "DOU"


def test_select_sidebar_source_empty_category(mock_st, monkeypatch):
    from app import _select_sidebar_source

    mock_st.sidebar.selectbox.return_value = "Економіка"
    monkeypatch.setattr("app.sources_for_category", lambda cat: [])
    assert _select_sidebar_source() is None
    mock_st.sidebar.warning.assert_called()
    mock_st.info.assert_called()


def test_select_sidebar_source_handles_widget_error(mock_st, monkeypatch):
    from app import _select_sidebar_source

    mock_st.sidebar.selectbox.side_effect = RuntimeError("widget boom")
    assert _select_sidebar_source() is None
    mock_st.error.assert_called()


def test_load_data_unknown_source(mock_st):
    from app import load_data

    load_data("Немає такого", "Вступ")
    mock_st.error.assert_called()


def test_render_compare_media(mock_st, monkeypatch):
    from app import render_compare_media

    cols = [MagicMock(), MagicMock()]
    mock_st.columns.return_value = cols
    mock_st.selectbox.return_value = "TSN"

    df = pd.DataFrame(
        {
            "title": ["Перемога команди", "Обстріл міста"],
            "content": ["a", "b"],
            "scraped_ok": [True, False],
            "published": ["", ""],
            "category": ["", ""],
            "link": ["https://a", "https://b"],
        }
    )
    monkeypatch.setattr("app._load_source", lambda name: df)
    monkeypatch.setattr("app.preprocess", lambda s: s.fillna("").astype(str))
    monkeypatch.setattr(
        "app.get_top_n_words",
        lambda titles, n: [("команда", 2), ("місто", 1)],
    )

    render_compare_media("NV")
    mock_st.subheader.assert_called()
