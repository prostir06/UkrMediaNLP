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


def test_render_corpus_search_requires_corpus(mock_st):
    from app import render_corpus_search

    mock_st.session_state = {}
    render_corpus_search()
    mock_st.info.assert_called_once()


def test_commit_corpus_load_preserves_previous_on_total_failure(mock_st):
    from app import _commit_corpus_load

    previous = pd.DataFrame({"title": ["previous"]})
    mock_st.session_state = {"corpus_df": previous}

    replaced = _commit_corpus_load(
        pd.DataFrame(),
        ["A: failed", "B: failed"],
        sources=["A", "B"],
        category="Новини",
    )

    assert replaced is False
    assert mock_st.session_state["corpus_df"] is previous
    mock_st.error.assert_called_once()


def test_commit_corpus_load_stores_empty_when_no_previous_corpus(mock_st):
    from app import _commit_corpus_load

    mock_st.session_state = {}

    replaced = _commit_corpus_load(
        pd.DataFrame(),
        ["A: failed"],
        sources=["A"],
        category="Новини",
    )

    assert replaced is True
    assert mock_st.session_state["corpus_df"].empty
    assert mock_st.session_state["corpus_sources"] == ["A"]


def test_invalidate_stale_corpus_clears_source_mismatch(mock_st):
    from app import _invalidate_stale_corpus

    mock_st.session_state = {
        "corpus_df": pd.DataFrame({"title": ["old"]}),
        "corpus_category": "Новини",
        "corpus_sources": ["A"],
    }

    invalidated = _invalidate_stale_corpus(
        category="Новини",
        current_sources=["B"],
        all_category=False,
    )

    assert invalidated is True
    assert mock_st.session_state["corpus_df"].empty
    mock_st.warning.assert_called_once()


def test_invalidate_stale_corpus_clears_category_mismatch(mock_st):
    from app import _invalidate_stale_corpus

    mock_st.session_state = {
        "corpus_df": pd.DataFrame({"title": ["old"]}),
        "corpus_category": "Новини",
        "corpus_sources": ["A"],
    }

    invalidated = _invalidate_stale_corpus(
        category="Спорт",
        current_sources=["A"],
        all_category=True,
    )

    assert invalidated is True
    assert mock_st.session_state["corpus_df"].empty


def test_load_data_dispatches_corpus_search_without_loading_source(mock_st, monkeypatch):
    from app import load_data

    render = MagicMock()
    monkeypatch.setattr("app.render_corpus_search", render)
    monkeypatch.setattr(
        "app.get_source_config",
        MagicMock(side_effect=AssertionError("single-source config must not be loaded")),
    )
    monkeypatch.setattr(
        "app._load_source",
        MagicMock(side_effect=AssertionError("single source must not be loaded")),
    )

    load_data("NV", "Пошук у корпусі")

    render.assert_called_once_with()


def test_render_topic_trends_wires_hybrid_terms_and_charts(mock_st, monkeypatch):
    from app import render_topic_trends

    corpus = pd.DataFrame(
        {
            "title": ["Футбол", "Матч"],
            "content": ["Збірна перемогла", "Футбол сьогодні"],
            "published": ["2024-03-01", "2024-03-02"],
            "source": ["A", "B"],
        }
    )
    mock_st.session_state = {"corpus_df": corpus}
    expander = MagicMock()
    expander.__enter__.return_value = expander
    expander.__exit__.return_value = False
    mock_st.expander.return_value = expander
    mock_st.text_area.return_value = "збірна\nфутбол"
    mock_st.multiselect.return_value = ["футбол", "збірна"]
    mock_st.radio.return_value = "Тиждень"
    mock_st.selectbox.return_value = "футбол"

    monkeypatch.setattr("app.get_cloud_light", lambda: False)
    monkeypatch.setattr("app.suggest_terms", lambda df, n: ["футбол", "матч"])
    monkeypatch.setattr("app.suggest_lda_labels", lambda df: ["спорт"])
    trends = pd.DataFrame({"bucket": [pd.Timestamp("2024-03-04")], "term": ["футбол"], "count": [2]})
    source_trends = pd.DataFrame(
        {"bucket": [pd.Timestamp("2024-03-04")], "source": ["A"], "count": [1]}
    )
    aggregate = MagicMock(return_value=trends)
    aggregate_by_source = MagicMock(return_value=source_trends)
    monkeypatch.setattr("app.aggregate_trends", aggregate)
    monkeypatch.setattr("app.aggregate_trends_by_source", aggregate_by_source)
    monkeypatch.setattr("app.build_trends_line", MagicMock(return_value="trend-figure"))
    monkeypatch.setattr(
        "app.build_source_trends_line",
        MagicMock(return_value="source-figure"),
    )

    render_topic_trends()

    aggregate.assert_called_once_with(corpus, ["футбол", "збірна"], freq="W-MON")
    aggregate_by_source.assert_called_once_with(corpus, "футбол", freq="W-MON")
    assert mock_st.plotly_chart.call_count == 2
    options = mock_st.multiselect.call_args.args[1]
    assert options == ["футбол", "матч", "спорт", "збірна"]


def test_load_data_dispatches_topic_trends_without_loading_source(mock_st, monkeypatch):
    from app import load_data

    render = MagicMock()
    monkeypatch.setattr("app.render_topic_trends", render)
    monkeypatch.setattr(
        "app._load_source",
        MagicMock(side_effect=AssertionError("single source must not be loaded")),
    )

    load_data("NV", "Тренди тем")

    render.assert_called_once_with()


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
