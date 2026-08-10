"""Smoke tests for feature render_* screens with mocked Streamlit."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ui.features import compare, corpus_search, corpus_trends, ner, ngrams, pos, textstat, topics

ST_TARGETS = (
    "ui.features.textstat.st",
    "ui.features.topics.st",
    "ui.features.pos.st",
    "ui.features.corpus_search.st",
    "ui.features.corpus_trends.st",
    "ui.features.compare.st",
    "ui.features.ngrams.st",
    "ui.features.ner.st",
)


@pytest.fixture
def mock_st(monkeypatch):
    st = MagicMock()
    st.session_state = {}
    st.columns.side_effect = lambda n, **k: [MagicMock() for _ in range(int(n))]
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


def test_render_corpus_search_direct(mock_st, monkeypatch):
    corpus = pd.DataFrame(
        {
            "title": ["Київ новини"],
            "content": ["Текст про Київ"],
            "source": ["NV"],
            "published": ["2024-01-01"],
        }
    )
    mock_st.session_state["corpus_df"] = corpus
    mock_st.text_input.return_value = "Київ"
    mock_st.radio.side_effect = ["Ключові слова", "Заголовках і текстах"]
    mock_st.checkbox.return_value = False
    monkeypatch.setattr(
        corpus_search,
        "search_corpus",
        lambda *a, **k: corpus.assign(snippet=["…Київ…"], relevance=[3]),
    )
    monkeypatch.setattr(corpus_search, "build_source_hit_bar", lambda *a, **k: MagicMock())
    corpus_search.render_corpus_search()
    mock_st.subheader.assert_called()


def test_render_corpus_search_semantic_mode(mock_st, monkeypatch):
    corpus = pd.DataFrame(
        {
            "title": ["Київ новини"],
            "content": ["Текст про Київ"],
            "source": ["NV"],
            "published": ["2024-01-01"],
        }
    )
    mock_st.session_state["corpus_df"] = corpus
    mock_st.text_input.return_value = "Київ"
    mock_st.radio.return_value = "Семантичний"
    monkeypatch.setattr(corpus_search, "embeddings_enabled", lambda: True)
    monkeypatch.setattr(
        corpus_search,
        "search_corpus_semantic",
        lambda *a, **k: corpus.assign(snippet=["…Київ…"], relevance=[0.9]),
    )
    monkeypatch.setattr(corpus_search, "build_source_hit_bar", lambda *a, **k: MagicMock())
    corpus_search.render_corpus_search()
    mock_st.subheader.assert_called()


def test_render_corpus_search_semantic_disabled_shows_info(mock_st, monkeypatch):
    corpus = pd.DataFrame(
        {
            "title": ["Київ"],
            "content": ["текст"],
            "source": ["NV"],
            "published": ["2024-01-01"],
        }
    )
    mock_st.session_state["corpus_df"] = corpus
    mock_st.text_input.return_value = "Київ"
    mock_st.radio.return_value = "Семантичний"
    monkeypatch.setattr(corpus_search, "embeddings_enabled", lambda: False)
    corpus_search.render_corpus_search()
    mock_st.info.assert_called()


def test_render_topic_trends_direct(mock_st, monkeypatch):
    corpus = pd.DataFrame(
        {
            "title": ["тема"],
            "content": ["текст"],
            "source": ["NV"],
            "published": ["2024-01-01"],
        }
    )
    mock_st.session_state["corpus_df"] = corpus
    mock_st.text_area.return_value = "тема"
    mock_st.multiselect.return_value = ["тема"]
    mock_st.radio.return_value = "День"
    monkeypatch.setattr(corpus_trends, "get_cloud_light", lambda: True)
    monkeypatch.setattr(corpus_trends, "suggest_terms", lambda *a, **k: ["тема"])
    monkeypatch.setattr(
        corpus_trends,
        "aggregate_trends",
        lambda *a, **k: pd.DataFrame(
            {"bucket": [pd.Timestamp("2024-01-01")], "term": ["тема"], "count": [1]}
        ),
    )
    monkeypatch.setattr(corpus_trends, "build_trends_line", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        corpus_trends,
        "aggregate_trends_by_source",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(corpus_trends, "build_source_trends_line", lambda *a, **k: None)
    corpus_trends.render_topic_trends()
    mock_st.subheader.assert_called()


def test_render_compare_media_direct(mock_st, monkeypatch):
    mock_st.selectbox.return_value = "TSN"
    df = pd.DataFrame({"title": ["a b"], "content": ["x"], "source": ["NV"]})
    monkeypatch.setattr(compare, "load_source", lambda name: df)
    monkeypatch.setattr(compare, "preprocess", lambda s: s.astype(str))
    monkeypatch.setattr(compare, "get_top_n_words", lambda *a, **k: [("слово", 2)])
    monkeypatch.setattr(
        compare,
        "classify_news_sentiment_batch",
        lambda *a, **k: ["нейтральна"],
    )
    compare.render_compare_media("NV")
    mock_st.subheader.assert_called()


def test_render_unigrams_direct(mock_st, monkeypatch):
    monkeypatch.setattr(ngrams, "get_top_n_words", lambda *a, **k: [("слово", 3)])
    ngrams.render_unigrams(pd.Series(["слово слово"]))
    mock_st.subheader.assert_called()
    mock_st.table.assert_called()


def test_render_ner_direct(mock_st, monkeypatch):
    mock_st.selectbox.return_value = "PER"
    mock_st.radio.return_value = "Лише заголовки"
    monkeypatch.setattr(ner, "plot_most_common_named_entity_barchart", lambda *a, **k: None)
    ner.render_ner(
        pd.DataFrame({"content": ["Текст"]}),
        pd.Series(["Іван у Києві"]),
    )
    mock_st.subheader.assert_called()
