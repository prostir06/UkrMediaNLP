"""Unit tests for ui.charts figure builders."""

import pytest

from exceptions import NLPAnalysisError


def test_build_sentiment_figure_news_rules():
    from ui.charts import build_sentiment_figure

    fig = build_sentiment_figure(
        ["перемога команди", "обстріл міста", "новий закон"],
        method="news_rules",
    )
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_sentiment_figure_unknown_method():
    from ui.charts import build_sentiment_figure

    with pytest.raises(ValueError):
        build_sentiment_figure(["a"], method="unknown")


def test_build_sentiment_figure_empty_returns_none(monkeypatch):
    from ui.charts import build_sentiment_figure

    monkeypatch.setattr(
        "nlp.news_sentiment.classify_news_sentiment_batch",
        lambda texts: [],
    )
    assert build_sentiment_figure(["x"], method="news_rules") is None


def test_build_sentiment_figure_wraps_unexpected(monkeypatch):
    from ui.charts import build_sentiment_figure

    def boom(texts):
        raise RuntimeError("classifier crashed")

    monkeypatch.setattr(
        "nlp.news_sentiment.classify_news_sentiment_batch",
        boom,
    )
    with pytest.raises(RuntimeError, match="Sentiment chart failed"):
        build_sentiment_figure(["x"], method="news_rules")


def test_build_ner_figure_empty(monkeypatch):
    from ui.charts import build_ner_figure

    monkeypatch.setattr("nlp.ner.extract_entities_batch", lambda texts, entity: [[], []])
    fig, title = build_ner_figure(["a", "b"], entity="PER")
    assert fig is None
    assert title


def test_build_ner_figure_with_entities(monkeypatch):
    from ui.charts import build_ner_figure

    monkeypatch.setattr(
        "nlp.ner.extract_entities_batch",
        lambda texts, entity: [["Зеленський"], ["Зеленський", "Шмигаль"]],
    )
    fig, title = build_ner_figure(["t1", "t2"], entity="PER")
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_pos_figure_empty(monkeypatch):
    from ui.charts import build_pos_figure

    monkeypatch.setattr("nlp.pos.extract_pos_batch", lambda texts: [[], []])
    assert build_pos_figure(["", ""]) is None


def test_build_pos_figure_with_tags(monkeypatch):
    from ui.charts import build_pos_figure

    monkeypatch.setattr(
        "nlp.pos.extract_pos_batch",
        lambda texts: [["NOUN", "VERB"], ["NOUN", "ADJ"]],
    )
    fig = build_pos_figure(["текст"])
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_build_emotion_figure_propagates_nlp_error(monkeypatch):
    from ui.charts import build_emotion_figure

    def boom(texts):
        raise NLPAnalysisError("missing model", step="emotions_load")

    monkeypatch.setattr("nlp.sentiment.classify_emotions_batch", boom)
    with pytest.raises(NLPAnalysisError):
        build_emotion_figure(["радість"])


def test_build_emotion_figure_soft_fail(monkeypatch):
    from ui.charts import build_emotion_figure

    monkeypatch.setattr(
        "nlp.sentiment.classify_emotions_batch",
        lambda texts: (_ for _ in ()).throw(RuntimeError("oom")),
    )
    assert build_emotion_figure(["x"]) is None


def test_build_emotion_figure_success(monkeypatch):
    from ui.charts import build_emotion_figure

    monkeypatch.setattr(
        "nlp.sentiment.classify_emotions_batch",
        lambda texts: [(["Радість", "Подив"], "Радість")],
    )
    fig = build_emotion_figure(["ура"])
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)
