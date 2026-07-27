import pandas as pd

from nlp.corpus import parse_manual_terms, suggest_terms


def test_parse_manual_terms():
    assert parse_manual_terms(" футбол \n\nЗбірна\nфутбол ") == ["футбол", "Збірна"]


def test_suggest_terms_from_titles(monkeypatch):
    monkeypatch.setattr(
        "nlp.ngrams.get_top_n_words",
        lambda corpus, n=10: [("футбол", 3), ("матч", 2)][:n],
    )
    df = pd.DataFrame({"title": ["a", "b"], "content": ["c", "d"]})
    assert suggest_terms(df, n=2) == ["футбол", "матч"]


def test_suggest_lda_labels_returns_empty_on_raise(monkeypatch):
    from nlp.corpus import suggest_lda_labels

    def _boom(*_args, **_kwargs):
        raise RuntimeError("lda unavailable")

    monkeypatch.setattr("nlp.topics.run_topic_modeling", _boom)
    df = pd.DataFrame({"content": ["text one", "text two"]})
    assert suggest_lda_labels(df, number_topics=3) == []


def test_suggest_lda_labels_strips_topic_prefix(monkeypatch):
    from nlp.corpus import suggest_lda_labels

    monkeypatch.setattr(
        "nlp.topics.run_topic_modeling",
        lambda *_args, **_kwargs: ["Тема 1: футбол, матч", "економіка"],
    )
    df = pd.DataFrame({"content": ["text one", "text two"]})

    assert suggest_lda_labels(df, number_topics=2) == ["футбол, матч", "економіка"]
