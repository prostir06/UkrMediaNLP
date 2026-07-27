import pandas as pd

from nlp.corpus import parse_manual_terms, suggest_terms


def test_parse_manual_terms():
    assert parse_manual_terms(" футбол \n\nЗбірна\nфутбол ") == ["футбол", "Збірна"]


def test_suggest_terms_from_titles(monkeypatch):
    monkeypatch.setattr(
        "nlp.corpus.get_top_n_words",
        lambda corpus, n=10: [("футбол", 3), ("матч", 2)][:n],
    )
    df = pd.DataFrame({"title": ["a", "b"], "content": ["c", "d"]})
    assert suggest_terms(df, n=2) == ["футбол", "матч"]
