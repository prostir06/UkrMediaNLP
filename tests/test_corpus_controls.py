import pandas as pd

from ui import corpus_controls
from ui.corpus_controls import (
    CORPUS_FUNCTIONS,
    build_corpus_from_sources,
    load_corpus_into_session,
    render_corpus_sidebar,
)


def test_build_corpus_partial_failure():
    def fake_load(name, progress_callback=None):
        if name == "Bad":
            raise RuntimeError("boom")
        return pd.DataFrame(
            {
                "title": [f"{name}-t"],
                "published": ["2024-06-01"],
                "content": ["body"],
                "source": [name],
                "link": ["u"],
                "description": [""],
            }
        )

    df, warnings = build_corpus_from_sources(
        ["Good", "Bad"],
        load_articles_fn=fake_load,
        max_sources=10,
        max_rows=50,
        date_from=None,
        date_to=None,
        include_missing=True,
    )

    assert len(df) == 1
    assert df.iloc[0]["source"] == "Good"
    assert any("Bad" in warning for warning in warnings)
    assert df.attrs.get("scrape_stats_by_source") == []


def test_build_corpus_caps_sources_then_filters_and_caps_rows():
    loaded = []

    def fake_load(name, progress_callback=None):
        loaded.append(name)
        return pd.DataFrame(
            {
                "title": [f"{name}-old", f"{name}-new"],
                "published": ["2024-05-01", "2024-06-02"],
                "source": [name, name],
            }
        )

    df, warnings = build_corpus_from_sources(
        ["One", "Two", "Three"],
        load_articles_fn=fake_load,
        max_sources=2,
        max_rows=1,
        date_from="2024-06-01",
        date_to="2024-06-30",
        include_missing=False,
    )

    assert loaded == ["One", "Two"]
    assert len(df) == 1
    assert df.iloc[0]["published"] == "2024-06-02"
    assert warnings == []


def test_build_corpus_total_failure_returns_empty_frame_and_warnings():
    def failing_load(name, progress_callback=None):
        raise RuntimeError("offline")

    df, warnings = build_corpus_from_sources(
        ["One", "Two"],
        load_articles_fn=failing_load,
        max_sources=10,
        max_rows=50,
        date_from=None,
        date_to=None,
        include_missing=True,
    )

    assert df.empty
    assert len(warnings) == 2
    assert all("offline" in warning for warning in warnings)


def test_load_corpus_into_session_uses_config_caps(monkeypatch):
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"source": ["NV"]}), []

    monkeypatch.setattr(corpus_controls, "build_corpus_from_sources", fake_build)
    monkeypatch.setattr(corpus_controls, "MAX_CORPUS_SOURCES", 3)
    monkeypatch.setattr(corpus_controls, "MAX_CORPUS_ARTICLES_TOTAL", 40)

    df, warnings = load_corpus_into_session(
        ["NV"],
        None,
        None,
        True,
        "Новини",
        load_articles_fn=lambda *_args, **_kwargs: pd.DataFrame(),
    )

    assert list(df["source"]) == ["NV"]
    assert warnings == []
    assert captured["max_sources"] == 3
    assert captured["max_rows"] == 40


def test_render_corpus_sidebar_uses_stable_widget_keys(monkeypatch):
    calls = {}

    class FakeSidebar:
        def checkbox(self, label, **kwargs):
            calls[kwargs["key"]] = kwargs
            return kwargs["key"] == "corpus_all_category"

        def multiselect(self, label, options, **kwargs):
            calls[kwargs["key"]] = {"options": options, **kwargs}
            return ["NV"]

        def date_input(self, label, **kwargs):
            calls[kwargs["key"]] = kwargs
            return None

        def button(self, label, **kwargs):
            calls[kwargs["key"]] = kwargs
            return True

    monkeypatch.setattr(corpus_controls, "st", type("FakeSt", (), {"sidebar": FakeSidebar()})())
    monkeypatch.setattr(
        corpus_controls,
        "sources_for_category",
        lambda category: ["NV", "Українська правда"],
    )

    result = render_corpus_sidebar("Новини")

    assert set(result) == {
        "sources",
        "all_category",
        "date_from",
        "date_to",
        "include_missing",
        "load_clicked",
    }
    assert result["sources"] == ["NV", "Українська правда"]
    assert calls["corpus_sources_ms"]["disabled"] is True
    assert set(calls) == {
        "corpus_all_category",
        "corpus_sources_ms",
        "corpus_date_from",
        "corpus_date_to",
        "corpus_include_missing",
        "corpus_load_btn",
    }


def test_corpus_functions_are_stable():
    assert CORPUS_FUNCTIONS == frozenset({"Пошук у корпусі", "Тренди тем"})
