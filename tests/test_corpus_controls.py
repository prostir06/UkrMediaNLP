import pandas as pd

from ui import corpus_controls
from ui.corpus_controls import (
    CORPUS_FUNCTIONS,
    build_corpus_from_sources,
    load_corpus_into_session,
    render_corpus_sidebar,
)


def test_build_corpus_concurrent_loads_all_sources(monkeypatch):
    monkeypatch.setattr(corpus_controls, "CORPUS_LOAD_WORKERS", 3)
    seen = []

    def fake_load(name, progress_callback=None):
        seen.append(name)
        return pd.DataFrame(
            {
                "title": [name],
                "published": ["2024-06-01"],
                "content": ["body"],
                "source": [name],
                "link": ["u"],
                "description": [""],
            }
        )

    df, warnings = build_corpus_from_sources(
        ["A", "B", "C"],
        load_articles_fn=fake_load,
        max_sources=10,
        max_rows=50,
        date_from=None,
        date_to=None,
        include_missing=True,
    )
    assert warnings == []
    assert set(seen) == {"A", "B", "C"}
    assert set(df["source"]) == {"A", "B", "C"}
    assert "search_blob" in df.columns


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

    assert set(loaded) == {"One", "Two"}
    assert len(df) == 1
    assert df.iloc[0]["published"] == "2024-06-02"
    assert warnings == []
    assert "search_blob" in df.columns


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
    monkeypatch.setattr(corpus_controls, "_try_load_from_store", lambda **_kwargs: None)
    monkeypatch.setattr(corpus_controls, "_try_upsert_to_store", lambda _df: (None, None))
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
    assert df.attrs.get("corpus_origin") == "live"


def test_load_corpus_prefers_store_when_nonempty(monkeypatch):
    stored = pd.DataFrame({"source": ["NV"], "title": ["from-store"]})
    stored.attrs["corpus_origin"] = "postgres"
    monkeypatch.setattr(corpus_controls, "_try_load_from_store", lambda **_kwargs: stored)

    def boom(**_kwargs):
        raise AssertionError("live load should be skipped")

    monkeypatch.setattr(corpus_controls, "build_corpus_from_sources", boom)

    df, warnings = load_corpus_into_session(
        ["NV"],
        None,
        None,
        True,
        "Новини",
        load_articles_fn=lambda *_args, **_kwargs: pd.DataFrame(),
    )
    assert warnings == []
    assert df.iloc[0]["title"] == "from-store"
    assert df.attrs["corpus_origin"] == "postgres"


def test_load_corpus_upserts_after_live(monkeypatch):
    live = pd.DataFrame({"source": ["NV"], "title": ["live"]})
    monkeypatch.setattr(corpus_controls, "_try_load_from_store", lambda **_kwargs: None)
    monkeypatch.setattr(
        corpus_controls,
        "build_corpus_from_sources",
        lambda **_kwargs: (live, []),
    )
    monkeypatch.setattr(corpus_controls, "_try_upsert_to_store", lambda _df: (2, None))

    df, _warnings = load_corpus_into_session(
        ["NV"],
        None,
        None,
        True,
        "Новини",
        load_articles_fn=lambda *_args, **_kwargs: pd.DataFrame(),
    )
    assert df.attrs["store_upserted"] == 2
    assert df.attrs["corpus_origin"] == "live"


def test_try_load_from_store_soft_fails(monkeypatch):
    monkeypatch.setattr(corpus_controls, "is_store_configured", lambda: True)

    def boom_scope():
        raise RuntimeError("db offline")

    monkeypatch.setattr(corpus_controls, "session_scope", boom_scope)
    assert (
        corpus_controls._try_load_from_store(
            ["NV"], None, None, True, "Новини", max_rows=10
        )
        is None
    )


def test_try_load_from_store_postprocess_soft_fails(monkeypatch):
    monkeypatch.setattr(corpus_controls, "is_store_configured", lambda: True)

    class _Scope:
        def __enter__(self):
            return object()

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(corpus_controls, "session_scope", lambda: _Scope())
    monkeypatch.setattr(
        corpus_controls,
        "load_corpus_from_store",
        lambda *_a, **_k: pd.DataFrame({"source": ["NV"], "title": ["t"]}),
    )
    monkeypatch.setattr(
        corpus_controls,
        "merge_source_frames",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("merge")),
    )
    assert (
        corpus_controls._try_load_from_store(
            ["NV"], None, None, True, "Новини", max_rows=10
        )
        is None
    )


def test_try_upsert_to_store_returns_error_message(monkeypatch):
    monkeypatch.setattr(corpus_controls, "is_store_configured", lambda: True)

    def boom_scope():
        raise RuntimeError("write failed")

    monkeypatch.setattr(corpus_controls, "session_scope", boom_scope)
    count, err = corpus_controls._try_upsert_to_store(
        pd.DataFrame({"source": ["NV"], "title": ["t"]})
    )
    assert count is None
    assert "write failed" in str(err)


def test_try_upsert_skips_when_offline(monkeypatch):
    monkeypatch.setattr(corpus_controls, "is_store_configured", lambda: False)
    assert corpus_controls._try_upsert_to_store(pd.DataFrame({"a": [1]})) == (None, None)


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
