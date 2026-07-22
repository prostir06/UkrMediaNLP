"""Unit tests for word-cloud helpers."""

import pandas as pd

from nlp.wordcloud_render import build_wordcloud_images


def test_build_wordcloud_empty_input():
    assert build_wordcloud_images([]) == []
    assert build_wordcloud_images(["", "  "], lemmatize=False) == []


def test_build_wordcloud_without_lemmatize(monkeypatch):
    # Tiny style keeps the test fast and memory-light.
    styles = [
        {
            "width": 100,
            "height": 60,
            "max_words": 20,
            "background_color": "white",
        }
    ]
    monkeypatch.setattr(
        "nlp.wordcloud_render._resolve_font_path",
        lambda: None,
    )
    images = build_wordcloud_images(
        ["Уряд ухвалив новий закон про економіку"],
        styles=styles,
        lemmatize=False,
    )
    assert len(images) == 1
    assert images[0].ndim == 3


def test_build_wordcloud_lemmatize_fallback(monkeypatch):
    styles = [
        {
            "width": 80,
            "height": 40,
            "max_words": 10,
            "background_color": "white",
        }
    ]

    def boom():
        raise RuntimeError("spaCy missing")

    monkeypatch.setattr("nlp.model_registry.resolve_spacy_nlp", boom)
    monkeypatch.setattr("nlp.wordcloud_render._resolve_font_path", lambda: None)

    images = build_wordcloud_images(
        pd.Series(["Президент відвідав Київ"]),
        styles=styles,
        lemmatize=True,
    )
    assert len(images) == 1


def test_build_wordcloud_skips_broken_style(monkeypatch):
    class BrokenCloud:
        def __init__(self, *args, **kwargs):
            raise ValueError("bad style")

    monkeypatch.setattr("nlp.wordcloud_render.WordCloud", BrokenCloud)
    monkeypatch.setattr("nlp.wordcloud_render._resolve_font_path", lambda: None)

    images = build_wordcloud_images(["закон уряд економіка"], lemmatize=False)
    assert images == []
