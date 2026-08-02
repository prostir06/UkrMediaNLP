"""Tests that nlp compute modules import without Streamlit."""

import importlib


def test_nlp_modules_import_without_streamlit_session():
    modules = [
        "nlp.sentiment",
        "nlp.sentiment_models",
        "nlp.sentiment_inference",
        "nlp.resource_guard",
        "nlp.ner",
        "nlp.pos",
        "nlp.topics",
        "nlp.summarization",
        "nlp.wordcloud_render",
        "nlp.ngrams",
        "nlp.news_sentiment",
        "media_sources",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        source = open(mod.__file__, encoding="utf-8").read()
        assert "import streamlit" not in source
        assert "from streamlit" not in source
