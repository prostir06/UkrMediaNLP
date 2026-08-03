"""Tests that nlp compute modules import without Streamlit."""

import ast
import importlib
from pathlib import Path


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
        "nlp_analysis",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        source = open(mod.__file__, encoding="utf-8").read()
        assert "import streamlit" not in source
        assert "from streamlit" not in source


def test_nlp_analysis_facade_has_no_ui_imports():
    tree = ast.parse(Path("nlp_analysis.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ui"), alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("ui"), module
