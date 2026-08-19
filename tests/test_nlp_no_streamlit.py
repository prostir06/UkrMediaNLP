"""Tests that nlp compute modules import without Streamlit."""

import ast
import importlib
from pathlib import Path


def _module_imports_streamlit(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "streamlit" or alias.name.startswith("streamlit."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "streamlit" or module.startswith("streamlit."):
                return True
    return False


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
        "nlp.embeddings",
        "media_sources",
        "config",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        source = open(mod.__file__, encoding="utf-8").read()
        assert "import streamlit" not in source
        assert "from streamlit" not in source


def test_config_source_has_no_streamlit_import():
    assert not _module_imports_streamlit(Path("config.py"))


def test_nlp_analysis_file_absent():
    assert not Path("nlp_analysis.py").exists()


def test_app_does_not_import_nlp_analysis_facade():
    tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "nlp_analysis":
            raise AssertionError("app.py must not import nlp_analysis")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nlp_analysis":
                    raise AssertionError("app.py must not import nlp_analysis")
