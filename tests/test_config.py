"""Tests for project configuration."""

from config import ARTICLE_COLUMNS, NEWS_SOURCES, NLP_FUNCTIONS_FULL, NLP_FUNCTIONS_LIGHT


def test_news_sources_have_required_keys():
    required = {"rss_url", "scraper", "intro"}
    for name, config in NEWS_SOURCES.items():
        assert required.issubset(config.keys()), f"Missing keys in {name}"


def test_news_sources_count():
    assert len(NEWS_SOURCES) == 8


def test_unian_uses_ukrainian_feed():
    assert "news_ukr" in NEWS_SOURCES["УНІАН"]["rss_url"]


def test_nlp_functions_include_core_features():

    expected_full = {
        "Вступ",
        "Огляд статей",
        "Уніграми",
        "Тональність (RoBERTa)",
        "Тональність (новини)",
        "Порівняння медіа",
        "Сумаризація",
    }
    assert expected_full.issubset(set(NLP_FUNCTIONS_FULL))
    assert "Тональність (RoBERTa)" not in NLP_FUNCTIONS_LIGHT
    assert "Тональність (новини)" in NLP_FUNCTIONS_LIGHT


def test_article_columns_schema():
    assert "title" in ARTICLE_COLUMNS
    assert "scraped_ok" in ARTICLE_COLUMNS


def test_get_cloud_light_reads_env(monkeypatch):
    from config import get_cloud_light

    monkeypatch.setenv("LIGHT_CLOUD", "1")
    assert get_cloud_light() is True


def test_get_cloud_light_reads_secrets(monkeypatch):
    import sys
    import types

    from config import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)

    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {"LIGHT_CLOUD": "true"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    # Force transformers-available path so secrets matter.
    monkeypatch.setattr("config._transformers_available", lambda: True)
    assert get_cloud_light() is True


def test_get_cloud_light_when_transformers_missing(monkeypatch):
    from config import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setattr("config._transformers_available", lambda: False)
    assert get_cloud_light() is True


def test_get_cloud_light_secrets_exception_returns_false(monkeypatch):
    import sys
    import types

    from config import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setattr("config._transformers_available", lambda: True)

    fake_st = types.ModuleType("streamlit")

    class BrokenSecrets:
        def get(self, *args, **kwargs):
            raise RuntimeError("secrets unavailable")

    fake_st.secrets = BrokenSecrets()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert get_cloud_light() is False
