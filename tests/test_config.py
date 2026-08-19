"""Tests for project configuration."""

from config import (
    ARTICLE_COLUMNS,
    MEDIA_CATEGORIES,
    NEWS_SOURCES,
    NLP_FUNCTIONS_FULL,
    NLP_FUNCTIONS_LIGHT,
    sources_for_category,
    validate_news_sources_schema,
)


def test_news_sources_have_required_keys():
    required = {"category", "rss_url", "scraper", "intro"}
    for name, config in NEWS_SOURCES.items():
        assert required.issubset(config.keys()), f"Missing keys in {name}"
    assert validate_news_sources_schema() == []


def test_news_sources_count():
    assert len(NEWS_SOURCES) == 28


def test_scrape_sample_urls_cover_all_sources():
    from config import SCRAPE_SAMPLE_URLS

    assert set(SCRAPE_SAMPLE_URLS) == set(NEWS_SOURCES)
    assert all(str(url).startswith("http") for url in SCRAPE_SAMPLE_URLS.values())


def test_news_source_config_typeddict_keys():
    from media_sources import REQUIRED_SOURCE_KEYS, NewsSourceConfig

    assert set(NewsSourceConfig.__annotations__) == set(REQUIRED_SOURCE_KEYS)


def test_media_categories_and_news_sources():
    assert MEDIA_CATEGORIES == (
        "Новини",
        "Економіка",
        "Спорт",
        "Технології",
    )
    news = sources_for_category("Новини")
    economy = sources_for_category("Економіка")
    sport = sources_for_category("Спорт")
    tech = sources_for_category("Технології")
    assert len(news) == 7
    assert len(economy) == 5
    assert len(sport) == 7
    assert len(tech) == 9
    assert "УНІАН" in news
    assert "Українська правда" not in news
    assert "Економічна правда" in economy
    assert "Football.ua" in sport
    assert "NV (Спорт)" in sport
    assert "DOU" in tech
    assert "AIN.UA" in tech


def test_sport_radio_svoboda_uses_api_rss():
    url = NEWS_SOURCES["Радіо Свобода (Спорт)"]["rss_url"]
    assert url.startswith("https://www.radiosvoboda.org/api/")
    assert "/z/21679" not in url


def test_economy_radio_svoboda_uses_api_rss():
    url = NEWS_SOURCES["Радіо Свобода (Економіка)"]["rss_url"]
    assert url.startswith("https://www.radiosvoboda.org/api/")
    assert "/z/2734" not in url


def test_epravda_uses_news_rss_not_catalog():
    url = NEWS_SOURCES["Економічна правда"]["rss_url"]
    assert "rss/news" in url
    assert "rss-info" not in url


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
    from runtime_env import get_cloud_light

    monkeypatch.setenv("LIGHT_CLOUD", "1")
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")
    assert get_cloud_light() is True


def test_get_cloud_light_default_stable_without_allow(monkeypatch):
    from runtime_env import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.delenv("ALLOW_HEAVY_NLP", raising=False)
    monkeypatch.setattr("runtime_env._transformers_available", lambda: True)
    assert get_cloud_light() is True


def test_get_cloud_light_full_when_allow_heavy(monkeypatch):
    from runtime_env import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")
    monkeypatch.setattr("runtime_env._transformers_available", lambda: True)
    assert get_cloud_light() is False


def test_get_cloud_light_reads_secrets(monkeypatch):
    import sys
    import types

    from runtime_env import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")

    fake_st = types.ModuleType("streamlit")
    fake_st.secrets = {"LIGHT_CLOUD": "true"}
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.setattr("runtime_env._transformers_available", lambda: True)
    assert get_cloud_light() is True


def test_get_cloud_light_when_transformers_missing(monkeypatch):
    from runtime_env import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")
    monkeypatch.setattr("runtime_env._transformers_available", lambda: False)
    assert get_cloud_light() is True


def test_get_cloud_light_secrets_exception_keeps_env_allow(monkeypatch):
    import sys
    import types

    from runtime_env import get_cloud_light

    monkeypatch.delenv("LIGHT_CLOUD", raising=False)
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")
    monkeypatch.setattr("runtime_env._transformers_available", lambda: True)

    fake_st = types.ModuleType("streamlit")

    class BrokenSecrets:
        def get(self, *args, **kwargs):
            raise RuntimeError("secrets unavailable")

    fake_st.secrets = BrokenSecrets()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    # Env allow still wins when secrets blow up.
    assert get_cloud_light() is False


def test_corpus_config_and_functions():
    from config import (
        MAX_CORPUS_ARTICLES_TOTAL,
        MAX_CORPUS_SOURCES,
        MAX_TREND_TERMS,
        NLP_FUNCTIONS_FULL,
        NLP_FUNCTIONS_LIGHT,
    )

    assert MAX_CORPUS_SOURCES == 10
    assert MAX_CORPUS_ARTICLES_TOTAL == 300
    assert MAX_TREND_TERMS == 8
    assert "Пошук у корпусі" in NLP_FUNCTIONS_FULL
    assert "Тренди тем" in NLP_FUNCTIONS_FULL
    assert "Пошук у корпусі" in NLP_FUNCTIONS_LIGHT
    assert "Тренди тем" in NLP_FUNCTIONS_LIGHT
