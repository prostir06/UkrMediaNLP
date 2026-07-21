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
    from config import NLP_FUNCTIONS_FULL, NLP_FUNCTIONS_LIGHT

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
