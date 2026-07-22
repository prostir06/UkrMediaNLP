"""Unit tests for media categories and source registry helpers."""

import pytest

from config import (
    MEDIA_CATEGORIES,
    NEWS_SOURCES,
    REQUIRED_SOURCE_KEYS,
    get_source_config,
    source_category,
    sources_for_category,
    validate_news_sources_schema,
)


def test_validate_news_sources_schema_clean():
    assert validate_news_sources_schema() == []


def test_validate_news_sources_schema_detects_problems():
    bad = {
        "Broken": {"rss_url": "not-a-url"},
        "BadCat": {
            "category": "Невідома",
            "rss_url": "https://example.com/rss",
            "scraper": "generic",
            "intro": "x",
        },
        "NotDict": "oops",
    }
    problems = validate_news_sources_schema(bad)
    assert any("Broken" in p for p in problems)
    assert any("BadCat" in p for p in problems)
    assert any("NotDict" in p for p in problems)


def test_validate_news_sources_schema_non_mapping():
    problems = validate_news_sources_schema("not-a-dict")  # type: ignore[arg-type]
    assert problems
    assert "mapping" in problems[0]


def test_sources_for_category_empty_and_invalid():
    assert sources_for_category(None) == []
    assert sources_for_category("") == []
    assert sources_for_category("   ") == []
    assert sources_for_category(123) == []  # type: ignore[arg-type]
    assert sources_for_category("Невідома категорія") == []


def test_sources_for_category_covers_all_media():
    collected: list[str] = []
    for category in MEDIA_CATEGORIES:
        names = sources_for_category(category)
        assert names, f"expected media in {category}"
        collected.extend(names)
    assert len(collected) == len(NEWS_SOURCES)
    assert set(collected) == set(NEWS_SOURCES)


def test_sources_for_category_skips_malformed(monkeypatch):
    monkeypatch.setattr(
        "config.NEWS_SOURCES",
        {
            "Ok": {"category": "Новини", "rss_url": "https://a", "scraper": "g", "intro": ""},
            "Bad": "nope",
            "Other": {"category": "Спорт", "rss_url": "https://b", "scraper": "g", "intro": ""},
        },
    )
    assert sources_for_category("Новини") == ["Ok"]
    assert sources_for_category("Спорт") == ["Other"]


def test_get_source_config_ok():
    cfg = get_source_config("NV")
    assert cfg["category"] == "Новини"
    assert cfg["rss_url"].startswith("https://")


def test_get_source_config_unknown():
    with pytest.raises(KeyError):
        get_source_config("Неіснуюче медіа")


def test_get_source_config_rejects_non_dict(monkeypatch):
    monkeypatch.setattr("config.NEWS_SOURCES", {"X": "bad"})
    with pytest.raises(KeyError):
        get_source_config("X")


def test_source_category_helpers():
    assert source_category("DOU") == "Технології"
    assert source_category("Football.ua") == "Спорт"
    assert source_category("Економічна правда") == "Економіка"
    assert source_category("missing") is None


def test_required_source_keys_constant():
    assert REQUIRED_SOURCE_KEYS == {"category", "rss_url", "scraper", "intro"}


def test_every_source_has_https_rss():
    for name, cfg in NEWS_SOURCES.items():
        assert cfg["rss_url"].startswith("https://"), name
        assert cfg["category"] in MEDIA_CATEGORIES, name
