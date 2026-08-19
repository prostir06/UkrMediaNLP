"""
Application configuration: NLP function lists, caps, and UI text.

Media registry lives in ``media_sources`` and is re-exported here for
compatibility (``from config import NEWS_SOURCES``).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Parse an integer env var; fall back to *default* on missing/invalid values."""
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float env var; fall back to *default* on missing/invalid values."""
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_REQUEST_TIMEOUT = 10

# Columns produced by RSS + scrape pipeline (see data_loader / rss).
ARTICLE_COLUMNS = [
    "title",
    "link",
    "description",
    "published",
    "category",
    "content",
    "source",
    "scraped_ok",
]

from media_sources import (  # noqa: E402
    MEDIA_CATEGORIES,
    NEWS_SOURCES,
    REQUIRED_SOURCE_KEYS,
    SCRAPE_SAMPLE_URLS,
    NewsSourceConfig,
    get_source_config,
    source_category,
    sources_for_category,
    validate_news_sources_schema,
)
from runtime_env import get_cloud_light  # noqa: E402

__all__ = [
    "ARTICLE_CACHE_ENABLED",
    "ARTICLE_COLUMNS",
    "CORPUS_LOAD_WORKERS",
    "DEFAULT_REQUEST_TIMEOUT",
    "DEFAULT_USER_AGENT",
    "HTTP_MAX_RETRIES",
    "HTTP_RETRY_BACKOFF",
    "MAX_ARTICLES",
    "MAX_CORPUS_ARTICLES_TOTAL",
    "MAX_CORPUS_SOURCES",
    "MAX_POS_ARTICLES",
    "MAX_POS_CONTENT_CHARS",
    "MAX_SENTIMENT_TITLES",
    "MAX_SUMMARY_ARTICLES",
    "MAX_TREND_TERMS",
    "MEDIA_CATEGORIES",
    "NEWS_SOURCES",
    "NGRAM_DESCRIPTION",
    "NLP_FUNCTIONS_FULL",
    "NLP_FUNCTIONS_LIGHT",
    "NewsSourceConfig",
    "REQUIRED_SOURCE_KEYS",
    "SCRAPE_DELAY_SECONDS",
    "SCRAPE_MAX_WORKERS",
    "SCRAPE_SAMPLE_URLS",
    "WORDCLOUD_DESCRIPTION",
    "get_cloud_light",
    "get_source_config",
    "source_category",
    "sources_for_category",
    "validate_news_sources_schema",
]


# Prefer ``runtime_env.get_cloud_light()`` at call sites (secrets may load later).

NLP_FUNCTIONS_FULL = [
    "Вступ",
    "Огляд статей",
    "Уніграми",
    "Біграми",
    "Триграми",
    "Ключові слова",
    "Хмара слів",
    "Статистика тексту",
    "Тематичне моделювання",
    "Розпізнавання сутностей",
    "Тональність (RoBERTa)",
    "Тональність (Емоції)",
    "Тональність (новини)",
    "Сумаризація",
    "Частини мови",
    "Порівняння медіа",
    "Пошук у корпусі",
    "Тренди тем",
]

NLP_FUNCTIONS_LIGHT = [
    "Вступ",
    "Огляд статей",
    "Уніграми",
    "Біграми",
    "Триграми",
    "Ключові слова",
    "Хмара слів",
    "Статистика тексту",
    "Тематичне моделювання",
    "Розпізнавання сутностей",
    "Тональність (новини)",
    "Сумаризація",
    "Частини мови",
    "Порівняння медіа",
    "Пошук у корпусі",
    "Тренди тем",
]


NGRAM_DESCRIPTION = (
    "N-грами — послідовності з N слів, що зустрічаються поруч. "
    "Уніграми (N=1) — окремі слова, біграми (N=2) — пари слів, "
    "триграми (N=3) — трійки слів."
)

WORDCLOUD_DESCRIPTION = (
    "Хмара слів показує частоту слів у заголовках: чим більше слово, "
    "тим частіше воно з'являлося в аналізованих матеріалах."
)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

SCRAPE_DELAY_SECONDS = _env_float("SCRAPE_DELAY_SECONDS", 1.0)
SCRAPE_MAX_WORKERS = _env_int("SCRAPE_MAX_WORKERS", 3)
HTTP_MAX_RETRIES = _env_int("HTTP_MAX_RETRIES", 3)
HTTP_RETRY_BACKOFF = _env_float("HTTP_RETRY_BACKOFF", 0.5)
MAX_ARTICLES = _env_int("MAX_ARTICLES", 50)
MAX_SENTIMENT_TITLES = _env_int("MAX_SENTIMENT_TITLES", 30)
MAX_SUMMARY_ARTICLES = _env_int("MAX_SUMMARY_ARTICLES", 10)
MAX_POS_ARTICLES = _env_int("MAX_POS_ARTICLES", 10)
MAX_CORPUS_SOURCES = _env_int("MAX_CORPUS_SOURCES", 10)
MAX_CORPUS_ARTICLES_TOTAL = _env_int("MAX_CORPUS_ARTICLES_TOTAL", 300)
CORPUS_LOAD_WORKERS = _env_int("CORPUS_LOAD_WORKERS", 3)
MAX_TREND_TERMS = _env_int("MAX_TREND_TERMS", 8)
MAX_POS_CONTENT_CHARS = _env_int("MAX_POS_CONTENT_CHARS", 5000)
ARTICLE_CACHE_ENABLED = os.environ.get("ARTICLE_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
}
