"""
Application configuration: Ukrainian media sources, NLP functions, and UI text.

Media are grouped by ``MEDIA_CATEGORIES`` (sidebar: Категорія → Медіа).
Each ``NEWS_SOURCES`` entry must include ``category``, ``rss_url``, ``scraper``,
and ``intro``. Category-specific RSS (Економіка / Спорт / Технології) reuse
site scrapers where selectors already match, otherwise ``generic``.
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

# Sidebar category labels (order is UI order). Must match NEWS_SOURCES[*].category.
MEDIA_CATEGORIES = (
    "Новини",
    "Економіка",
    "Спорт",
    "Технології",
)

REQUIRED_SOURCE_KEYS = frozenset({"category", "rss_url", "scraper", "intro"})

NEWS_SOURCES = {
    "NV": {
        "category": "Новини",
        "rss_url": "https://nv.ua/ukr/rss/all.xml",
        "scraper": "nv",
        "intro": """
**NV** — українське мультимедійне видання, що висвітлює політику, економіку,
технології та суспільні теми. Один із найпопулярніших новинних порталів України.
""",
    },
    "Радіо Свобода": {
        "category": "Новини",
        "rss_url": "https://www.radiosvoboda.org/api/zrqitl-vomx-tpeoumq",
        "scraper": "radiosvoboda",
        "intro": """
**Радіо Свобода** — міжнародне мультимедійне видання, яке висвітлює події в Україні
та регіоні. Частина мережі Radio Free Europe / Radio Liberty.
""",
    },
    "Українська правда": {
        "category": "Новини",
        "rss_url": "https://www.pravda.com.ua/rss/view_mainnews/",
        "scraper": "pravda",
        "intro": """
**Українська правда** — інтернет-видання, засноване у 2000 році. Один із найвідоміших
незалежних медіаресурсів України з фокусом на політику та суспільство.
""",
    },
    "Liga.net": {
        "category": "Новини",
        "rss_url": "https://news.liga.net/ua/top/rss.xml",
        "scraper": "liga",
        "intro": """
**Liga.net** — новинний портал з акцентом на бізнес, економіку, політику
та аналітику. Входить до медіахолдингу Liga.net.
""",
    },
    "RBC-UA": {
        "category": "Новини",
        "rss_url": "https://www.rbc.ua/static/rss/ukrnet.strong.ukr.rss.xml",
        "scraper": "rbc",
        "intro": """
**RBC-UA** — українське видання міжнародної мережі RBK, що висвітлює економічні
та політичні новини України.
""",
    },
    "Інтерфакс-Україна": {
        "category": "Новини",
        "rss_url": "https://interfax.com.ua/news/last.rss",
        "scraper": "interfax",
        "intro": """
**Інтерфакс-Україна** — інформаційне агентство, яке оперативно публікує новини
з політики, економіки та суспільного життя.
""",
    },
    "TSN": {
        "category": "Новини",
        "rss_url": "https://tsn.ua/rss/full.rss",
        "scraper": "tsn",
        "intro": """
**TSN** — телеканал «1+1» та новинний портал tsn.ua. Охоплює політику, події,
спорт, шоу-бізнес та lifestyle.
""",
    },
    "УНІАН": {
        "category": "Новини",
        "rss_url": "https://rss.unian.ua/site/news_ukr.rss",
        "scraper": "unian",
        "intro": """
**УНІАН** — одне з найбільших інформаційних агентств України. Публікує новини
політики, економіки, технологій та регіональних подій українською мовою.
""",
    },
    "Економічна правда": {
        "category": "Економіка",
        # /rss-info/ — HTML-каталог, не стрічка; робочий RSS:
        "rss_url": "https://epravda.com.ua/rss/news/",
        "scraper": "pravda",
        "intro": """
**Економічна правда** — бізнес і економіка від видавничої групи «Українська правда».
""",
    },
    "Бізнес Цензор": {
        "category": "Економіка",
        "rss_url": "https://assets.censor.net/rss/biz.censor.net/rss_uk_events.xml",
        "scraper": "generic",
        "intro": """
**Бізнес Цензор** — економічні та бізнес-новини censor.net.
""",
    },
    "NV (Економіка)": {
        "category": "Економіка",
        "rss_url": "https://nv.ua/ukr/rss/2292.xml",
        "scraper": "nv",
        "intro": """
**NV (Економіка)** — бізнес і фінанси на NV (biz.nv.ua).
""",
    },
    "Радіо Свобода (Економіка)": {
        "category": "Економіка",
        # Сторінка /z/2734 — HTML; робочий RSS з rssfeeds:
        "rss_url": "https://www.radiosvoboda.org/api/zvpk_l-vomx-tpeujjv",
        "scraper": "radiosvoboda",
        "intro": """
**Радіо Свобода (Економіка)** — економічні новини Радіо Свобода.
""",
    },
    "Liga.net (Економіка)": {
        "category": "Економіка",
        "rss_url": "https://news.liga.net/economics/rss.xml",
        "scraper": "liga",
        "intro": """
**Liga.net (Економіка)** — економічний розділ Liga.net.
""",
    },
    "NV (Спорт)": {
        "category": "Спорт",
        "rss_url": "https://nv.ua/ukr/rss/2371.xml",
        "scraper": "nv",
        "intro": """
**NV (Спорт)** — спортивна стрічка NV: футбол, інші види спорту та головні події.
""",
    },
    "Радіо Свобода (Спорт)": {
        "category": "Спорт",
        # Сторінка /z/21679 — HTML; робочий RSS з rssfeeds:
        "rss_url": "https://www.radiosvoboda.org/api/ztpmmyl-vomx-tpekjymv",
        "scraper": "radiosvoboda",
        "intro": """
**Радіо Свобода (Спорт)** — спортивні новини Радіо Свобода (зона «Новини | Спорт»).
""",
    },
    "Liga.net (Спорт)": {
        "category": "Спорт",
        "rss_url": "https://news.liga.net/sport/rss.xml",
        "scraper": "liga",
        "intro": """
**Liga.net (Спорт)** — спортивний розділ Liga.net: футбол та інші види спорту.
""",
    },
    "Champion": {
        "category": "Спорт",
        "rss_url": "https://champion.com.ua/ukr/rss/",
        "scraper": "generic",
        "intro": """
**Champion** — українське спортивне видання champion.com.ua.
""",
    },
    "Football.ua": {
        "category": "Спорт",
        "rss_url": "https://football.ua/rss2.ashx",
        "scraper": "generic",
        "intro": """
**Football.ua** — футбольні новини України та світу.
""",
    },
    "Суспільне Спорт": {
        "category": "Спорт",
        "rss_url": "https://suspilne.media/sport/rss/latest.rss",
        "scraper": "generic",
        "intro": """
**Суспільне Спорт** — спортивні новини Суспільного мовлення.
""",
    },
    "Tribuna": {
        "category": "Спорт",
        "rss_url": "https://rss.ua.tribuna.com/uk/feed.xml",
        "scraper": "generic",
        "intro": """
**Tribuna** — спортивні новини Tribuna.com українською.
""",
    },
    "NV (Технології)": {
        "category": "Технології",
        "rss_url": "https://nv.ua/ukr/rss/2346.xml",
        "scraper": "nv",
        "intro": """
**NV (Технології)** — технології та інновації на NV.
""",
    },
    "Liga.net (Технології)": {
        "category": "Технології",
        "rss_url": "https://tech.liga.net/top/rss.xml",
        "scraper": "liga",
        "intro": """
**Liga.net (Технології)** — tech-розділ Liga.net.
""",
    },
    "ITC.ua": {
        "category": "Технології",
        "rss_url": "https://feeds.feedburner.com/itcua",
        "scraper": "generic",
        "intro": """
**ITC.ua** — новини гаджетів, IT та цифрових технологій.
""",
    },
    "DOU": {
        "category": "Технології",
        "rss_url": "https://dou.ua/feed/",
        "scraper": "generic",
        "intro": """
**DOU** — спільнота українських IT-фахівців: новини та аналітика.
""",
    },
    "Mezha": {
        "category": "Технології",
        "rss_url": "https://mezha.ua/feed/",
        "scraper": "generic",
        "intro": """
**Mezha** — технології, гаджети та ігри (mezha.ua).
""",
    },
    "dev.ua": {
        "category": "Технології",
        "rss_url": "https://dev.ua/rss",
        "scraper": "generic",
        "intro": """
**dev.ua** — IT-новини України та світу.
""",
    },
    "Speka": {
        "category": "Технології",
        "rss_url": "https://speka.ua/rss",
        "scraper": "generic",
        "intro": """
**Speka** — технології, бізнес і інновації.
""",
    },
    "Vector": {
        "category": "Технології",
        "rss_url": "https://vctr.media/wp-content/uploads/rss.xml",
        "scraper": "generic",
        "intro": """
**Vector** — медіа про технології, стартапи та бізнес (vctr.media).
""",
    },
    "AIN.UA": {
        "category": "Технології",
        "rss_url": "https://ain.ua/feed/",
        "scraper": "generic",
        "intro": """
**AIN.UA** — новини IT, стартапів та цифрової економіки України.
""",
    },
}


def sources_for_category(category: str | None) -> list[str]:
    """
    Return media names registered for the given sidebar category.

    Empty / non-string *category* yields ``[]``. Malformed ``NEWS_SOURCES``
    entries (non-dict or missing ``category``) are skipped safely.
    """
    if not isinstance(category, str):
        return []
    category = category.strip()
    if not category:
        return []

    names: list[str] = []
    try:
        items = list(NEWS_SOURCES.items())
    except (AttributeError, TypeError) as exc:
        logger.warning("NEWS_SOURCES is not iterable: %s", exc)
        return []

    for name, config in items:
        try:
            if not isinstance(config, dict):
                logger.warning("Skipping non-dict source config: %r", name)
                continue
            if config.get("category") == category:
                names.append(str(name))
        except (TypeError, AttributeError) as exc:
            logger.warning("Skipping source %r: %s", name, exc)
            continue
    return names


def get_source_config(source_name: str) -> dict:
    """
    Return the config dict for *source_name*.

    Raises:
        KeyError: Unknown source or non-dict entry.
    """
    try:
        config = NEWS_SOURCES[source_name]
    except KeyError:
        raise
    except TypeError as exc:
        raise KeyError(source_name) from exc

    if not isinstance(config, dict):
        raise KeyError(source_name)
    return config


def source_category(source_name: str) -> str | None:
    """Return the sidebar category for *source_name*, or ``None`` if unknown."""
    try:
        category = get_source_config(source_name).get("category")
    except KeyError:
        return None
    except (TypeError, AttributeError) as exc:
        logger.debug("source_category(%r) failed: %s", source_name, exc)
        return None
    return category if isinstance(category, str) else None


def validate_news_sources_schema(
    sources: dict | None = None,
) -> list[str]:
    """
    Validate media registry shape; return a list of human-readable problems.

    Used by unit tests and optional startup checks. Does not raise.
    """
    problems: list[str] = []
    registry = NEWS_SOURCES if sources is None else sources
    try:
        items = list(registry.items())
    except (AttributeError, TypeError) as exc:
        return [f"NEWS_SOURCES is not a mapping: {exc}"]

    known_categories = set(MEDIA_CATEGORIES)
    for name, config in items:
        if not isinstance(config, dict):
            problems.append(f"{name}: config must be a dict")
            continue
        missing = REQUIRED_SOURCE_KEYS - set(config.keys())
        if missing:
            problems.append(f"{name}: missing keys {sorted(missing)}")
            continue
        category = config.get("category")
        if category not in known_categories:
            problems.append(f"{name}: unknown category {category!r}")
        rss_url = config.get("rss_url")
        if not isinstance(rss_url, str) or not rss_url.startswith(("http://", "https://")):
            problems.append(f"{name}: rss_url must be an http(s) URL")
    return problems


def _transformers_available() -> bool:
    """Return True when the transformers package is importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def _truthy_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def get_cloud_light() -> bool:
    """
    Resolve light-UI mode (hide RoBERTa / emotions).

    Light (stable) when:
    * ``LIGHT_CLOUD`` env/secret is set, or
    * ``transformers`` is missing, or
    * ``ALLOW_HEAVY_NLP`` is not enabled (default).

    Heavy models can hard-crash the Streamlit process (browser then shows
    ``WebSocket onclose`` / ``ERR_EMPTY_RESPONSE``). Opt in with
    ``ALLOW_HEAVY_NLP=1`` (env or Streamlit secrets).
    """
    if _truthy_flag(os.environ.get("LIGHT_CLOUD")):
        return True
    if not _transformers_available():
        return True

    allow_heavy = _truthy_flag(os.environ.get("ALLOW_HEAVY_NLP"))
    try:
        import streamlit as st

        if _truthy_flag(st.secrets.get("LIGHT_CLOUD", "")):
            return True
        allow_heavy = allow_heavy or _truthy_flag(
            st.secrets.get("ALLOW_HEAVY_NLP", ""),
        )
    except Exception:
        pass

    # Default: light/stable UI. Full NLP only when explicitly allowed.
    return not allow_heavy


# Prefer ``get_cloud_light()`` at call sites (secrets may load later).

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
]

NLP_FUNCTIONS = NLP_FUNCTIONS_LIGHT if get_cloud_light() else NLP_FUNCTIONS_FULL


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
MAX_POS_CONTENT_CHARS = _env_int("MAX_POS_CONTENT_CHARS", 5000)
ARTICLE_CACHE_ENABLED = os.environ.get("ARTICLE_CACHE", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Used by scripts/scraper_health_check.py for optional per-source smoke URLs.
SCRAPE_SAMPLE_URLS: dict[str, str] = {
    "NV": "https://nv.ua/ukr/",
    "Радіо Свобода": "https://www.radiosvoboda.org/a/",
    "Українська правда": "https://www.pravda.com.ua/news/",
    "Liga.net": "https://news.liga.net/ua/",
    "RBC-UA": "https://www.rbc.ua/ukr/",
    "Інтерфакс-Україна": "https://interfax.com.ua/news/",
    "TSN": "https://tsn.ua/",
    "УНІАН": "https://www.unian.ua/",
}
