"""
Ukrainian media registry: categories, sources, and scrape sample URLs.

This module is Streamlit-free. ``NEWS_SOURCES`` is the canonical registry;
``config`` re-exports it for backward-compatible imports.

Error policy for helpers:
* Soft-fail → empty list / ``None`` / skip entry (UI stays usable).
* Hard-fail → ``KeyError`` from ``get_source_config`` for unknown names.
* Schema validation never raises; it returns a problem list.

HTML5 / CSS3 / StandardJS do not apply here (Python / PEP 8 only).
"""

from __future__ import annotations

import logging
from typing import TypedDict
from urllib.parse import urlparse

# Module-level logger for soft-fail paths (malformed registry entries).
logger = logging.getLogger(__name__)


class NewsSourceConfig(TypedDict):
    """
    Typed schema for one ``NEWS_SOURCES`` entry.

    All four keys are required at runtime (see ``REQUIRED_SOURCE_KEYS``).
    """

    category: str
    rss_url: str
    scraper: str
    intro: str


# Sidebar category labels; order is UI order. Must match NEWS_SOURCES[*].category.
MEDIA_CATEGORIES = (
    "Новини",
    "Економіка",
    "Спорт",
    "Технології",
)

# Keys every source dict must provide (validated by validate_news_sources_schema).
REQUIRED_SOURCE_KEYS = frozenset({"category", "rss_url", "scraper", "intro"})


# ---------------------------------------------------------------------------
# Canonical media registry (28 sources after removing «Українська правда»).
# Each entry: category, RSS URL, scraper registry name, markdown intro.
# ---------------------------------------------------------------------------
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

    Args:
        category: Sidebar label such as ``"Новини"``, or ``None``.

    Returns:
        Ordered list of source names (registry insertion order).
    """
    # Reject non-strings early so callers can pass widget values blindly.
    if not isinstance(category, str):
        return []
    # Normalize whitespace so ``" Новини "`` still matches.
    category = category.strip()
    if not category:
        return []

    names: list[str] = []
    try:
        # Snapshot items so concurrent mutation (tests) cannot break iteration.
        items = list(NEWS_SOURCES.items())
    except (AttributeError, TypeError) as exc:
        logger.warning("NEWS_SOURCES is not iterable: %s", exc)
        return []

    for name, config in items:
        try:
            # Skip corrupt entries instead of failing the whole sidebar.
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
        KeyError: Unknown source or non-dict entry (callers treat both as missing).
    """
    try:
        config = NEWS_SOURCES[source_name]
    except KeyError:
        # Preserve KeyError so UI can show «unknown source».
        raise
    except TypeError as exc:
        # NEWS_SOURCES replaced with a non-mapping in a broken test monkeypatch.
        raise KeyError(source_name) from exc

    if not isinstance(config, dict):
        raise KeyError(source_name)
    return config


def source_category(source_name: str) -> str | None:
    """
    Return the sidebar category for *source_name*, or ``None`` if unknown.

    Soft-fails on any lookup / type error so routers never crash on a bad name.
    """
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

    Args:
        sources: Optional override registry (tests). Defaults to ``NEWS_SOURCES``.
    """
    problems: list[str] = []
    registry = NEWS_SOURCES if sources is None else sources
    try:
        items = list(registry.items())
    except (AttributeError, TypeError) as exc:
        return [f"NEWS_SOURCES is not a mapping: {exc}"]

    known_categories = set(MEDIA_CATEGORIES)
    for name, config in items:
        try:
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
            if not isinstance(rss_url, str) or not rss_url.startswith(
                ("http://", "https://")
            ):
                problems.append(f"{name}: rss_url must be an http(s) URL")
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"{name}: validation crashed: {exc}")
    return problems


def _sample_url_from_rss(rss_url: str) -> str:
    """
    Derive a site origin URL from an RSS feed URL for health-check debug.

    Returns the original string when parsing fails or scheme/host are missing.
    """
    try:
        parsed = urlparse(str(rss_url or ""))
    except Exception as exc:
        logger.debug("_sample_url_from_rss parse failed for %r: %s", rss_url, exc)
        return str(rss_url or "")
    if not parsed.scheme or not parsed.netloc:
        return str(rss_url or "")
    return f"{parsed.scheme}://{parsed.netloc}/"


# Prefer human landing pages for health-check debug output; fall back to RSS origin.
_SCRAPE_SAMPLE_OVERRIDES: dict[str, str] = {
    "NV": "https://nv.ua/ukr/",
    "Радіо Свобода": "https://www.radiosvoboda.org/a/",
    "Liga.net": "https://news.liga.net/ua/",
    "RBC-UA": "https://www.rbc.ua/ukr/",
    "Інтерфакс-Україна": "https://interfax.com.ua/news/",
    "TSN": "https://tsn.ua/",
    "УНІАН": "https://www.unian.ua/",
    "Економічна правда": "https://epravda.com.ua/",
    "Бізнес Цензор": "https://biz.censor.net/",
    "NV (Економіка)": "https://nv.ua/ukr/economy.html",
    "Радіо Свобода (Економіка)": "https://www.radiosvoboda.org/",
    "Liga.net (Економіка)": "https://news.liga.net/economics/",
    "NV (Спорт)": "https://nv.ua/ukr/sport.html",
    "Радіо Свобода (Спорт)": "https://www.radiosvoboda.org/",
    "Liga.net (Спорт)": "https://news.liga.net/sport/",
    "Champion": "https://champion.com.ua/",
    "Football.ua": "https://football.ua/",
    "Суспільне Спорт": "https://suspilne.media/sport/",
    "Tribuna": "https://ua.tribuna.com/",
    "NV (Технології)": "https://nv.ua/ukr/techno.html",
    "Liga.net (Технології)": "https://tech.liga.net/",
    "ITC.ua": "https://itc.ua/",
    "DOU": "https://dou.ua/",
    "Mezha": "https://mezha.ua/",
    "dev.ua": "https://dev.ua/",
    "Speka": "https://speka.ua/",
    "Vector": "https://vctr.media/",
    "AIN.UA": "https://ain.ua/",
}


def build_scrape_sample_urls(
    sources: dict | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Build ``{source_name: landing_or_origin_url}`` for scraper health debug.

    Soft-skips sources whose config cannot supply a usable URL.
    """
    registry = NEWS_SOURCES if sources is None else sources
    mapping = _SCRAPE_SAMPLE_OVERRIDES if overrides is None else overrides
    result: dict[str, str] = {}
    try:
        items = list(registry.items())
    except (AttributeError, TypeError) as exc:
        logger.warning("build_scrape_sample_urls: bad registry: %s", exc)
        return {}

    for name, config in items:
        try:
            if name in mapping:
                result[str(name)] = str(mapping[name])
                continue
            if not isinstance(config, dict):
                logger.warning("sample URL skip non-dict source %r", name)
                continue
            rss = str(config.get("rss_url", "") or "")
            derived = _sample_url_from_rss(rss)
            if derived:
                result[str(name)] = derived
        except Exception as exc:
            logger.warning("sample URL skip %r: %s", name, exc)
            continue
    return result


# Eager map used by health scripts and tests (rebuilt via build_scrape_sample_urls).
try:
    SCRAPE_SAMPLE_URLS: dict[str, str] = build_scrape_sample_urls()
except Exception as exc:  # pragma: no cover - registry is static at import
    logger.error("SCRAPE_SAMPLE_URLS init failed: %s", exc)
    SCRAPE_SAMPLE_URLS = {}

