"""Ukrainian media registry: categories, sources, and scrape sample URLs."""

from __future__ import annotations

import logging
from typing import TypedDict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class NewsSourceConfig(TypedDict):
    """Schema for one NEWS_SOURCES entry."""

    category: str
    rss_url: str
    scraper: str
    intro: str


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


def _sample_url_from_rss(rss_url: str) -> str:
    parsed = urlparse(str(rss_url))
    if not parsed.scheme or not parsed.netloc:
        return str(rss_url)
    return f"{parsed.scheme}://{parsed.netloc}/"


# Prefer human landing pages for health-check debug output; fall back to RSS origin.
_SCRAPE_SAMPLE_OVERRIDES: dict[str, str] = {
    "NV": "https://nv.ua/ukr/",
    "Радіо Свобода": "https://www.radiosvoboda.org/a/",
    "Українська правда": "https://www.pravda.com.ua/news/",
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


SCRAPE_SAMPLE_URLS: dict[str, str] = {
    name: _SCRAPE_SAMPLE_OVERRIDES.get(
        name,
        _sample_url_from_rss(str(config.get("rss_url", ""))),
    )
    for name, config in NEWS_SOURCES.items()
}

