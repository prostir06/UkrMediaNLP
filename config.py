"""
Application configuration: Ukrainian news sources, NLP functions, and UI text.
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


DEFAULT_REQUEST_TIMEOUT = 10

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

NEWS_SOURCES = {
    "NV": {
        "rss_url": "https://nv.ua/ukr/rss/all.xml",
        "scraper": "nv",
        "intro": """
**NV** — українське мультимедійне видання, що висвітлює політику, економіку,
технології та суспільні теми. Один із найпопулярніших новинних порталів України.
""",
    },
    "Радіо Свобода": {
        "rss_url": "https://www.radiosvoboda.org/api/zrqitl-vomx-tpeoumq",
        "scraper": "radiosvoboda",
        "intro": """
**Радіо Свобода** — міжнародне мультимедійне видання, яке висвітлює події в Україні
та регіоні. Частина мережі Radio Free Europe / Radio Liberty.
""",
    },
    "Українська правда": {
        "rss_url": "https://www.pravda.com.ua/rss/view_mainnews/",
        "scraper": "pravda",
        "intro": """
**Українська правда** — інтернет-видання, засноване у 2000 році. Один із найвідоміших
незалежних медіаресурсів України з фокусом на політику та суспільство.
""",
    },
    "Liga.net": {
        "rss_url": "https://news.liga.net/ua/top/rss.xml",
        "scraper": "liga",
        "intro": """
**Liga.net** — новинний портал з акцентом на бізнес, економіку, політику
та аналітику. Входить до медіахолдингу Liga.net.
""",
    },
    "RBC-UA": {
        "rss_url": "https://www.rbc.ua/static/rss/ukrnet.strong.ukr.rss.xml",
        "scraper": "rbc",
        "intro": """
**RBC-UA** — українське видання міжнародної мережі RBK, що висвітлює економічні
та політичні новини України.
""",
    },
    "Інтерфакс-Україна": {
        "rss_url": "https://interfax.com.ua/news/last.rss",
        "scraper": "interfax",
        "intro": """
**Інтерфакс-Україна** — інформаційне агентство, яке оперативно публікує новини
з політики, економіки та суспільного життя.
""",
    },
    "TSN": {
        "rss_url": "https://tsn.ua/rss/full.rss",
        "scraper": "tsn",
        "intro": """
**TSN** — телеканал «1+1» та новинний портал tsn.ua. Охоплює політику, події,
спорт, шоу-бізнес та lifestyle.
""",
    },
    "УНІАН": {
        "rss_url": "https://rss.unian.ua/site/news_ukr.rss",
        "scraper": "unian",
        "intro": """
**УНІАН** — одне з найбільших інформаційних агентств України. Публікує новини
політики, економіки, технологій та регіональних подій українською мовою.
""",
    },
}

def _transformers_available() -> bool:
    """Return True when the transformers package is importable."""
    try:
        import importlib.util

        return importlib.util.find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


# Light UI when LIGHT_CLOUD=1 or when transformers is not installed (Cloud light deps).
CLOUD_LIGHT_MODE = (
    os.environ.get("LIGHT_CLOUD", "").lower() in {"1", "true", "yes"}
    or not _transformers_available()
)

NLP_FUNCTIONS_FULL = [
    "Вступ",
    "Огляд статей",
    "Уніграми",
    "Біграми",
    "Триграми",
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
    "Хмара слів",
    "Статистика тексту",
    "Тематичне моделювання",
    "Розпізнавання сутностей",
    "Тональність (новини)",
    "Сумаризація",
    "Частини мови",
    "Порівняння медіа",
]

NLP_FUNCTIONS = NLP_FUNCTIONS_LIGHT if CLOUD_LIGHT_MODE else NLP_FUNCTIONS_FULL

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
