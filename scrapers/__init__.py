"""Scraper registry for Ukrainian media sources."""

from scrapers.generic import GenericScraper
from scrapers.structured import StructuredDataScraper
from scrapers.site_scrapers import (
    INTERFAX_SCRAPER,
    LIGA_SCRAPER,
    NV_SCRAPER,
    PRAVDA_SCRAPER,
    RADIOSVOBODA_SCRAPER,
    RBC_SCRAPER,
    TSN_SCRAPER,
    UNIAN_SCRAPER,
)

GENERIC_SCRAPER = GenericScraper()
STRUCTURED_SCRAPER = StructuredDataScraper()

SCRAPER_REGISTRY = {
    "nv": NV_SCRAPER,
    "radiosvoboda": RADIOSVOBODA_SCRAPER,
    "pravda": PRAVDA_SCRAPER,
    "liga": LIGA_SCRAPER,
    "rbc": RBC_SCRAPER,
    "interfax": INTERFAX_SCRAPER,
    "tsn": TSN_SCRAPER,
    "unian": UNIAN_SCRAPER,
}


def get_scraper(name: str):
    return SCRAPER_REGISTRY.get(name, GENERIC_SCRAPER)
