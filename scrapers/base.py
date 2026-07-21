"""Base scraper interface."""

from abc import ABC, abstractmethod


class BaseScraper(ABC):
    """Extract main article text from an HTML page."""

    @abstractmethod
    def extract(self, html: bytes, url: str) -> str:
        """Return plain article text or empty string."""
