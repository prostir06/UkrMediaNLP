"""
Custom application exceptions.

Each exception represents a recoverable or reportable failure domain so callers
can catch specific error types instead of broad ``Exception``.
"""


class RSSFeedError(Exception):
    """Raised when an RSS feed cannot be fetched or parsed."""

    def __init__(self, message: str, feed_url: str = "") -> None:
        super().__init__(message)
        self.feed_url = feed_url


class ScrapingError(Exception):
    """Raised when article HTML cannot be downloaded or parsed."""

    def __init__(self, message: str, url: str = "") -> None:
        super().__init__(message)
        self.url = url


class NLPAnalysisError(Exception):
    """Raised when an NLP model or analysis step fails."""

    def __init__(self, message: str, step: str = "") -> None:
        super().__init__(message)
        self.step = step


class DataLoaderError(Exception):
    """Raised when article loading fails for configuration or pipeline reasons."""

    def __init__(self, message: str, source_name: str = "") -> None:
        super().__init__(message)
        self.source_name = source_name
