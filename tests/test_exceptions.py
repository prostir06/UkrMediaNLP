"""Tests for custom application exceptions."""

from exceptions import (
    DataLoaderError,
    NLPAnalysisError,
    RSSFeedError,
    ScrapingError,
)


def test_rss_feed_error_stores_feed_url():
    error = RSSFeedError("Invalid feed", feed_url="https://example.com/rss")
    assert str(error) == "Invalid feed"
    assert error.feed_url == "https://example.com/rss"


def test_scraping_error_stores_url():
    error = ScrapingError("Parse failed", url="https://example.com/news/1")
    assert error.url == "https://example.com/news/1"


def test_nlp_analysis_error_stores_step():
    error = NLPAnalysisError("Model missing", step="spacy_load")
    assert error.step == "spacy_load"


def test_data_loader_error_stores_source_name():
    error = DataLoaderError("Unknown source", source_name="TestMedia")
    assert error.source_name == "TestMedia"
