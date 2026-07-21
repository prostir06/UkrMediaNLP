"""Tests for JSON-LD structured data scraper."""

from scrapers.structured import StructuredDataScraper, _extract_body_from_item


def test_structured_scraper_extracts_article_body():
    body_text = (
        "Uryad Ukrainy ukhvalyv novyy zakon pro enerhetychnu bezpeku krayiny. "
        "Dokument vyznachaie zakhody dlia zmitsnennia enerhetychnoi nezalezhnosti."
    )
    html = f"""
    <!DOCTYPE html>
    <html lang="uk">
    <head>
      <script type="application/ld+json">
      {{
        "@type": "NewsArticle",
        "articleBody": "{body_text}"
      }}
      </script>
    </head>
    <body></body>
    </html>
    """.encode("utf-8")

    text = StructuredDataScraper().extract(html, "https://example.com/a")
    assert "enerhetychnu bezpeku" in text


def test_structured_scraper_returns_empty_for_missing_jsonld():
    html = b"<!DOCTYPE html><html lang='uk'><body><p>Plain text only.</p></body></html>"
    assert StructuredDataScraper().extract(html, "https://example.com/b") == ""


def test_structured_scraper_handles_invalid_json():
    html = b"""
    <!DOCTYPE html>
    <html><head>
      <script type="application/ld+json">{ broken json</script>
    </head></html>
    """
    assert StructuredDataScraper().extract(html, "https://example.com/c") == ""


def test_structured_scraper_parses_graph_container():
    body = "A" * 81
    html = f"""
    <!DOCTYPE html>
    <html><head>
      <script type="application/ld+json">
      {{
        "@graph": [
          {{"@type": "WebPage", "name": "index"}},
          {{"@type": "NewsArticle", "articleBody": "{body}"}}
        ]
      }}
      </script>
    </head></html>
    """.encode("utf-8")

    text = StructuredDataScraper().extract(html, "https://example.com/d")
    assert len(text) > 80


def test_extract_body_rejects_short_description():
    item = {"@type": "NewsArticle", "description": "Too short"}
    assert _extract_body_from_item(item) == ""


def test_extract_body_rejects_wrong_type():
    item = {"@type": "Organization", "description": "X" * 100}
    assert _extract_body_from_item(item) == ""


def test_structured_scraper_handles_empty_html():
    assert StructuredDataScraper().extract(b"", "https://example.com/e") == ""
