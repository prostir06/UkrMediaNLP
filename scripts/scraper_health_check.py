"""Weekly smoke test: verify at least one article scrapes per media source."""

import sys

from config import NEWS_SOURCES, SCRAPE_SAMPLE_URLS
from data_loader import fetch_articles

MIN_SUCCESS_RATE = 0.5


def main() -> int:
    print("Sample landing URLs (debug):")
    for name, url in SCRAPE_SAMPLE_URLS.items():
        print(f"  {name}: {url}")

    failures = []
    for source_name, config in NEWS_SOURCES.items():
        df = fetch_articles(
            source_name=source_name,
            feed_url=config["rss_url"],
            scraper_name=config.get("scraper", "generic"),
            max_articles=5,
        )
        if df.empty:
            failures.append(f"{source_name}: no articles")
            continue

        rate = float(df["scraped_ok"].mean())
        print(f"{source_name}: scraped_ok={rate:.0%} ({len(df)} articles)")
        if rate < MIN_SUCCESS_RATE:
            failures.append(
                f"{source_name}: success rate {rate:.0%} < {MIN_SUCCESS_RATE:.0%}"
            )

    if failures:
        print("Scraper health check FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Scraper health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
