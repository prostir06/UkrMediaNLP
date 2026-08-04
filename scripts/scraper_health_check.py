"""Weekly / on-demand scraper health check against live RSS."""

from __future__ import annotations

import os
import sys
import time

from data_loader import fetch_articles
from media_sources import NEWS_SOURCES, SCRAPE_SAMPLE_URLS

MIN_SUCCESS_RATE = 0.5
DEFAULT_SUBSET_SIZE = 5
MAX_RETRIES = 2


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _selected_sources() -> dict:
    """Full registry when SCRAPER_HEALTH_FULL=1; otherwise a small stable subset."""
    if _truthy("SCRAPER_HEALTH_FULL"):
        return dict(NEWS_SOURCES)
    # Prefer named sample overrides, then fill from registry order.
    names: list[str] = []
    for name in SCRAPE_SAMPLE_URLS:
        if name in NEWS_SOURCES and name not in names:
            names.append(name)
        if len(names) >= DEFAULT_SUBSET_SIZE:
            break
    if len(names) < DEFAULT_SUBSET_SIZE:
        for name in NEWS_SOURCES:
            if name not in names:
                names.append(name)
            if len(names) >= DEFAULT_SUBSET_SIZE:
                break
    return {name: NEWS_SOURCES[name] for name in names}


def _check_source(source_name: str, config: dict) -> str | None:
    """Return failure message or None on success. Retries transient empty/low rates."""
    last_error = f"{source_name}: unknown failure"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = fetch_articles(
                source_name=source_name,
                feed_url=config["rss_url"],
                scraper_name=config.get("scraper", "generic"),
                max_articles=5,
            )
        except Exception as exc:  # noqa: BLE001 — health script soft-fails per source
            last_error = f"{source_name}: fetch error ({exc})"
            time.sleep(attempt)
            continue

        if df.empty:
            last_error = f"{source_name}: no articles"
            time.sleep(attempt)
            continue

        rate = float(df["scraped_ok"].mean())
        print(f"{source_name}: scraped_ok={rate:.0%} ({len(df)} articles, try={attempt})")
        if rate < MIN_SUCCESS_RATE:
            last_error = (
                f"{source_name}: success rate {rate:.0%} < {MIN_SUCCESS_RATE:.0%}"
            )
            time.sleep(attempt)
            continue
        return None
    return last_error


def main() -> int:
    full = _truthy("SCRAPER_HEALTH_FULL")
    sources = _selected_sources()
    mode = "full" if full else f"subset({len(sources)})"
    print(f"Scraper health mode={mode}")
    print("Sample landing URLs (debug):")
    for name in sources:
        url = SCRAPE_SAMPLE_URLS.get(name, "")
        if url:
            print(f"  {name}: {url}")

    failures = []
    for source_name, config in sources.items():
        err = _check_source(source_name, config)
        if err:
            failures.append(err)

    if failures:
        print("Scraper health check FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Scraper health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
