"""
URL validation utilities to mitigate SSRF when fetching RSS article links.

Security model
--------------
1. Only ``http`` and ``https`` schemes are permitted.
2. Hostnames must match the allowlist derived from ``NEWS_SOURCES`` RSS URLs.
3. DNS resolution must not return private, loopback, or link-local addresses.
4. Redirect targets are re-validated after the HTTP response is received.
"""

import ipaddress
import logging
import socket
from functools import lru_cache
from urllib.parse import urlparse

from config import NEWS_SOURCES

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = frozenset({"http", "https"})
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MB


def _hostname_from_url(url: str) -> str:
    """
    Extract a normalised hostname from a URL string.

    Strips a leading ``www.`` prefix so ``www.unian.ua`` matches ``unian.ua``.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        logger.debug("Cannot parse URL hostname: %s", exc)
        return ""

    hostname = (parsed.hostname or "").lower().strip(".")
    if hostname.startswith("www."):
        return hostname[4:]
    return hostname


@lru_cache(maxsize=1)
def get_allowed_domains() -> frozenset[str]:
    """
    Build the hostname allowlist from configured RSS feed URLs.

    Covers every sidebar category (Новини / Економіка / Спорт / Технології).
    Includes both full hostnames (e.g. ``rss.unian.ua``) and registrable
    domains (e.g. ``unian.ua``) plus known article-page subdomains.
    """
    domains: set[str] = set()
    try:
        source_configs = list(NEWS_SOURCES.values())
    except (AttributeError, TypeError) as exc:
        logger.error("NEWS_SOURCES unavailable for allowlist: %s", exc)
        source_configs = []

    for config in source_configs:
        try:
            if not isinstance(config, dict):
                continue
            hostname = _hostname_from_url(str(config.get("rss_url", "")))
        except (KeyError, TypeError, AttributeError) as exc:
            logger.warning("Invalid NEWS_SOURCES entry skipped: %s", exc)
            continue
        if hostname:
            domains.add(hostname)
            parts = hostname.split(".")
            if len(parts) >= 2:
                domains.add(".".join(parts[-2:]))

    # Article pages may live on subdomains not present in RSS URLs.
    extra = {
        "nv.ua",
        "sport.nv.ua",
        "biz.nv.ua",
        "radiosvoboda.org",
        "pravda.com.ua",
        "epravda.com.ua",
        "liga.net",
        "news.liga.net",
        "rbc.ua",
        "interfax.com.ua",
        "tsn.ua",
        "unian.ua",
        "unian.net",
        "rss.unian.ua",
        "rss.unian.net",
        "champion.com.ua",
        "football.ua",
        "suspilne.media",
        "tribuna.com",
        "ua.tribuna.com",
        "rss.ua.tribuna.com",
        "censor.net",
        "biz.censor.net",
        "assets.censor.net",
        "tech.liga.net",
        "itc.ua",
        "feeds.feedburner.com",
        "dou.ua",
        "mezha.ua",
        "dev.ua",
        "speka.ua",
        "vctr.media",
        "ain.ua",
    }
    domains.update(extra)
    return frozenset(domains)


def _is_private_ip(ip_str: str) -> bool:
    """Return True when *ip_str* is private, loopback, or otherwise unsafe."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Unparseable values are treated as unsafe.
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def _resolve_host_is_public(hostname: str) -> bool:
    """
    Return False when any resolved address is private or reserved.

    DNS failures also return False so unresolved hostnames are never fetched.
    """
    if not hostname:
        return False

    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        ):
            if family in (socket.AF_INET, socket.AF_INET6):
                if _is_private_ip(sockaddr[0]):
                    return False
    except socket.gaierror as exc:
        logger.warning("DNS resolution failed for %s: %s", hostname, exc)
        return False
    except OSError as exc:
        logger.warning("DNS lookup error for %s: %s", hostname, exc)
        return False
    return True


def _hostname_matches_allowlist(hostname: str, allowed: frozenset[str]) -> bool:
    """Check exact match or subdomain suffix against the allowlist."""
    return hostname in allowed or any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in allowed
    )


def is_allowed_url(url: str) -> bool:
    """
    Validate that a URL is safe to fetch.

    Returns:
        True when scheme, domain, and DNS resolution all pass SSRF checks.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url.strip())
    except ValueError as exc:
        logger.warning("Blocked malformed URL: %s", exc)
        return False

    if parsed.scheme not in ALLOWED_SCHEMES:
        logger.warning("Blocked URL with disallowed scheme: %s", url)
        return False

    hostname = _hostname_from_url(url)
    if not hostname:
        logger.warning("Blocked URL without hostname: %s", url)
        return False

    allowed = get_allowed_domains()
    if not _hostname_matches_allowlist(hostname, allowed):
        logger.warning("Blocked URL with disallowed host: %s", hostname)
        return False

    if not _resolve_host_is_public(hostname):
        logger.warning("Blocked URL resolving to private/reserved IP: %s", hostname)
        return False

    return True
