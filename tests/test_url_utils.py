"""Tests for SSRF URL validation."""

from url_utils import (
    _registrable_domain,
    get_allowed_domains,
    is_allowed_url,
)


def test_allowed_domains_includes_configured_media():
    get_allowed_domains.cache_clear()
    domains = get_allowed_domains()
    assert "nv.ua" in domains
    assert "pravda.com.ua" in domains
    assert "unian.ua" in domains or "rss.unian.ua" in domains
    # Public-suffix truncation must never allow the bare multi-part TLD.
    assert "com.ua" not in domains
    assert "org.ua" not in domains


def test_registrable_domain_respects_multipart_ua_suffix():
    assert _registrable_domain("pravda.com.ua") == "pravda.com.ua"
    assert _registrable_domain("news.pravda.com.ua") == "pravda.com.ua"
    assert _registrable_domain("rss.unian.ua") == "unian.ua"
    assert _registrable_domain("com.ua") is None
    assert _registrable_domain("assets.censor.net") == "censor.net"


def test_allowed_domains_includes_category_media():
    get_allowed_domains.cache_clear()
    domains = get_allowed_domains()
    assert "dou.ua" in domains
    assert "epravda.com.ua" in domains
    assert "football.ua" in domains
    assert "ain.ua" in domains
    assert "champion.com.ua" in domains


def test_blocks_evil_com_ua_despite_pravda(monkeypatch):
    """Registrable-domain heuristic must not treat com.ua as trusted."""
    monkeypatch.setattr(
        "url_utils._resolve_host_is_public",
        lambda hostname: True,
    )
    assert is_allowed_url("https://evil.com.ua/steal") is False
    assert is_allowed_url("https://www.pravda.com.ua/news/1/") is True


def test_blocks_file_scheme():
    assert is_allowed_url("file:///etc/passwd") is False


def test_blocks_localhost(monkeypatch):
    monkeypatch.setattr(
        "url_utils._resolve_host_is_public",
        lambda hostname: False,
    )
    assert is_allowed_url("http://127.0.0.1/news/1") is False


def test_allows_unian_article_url(monkeypatch):
    monkeypatch.setattr(
        "url_utils._resolve_host_is_public",
        lambda hostname: True,
    )
    url = "https://www.unian.ua/pogoda/news/test-13447251.html"
    assert is_allowed_url(url) is True


def test_blocks_unknown_domain(monkeypatch):
    monkeypatch.setattr(
        "url_utils._resolve_host_is_public",
        lambda hostname: True,
    )
    assert is_allowed_url("https://evil.example.com/article") is False


def test_blocks_empty_url():
    assert is_allowed_url("") is False
    assert is_allowed_url(None) is False  # type: ignore[arg-type]


def test_blocks_malformed_url():
    assert is_allowed_url("not a valid url :::") is False


def test_hostname_matches_subdomain(monkeypatch):
    monkeypatch.setattr(
        "url_utils._resolve_host_is_public",
        lambda hostname: True,
    )
    assert is_allowed_url("https://news.nv.ua/article/1") is True


def test_dns_oserror_blocks_url(monkeypatch):

    def raise_oserror(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("url_utils.socket.getaddrinfo", raise_oserror)
    monkeypatch.setattr(
        "url_utils._hostname_matches_allowlist",
        lambda hostname, allowed: True,
    )
    assert is_allowed_url("https://www.unian.ua/news/1") is False
