"""Tests for SSRF URL validation."""

import socket

import pytest

from url_utils import get_allowed_domains, is_allowed_url


def test_allowed_domains_includes_configured_media():
    domains = get_allowed_domains()
    assert "nv.ua" in domains
    assert "pravda.com.ua" in domains
    assert "unian.ua" in domains or "rss.unian.ua" in domains


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
    import socket

    def raise_oserror(*args, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("url_utils.socket.getaddrinfo", raise_oserror)
    monkeypatch.setattr(
        "url_utils._hostname_matches_allowlist",
        lambda hostname, allowed: True,
    )
    assert is_allowed_url("https://www.unian.ua/news/1") is False
