"""Unit tests for runtime_env defaults."""

import os

from runtime_env import apply_runtime_env


def test_apply_runtime_env_sets_defaults(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_DISK_LOCK", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("QUANTIZE_CPU", raising=False)
    monkeypatch.delenv("HF_HUB_DOWNLOAD_TIMEOUT", raising=False)
    monkeypatch.delenv("ALLOW_HEAVY_NLP", raising=False)

    apply_runtime_env()

    assert os.environ["HF_HUB_DISABLE_DISK_LOCK"] == "1"
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["QUANTIZE_CPU"] == "0"
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "120"
    assert os.environ["ALLOW_HEAVY_NLP"] == "0"


def test_apply_runtime_env_does_not_override(monkeypatch):
    monkeypatch.setenv("QUANTIZE_CPU", "1")
    monkeypatch.setenv("ALLOW_HEAVY_NLP", "1")
    apply_runtime_env()
    assert os.environ["QUANTIZE_CPU"] == "1"
    assert os.environ["ALLOW_HEAVY_NLP"] == "1"
