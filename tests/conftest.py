"""Pytest configuration — disable persistent article cache during unit tests."""

import os

os.environ.setdefault("ARTICLE_CACHE", "0")
