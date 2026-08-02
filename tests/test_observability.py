"""Tests for structured step logging."""

import logging

import pytest

from observability import log_step


def test_log_step_emits_structured_fields(caplog):
    logger = logging.getLogger("test.observability")
    with caplog.at_level(logging.INFO, logger="test.observability"):
        with log_step(logger, step="scrape", source="NV"):
            pass

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "step=scrape" in message
    assert "source=NV" in message
    assert "status=ok" in message
    assert "elapsed_ms=" in message


def test_log_step_marks_error_and_reraises(caplog):
    logger = logging.getLogger("test.observability.error")
    with caplog.at_level(logging.INFO, logger="test.observability.error"):
        with pytest.raises(RuntimeError, match="boom"):
            with log_step(logger, step="corpus_load", source="A"):
                raise RuntimeError("boom")

    message = caplog.records[0].getMessage()
    assert "step=corpus_load" in message
    assert "status=error" in message
