"""Tests for structured logging configuration."""

from __future__ import annotations

import json
import logging

import pytest

from synth_panel.logging_config import setup_logging


@pytest.fixture(autouse=True)
def _reset_logger():
    """Remove handlers added by setup_logging between tests."""
    logger = logging.getLogger("synth_panel")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    yield
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)


def test_default_plaintext(monkeypatch, capsys):
    monkeypatch.delenv("SYNTHPANEL_LOG_FORMAT", raising=False)
    setup_logging("debug")
    logger = logging.getLogger("synth_panel")
    logger.info("hello world")
    captured = capsys.readouterr()
    assert "hello world" in captured.err
    # Should NOT be valid JSON
    for line in captured.err.strip().splitlines():
        with pytest.raises(json.JSONDecodeError):
            json.loads(line)


def test_json_format(monkeypatch, capsys):
    monkeypatch.setenv("SYNTHPANEL_LOG_FORMAT", "json")
    setup_logging("debug")
    logger = logging.getLogger("synth_panel")
    logger.info("test message")
    captured = capsys.readouterr()
    line = captured.err.strip()
    obj = json.loads(line)
    assert obj["level"] == "INFO"
    assert obj["logger"] == "synth_panel"
    assert obj["message"] == "test message"
    assert "timestamp" in obj


def test_json_format_case_insensitive(monkeypatch, capsys):
    monkeypatch.setenv("SYNTHPANEL_LOG_FORMAT", "JSON")
    setup_logging("debug")
    logger = logging.getLogger("synth_panel")
    logger.warning("warn msg")
    captured = capsys.readouterr()
    obj = json.loads(captured.err.strip())
    assert obj["level"] == "WARNING"
    assert obj["message"] == "warn msg"


def test_json_timestamp_is_iso(monkeypatch, capsys):
    monkeypatch.setenv("SYNTHPANEL_LOG_FORMAT", "json")
    setup_logging("debug")
    logger = logging.getLogger("synth_panel")
    logger.info("ts check")
    captured = capsys.readouterr()
    obj = json.loads(captured.err.strip())
    # ISO 8601 with timezone offset
    assert obj["timestamp"].endswith("+00:00")


def test_reconfigure_switches_format(monkeypatch, capsys):
    """Calling setup_logging again with a different format updates the handler."""
    monkeypatch.setenv("SYNTHPANEL_LOG_FORMAT", "json")
    setup_logging("debug")
    logger = logging.getLogger("synth_panel")
    logger.info("json line")
    captured = capsys.readouterr()
    json.loads(captured.err.strip())  # should parse

    monkeypatch.delenv("SYNTHPANEL_LOG_FORMAT")
    setup_logging("debug")
    logger.info("text line")
    captured = capsys.readouterr()
    assert "text line" in captured.err
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.err.strip())
