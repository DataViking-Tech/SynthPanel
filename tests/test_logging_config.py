"""Tests for althing.logging_config (CLI logging behavior)."""

from __future__ import annotations

import logging

import pytest

from althing.logging_config import _NOISY_LOGGERS, setup_logging


@pytest.fixture(autouse=True)
def _reset_loggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from each other without changing package logger wiring."""
    del logging.root.handlers[:]

    monkeypatch.delenv("ALTHING_LOG_LEVEL", raising=False)

    synth = logging.getLogger("althing")
    # Clear handlers althing attaches in setup_logging between tests.
    while synth.handlers:
        synth.removeHandler(synth.handlers[-1])

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)


def test_default_info_synth_and_noisy_warning() -> None:
    setup_logging()
    assert logging.getLogger("althing").getEffectiveLevel() == logging.INFO
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_verbose_debug_synth_only_noisy_stays_warning() -> None:
    setup_logging("debug")
    assert logging.getLogger("althing").getEffectiveLevel() == logging.DEBUG
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING


def test_debug_all_elevates_noisy_libraries() -> None:
    setup_logging(debug_all=True)
    assert logging.getLogger("althing").getEffectiveLevel() == logging.DEBUG
    for name in _NOISY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.DEBUG


def test_env_syn_log_level_debug_does_not_flood_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALTHING_LOG_LEVEL", "debug")
    setup_logging()
    assert logging.getLogger("althing").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
