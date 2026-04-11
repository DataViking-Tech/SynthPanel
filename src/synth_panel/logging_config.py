"""Structured logging configuration for synthpanel.

Provides a single ``setup_logging`` entry point that configures the
Python logging hierarchy for the ``synth_panel`` namespace.  Verbosity
is resolved from (in priority order):

1. Explicit *verbosity* argument (``"debug"`` / ``"warning"``).
2. ``SYNTHPANEL_LOG_LEVEL`` environment variable.
3. Default: ``INFO``.

Output format is controlled by ``SYNTHPANEL_LOG_FORMAT``:

- ``"text"`` (default): human-readable plaintext lines.
- ``"json"``: one JSON object per line with keys
  ``level``, ``logger``, ``timestamp``, ``message``.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

_VERBOSITY_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "info": logging.INFO,
}


class _JSONFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "logger": record.name,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


def _make_formatter() -> logging.Formatter:
    fmt = os.environ.get("SYNTHPANEL_LOG_FORMAT", "").strip().lower()
    if fmt == "json":
        return _JSONFormatter()
    return logging.Formatter(_LOG_FORMAT)


def setup_logging(verbosity: str | None = None) -> None:
    """Configure logging for the ``synth_panel`` package.

    Args:
        verbosity: One of ``"debug"``, ``"info"``, ``"warning"``.
            When *None*, falls back to the ``SYNTHPANEL_LOG_LEVEL``
            environment variable (case-insensitive), then to ``INFO``.
    """
    if verbosity is not None:
        level = _VERBOSITY_MAP.get(verbosity.lower(), logging.INFO)
    else:
        env = os.environ.get("SYNTHPANEL_LOG_LEVEL", "").strip().lower()
        level = _VERBOSITY_MAP.get(env, logging.INFO)

    formatter = _make_formatter()

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger("synth_panel")
    logger.setLevel(level)
    # Avoid duplicate handlers if called more than once.
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)
        logger.setLevel(level)
