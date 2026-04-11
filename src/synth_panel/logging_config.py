"""Structured logging configuration for synthpanel.

Provides a single ``setup_logging`` entry point that configures the
Python logging hierarchy for the ``synth_panel`` namespace.  Verbosity
is resolved from (in priority order):

1. Explicit *verbosity* argument (``"debug"`` / ``"warning"``).
2. ``SYNTHPANEL_LOG_LEVEL`` environment variable.
3. Default: ``INFO``.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

_VERBOSITY_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "info": logging.INFO,
}


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

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    logger = logging.getLogger("synth_panel")
    logger.setLevel(level)
    # Avoid duplicate handlers if called more than once.
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.setLevel(level)
