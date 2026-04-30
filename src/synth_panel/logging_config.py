"""Structured logging configuration for synthpanel.

Provides a single ``setup_logging`` entry point that configures the
Python logging hierarchy for the ``synth_panel`` namespace. Verbosity
is resolved from (in priority order):

1. ``debug_all=True`` (CLI ``--debug-all``): synthpanel and known
   third-party loggers at DEBUG.
2. Explicit *verbosity* argument (``"debug"`` / ``"warning"``).
3. ``SYNTHPANEL_LOG_LEVEL`` environment variable.
4. Default: ``INFO``.

Unless ``debug_all=True``, known chatty libraries (HTTP clients,
MCP/websocket stacks) stay capped at WARNING so ``--verbose`` debugs
``synth_panel`` only without drowning traces in SDK noise.
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


# Roots for ecosystems that spam at INFO/DEBUG when left unchecked.
_NOISY_LOGGERS: tuple[str, ...] = (
    "httpcore",
    "httpx",
    "hpack",
    "mcp",
    "multipart",
    "urllib3",
    "websockets",
)


def setup_logging(verbosity: str | None = None, *, debug_all: bool = False) -> None:
    """Configure logging for the ``synth_panel`` package.

    Args:
        verbosity: One of ``"debug"``, ``"info"``, ``"warning"``.
            When *None*, falls back to ``SYNTHPANEL_LOG_LEVEL`` (case-insensitive),
            then ``INFO``. Ignored for level picking when ``debug_all`` is True
            (synthpanel always DEBUG in that mode).
        debug_all: When True, elevate ``_NOISY_LOGGERS`` to DEBUG as well.
    """
    if debug_all:
        synth_level = logging.DEBUG
        noisy_level = logging.DEBUG
    elif verbosity is not None:
        synth_level = _VERBOSITY_MAP.get(verbosity.lower(), logging.INFO)
        noisy_level = logging.WARNING
    else:
        env = os.environ.get("SYNTHPANEL_LOG_LEVEL", "").strip().lower()
        synth_level = _VERBOSITY_MAP.get(env, logging.INFO)
        noisy_level = logging.WARNING

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    logger = logging.getLogger("synth_panel")
    logger.setLevel(synth_level)
    # Avoid duplicate handlers if called more than once.
    if not logger.handlers:
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.setLevel(synth_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)
