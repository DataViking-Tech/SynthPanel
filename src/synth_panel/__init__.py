"""synthpanel — synthetic focus groups for AI personas.

The package root re-exports the public Python SDK so callers can write::

    from synth_panel import quick_poll, run_panel, run_prompt

See :mod:`synth_panel.sdk` for full API documentation. The CLI lives in
:mod:`synth_panel.main` (entry point ``synthpanel``); the MCP server lives
in :mod:`synth_panel.mcp.server` and requires the optional ``[mcp]``
extra — the SDK itself works on a plain ``pip install synthpanel``.
"""

from __future__ import annotations

from synth_panel.sdk import (
    PanelResult,
    PollResult,
    PromptResult,
    extend_panel,
    get_panel_result,
    list_instruments,
    list_panel_results,
    list_personas,
    quick_poll,
    run_panel,
    run_prompt,
)

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("synthpanel")
except Exception:
    __version__ = "0.0.0-dev"


__all__ = [
    "PanelResult",
    "PollResult",
    "PromptResult",
    "__version__",
    "extend_panel",
    "get_panel_result",
    "list_instruments",
    "list_panel_results",
    "list_personas",
    "quick_poll",
    "run_panel",
    "run_prompt",
]
