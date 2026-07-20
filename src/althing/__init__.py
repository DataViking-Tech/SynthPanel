"""althing — synthetic focus groups for AI personas.

The package root re-exports the public Python SDK so callers can write::

    from althing import quick_poll, run_panel, run_prompt

See :mod:`althing.sdk` for full API documentation. The CLI lives in
:mod:`althing.main` (entry point ``althing``); the MCP server lives
in :mod:`althing.mcp.server` and requires the optional ``[mcp]``
extra — the SDK itself works on a plain ``pip install althing``.
"""

from __future__ import annotations

# Legacy SYNTHPANEL_* env-var bridge — must run before any submodule reads
# os.environ, hence first import.
from althing import _compat_env as _compat_env
from althing.__version__ import __version__
from althing.sdk import (
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
