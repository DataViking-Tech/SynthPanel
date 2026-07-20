"""althing — synthetic focus groups for AI personas.

The package root re-exports the public Python SDK so callers can write::

    from althing import quick_poll, run_panel, run_prompt

See :mod:`althing.sdk` for full API documentation. The CLI lives in
:mod:`althing.main` (entry point ``althing``); the MCP server lives
in :mod:`althing.mcp.server` and requires the optional ``[mcp]``
extra — the SDK itself works on a plain ``pip install althing``.
"""

from __future__ import annotations

import os as _os
import warnings as _warnings

# SynthPanel → Althing rename (1.0): honor legacy SYNTHPANEL_* environment
# variables for one deprecation cycle by mirroring them into their ALTHING_*
# equivalents when the new name is unset. Central bridge here so every
# downstream os.environ reader inherits the fallback.
_legacy = [k for k in _os.environ if k.startswith("SYNTHPANEL_")]
for _k in _legacy:
    _new = "ALTHING_" + _k[len("SYNTHPANEL_") :]
    _os.environ.setdefault(_new, _os.environ[_k])
if _legacy:
    _warnings.warn(
        f"SYNTHPANEL_* environment variables are deprecated after the rename "
        f"to althing — set {', '.join(sorted(('ALTHING_' + k[len('SYNTHPANEL_'):]) for k in _legacy))} "
        f"instead. Legacy names will stop working in a future major release.",
        DeprecationWarning,
        stacklevel=2,
    )
del _legacy

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
