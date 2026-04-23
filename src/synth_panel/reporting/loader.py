"""Loader stubs for post-hoc panel reporting.

Scaffold for sp-viz-layer T1 — bodies raise :class:`NotImplementedError`.
T2 fills in the path-or-ID resolution and the failure taxonomy.
"""

from __future__ import annotations

from typing import Any


class ReportLoadError(Exception):
    """Raised when a panel result cannot be loaded for reporting.

    Attributes:
        code: One of ``"not_found"``, ``"invalid_json"``, ``"invalid_shape"``.
    """

    code: str

    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(message or code)
        self.code = code


def load_panel_json(result_ref: str) -> dict[str, Any]:
    """Resolve a panel result ID or file path to a parsed JSON dict.

    Mirrors the path-or-ID resolution used by ``handle_panel_inspect`` and
    ``handle_analyze``: paths ending in ``.json`` that exist on disk are read
    directly; anything else is delegated to
    :func:`synth_panel.mcp.data.get_panel_result`.

    Raises:
        ReportLoadError: with ``.code`` set to ``"not_found"``,
            ``"invalid_json"``, or ``"invalid_shape"`` per the failure mode.
    """

    raise NotImplementedError("sp-viz-layer T2: implement load_panel_json")
