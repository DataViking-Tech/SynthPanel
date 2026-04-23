"""Loader for post-hoc panel reporting (sp-viz-layer T2).

Resolves a RESULT_ID-or-path to a parsed panel-result ``dict`` and raises a
single exception type, :class:`ReportLoadError`, with a discriminator
``.code`` attribute for the three failure modes used by the CLI.

The resolution policy mirrors :func:`handle_panel_inspect` and
:func:`handle_analyze` for behavioural parity: a ``result_ref`` that ends
in ``.json`` *and* refers to an existing path is read from disk; anything
else is delegated to :func:`synth_panel.mcp.data.get_panel_result`. This
means an ID that happens to end in ``.json`` or a path that looks like an
ID but does not exist on disk both fall through to the MCP-data store.
"""

from __future__ import annotations

import json
from pathlib import Path
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

    If ``result_ref`` ends in ``.json`` *and* the corresponding path exists,
    it is read from disk and ``data["id"]`` is defaulted to the path stem.
    Otherwise the call is delegated to
    :func:`synth_panel.mcp.data.get_panel_result`.

    Raises:
        ReportLoadError: with ``.code`` set to ``"not_found"`` for missing
            paths/IDs, ``"invalid_json"`` for malformed JSON, or
            ``"invalid_shape"`` if the parsed root is not a ``dict``.
    """

    path = Path(result_ref)
    if result_ref.endswith(".json") and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReportLoadError("invalid_json", f"not valid JSON: {result_ref}: {exc}") from exc
        if isinstance(data, dict):
            data.setdefault("id", path.stem)
    else:
        from synth_panel.mcp.data import get_panel_result

        try:
            data = get_panel_result(result_ref)
        except FileNotFoundError as exc:
            raise ReportLoadError("not_found", f"panel result not found: {result_ref}") from exc

    if not isinstance(data, dict):
        raise ReportLoadError("invalid_shape", "panel result is not a JSON object")

    return data
