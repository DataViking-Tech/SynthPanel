"""AC-4: v1.0.x → v1.1.0 grace-period shim for ``decision_being_informed``.

Sits in front of :func:`synth_panel.structured.validate.validate_request` on
the MCP request path. Under the v1.0.x grace window, a panel-running call that
omits ``decision_being_informed`` is allowed through with a synthesized
placeholder and a loud :data:`W_DECISION_MISSING` warning so the operator can
see legacy traffic. Setting ``SYNTHPANEL_SCHEMA_MIN="1.1.0"`` (or any later
version) flips the shim into hard-reject mode: no synthesis, no warning, and
the underlying validator is left to return ``MISSING_DECISION``.

The placeholder ``"unspecified-legacy-call"`` is itself a legal
``decision_being_informed`` value under the v1.0.0 contract, so the
synthesized payload survives the validator that sits behind this shim. The
literal string is visible in the verdict's ``meta.decision_being_informed``
and on every transcript row, exactly as the migration doc promises.
"""

from __future__ import annotations

import logging
import os
from typing import Any

_logger = logging.getLogger("synth_panel.mcp.compat")

LEGACY_DECISION_PLACEHOLDER = "unspecified-legacy-call"
"""Synthesized value injected when a legacy caller omits the field.

Length 24, single line - passes the v1.0.0 validator's 12-280 / no-newline
rules. Echoed verbatim into the verdict so audits can identify legacy traffic.
"""

W_DECISION_MISSING = "W_DECISION_MISSING"
"""Warning code logged when the shim synthesizes a placeholder."""

_DECISION_FIELD = "decision_being_informed"
_DECISION_TOOLS = frozenset({"run_panel", "run_quick_poll", "extend_panel"})
_SCHEMA_MIN_ENV = "SYNTHPANEL_SCHEMA_MIN"
_GRACE_CUTOVER = (1, 1, 0)


def apply_legacy_grace(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Return a payload with the v1.0.x grace-period shim applied.

    For panel-running tools (``run_panel``, ``run_quick_poll``,
    ``extend_panel``), if ``decision_being_informed`` is missing, empty, or
    not a non-empty string after trimming, inject the placeholder
    :data:`LEGACY_DECISION_PLACEHOLDER` and emit a :data:`W_DECISION_MISSING`
    warning. ``run_prompt`` and unknown tools are passed through unchanged.

    The hard-reject path activates when ``SYNTHPANEL_SCHEMA_MIN`` is set to
    a version ``>= 1.1.0``: the shim becomes a no-op and the downstream
    validator is left to reject the call with ``MISSING_DECISION``.

    The input payload is never mutated; a shallow copy is returned.
    """
    out = dict(payload)
    if tool not in _DECISION_TOOLS:
        return out
    if _hard_reject_enabled():
        return out
    if _has_usable_decision(out):
        return out

    _logger.warning(
        "%s: %r missing on %s under v1.0.x grace; synthesized %r. v1.1.0 will hard-reject — migrate before then.",
        W_DECISION_MISSING,
        _DECISION_FIELD,
        tool,
        LEGACY_DECISION_PLACEHOLDER,
    )
    out[_DECISION_FIELD] = LEGACY_DECISION_PLACEHOLDER
    return out


def _has_usable_decision(payload: dict[str, Any]) -> bool:
    raw = payload.get(_DECISION_FIELD)
    if not isinstance(raw, str):
        return False
    return bool(raw.strip())


def _hard_reject_enabled() -> bool:
    raw = os.environ.get(_SCHEMA_MIN_ENV, "").strip()
    if not raw:
        return False
    parsed = _parse_version(raw)
    if parsed is None:
        return False
    return parsed >= _GRACE_CUTOVER


def _parse_version(s: str) -> tuple[int, int, int] | None:
    parts = s.split(".")
    if len(parts) < 2:
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1])
        patch = int(parts[2]) if len(parts) >= 3 else 0
    except ValueError:
        return None
    return (major, minor, patch)


__all__ = [
    "LEGACY_DECISION_PLACEHOLDER",
    "W_DECISION_MISSING",
    "apply_legacy_grace",
]
