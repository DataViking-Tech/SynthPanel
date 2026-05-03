"""Drift contract — post-3-strike exhaustion behavior (AC-8).

When the structured-output engine's 3-strike retry exhausts on haiku
malformed output, the server's contract-level response depends on
``SYNTHPANEL_DRIFT_DEGRADE``:

* **Off** (v1.0.0 default) — return a typed
  :class:`~synth_panel.structured.validate.ErrorEnvelope` with
  ``error_code="SCHEMA_DRIFT"`` and ``retry_safe=True``. The caller is
  expected to retry the request with cleaner stimulus or escalate.
* **On** (v1.1.0 default) — return the **degraded artifact** — a
  ``panel_verdict`` carrying
  ``flags=[{"code": "schema_drift", "severity": "warn"}]``. The panel
  ran, the user gets partial signal, the flag is the contract for
  "trust this less."

The typed ``SCHEMA_DRIFT`` error code is reserved at the contract level
for the catastrophic case where the degraded artifact itself fails
re-validation; ordinary 3-strike exhaustion under degrade-on must NOT
promote to it. See SPEC.md §12.5 and docs/response-contract.md.

This module is the contract pivot, not the strike loop. The per-attempt
retry policy lives in :mod:`synth_panel.structured.output`; this module
decides what shape leaves the server once that loop has given up.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

from synth_panel.structured.validate import ErrorEnvelope

logger = logging.getLogger(__name__)

DRIFT_DEGRADE_ENV: str = "SYNTHPANEL_DRIFT_DEGRADE"

# Flip to True for the v1.1.0 schema cut. Kept as a module constant so
# tests can pin v1.0.0 behavior without duplicating the string literal.
DEFAULT_DEGRADE_V1_0_0: bool = False

_SCHEMA_VERSION = "1.0.0"
_DRIFT_FLAG_CODE = "schema_drift"
_DRIFT_FLAG_SEVERITY = "warn"

# Conventional posix-shell idioms. Anything outside both sets falls
# back to the v1.0.0 default rather than guessing intent.
_TRUTHY = frozenset({"1", "on", "true", "yes"})
_FALSY = frozenset({"0", "off", "false", "no", ""})

_DEGRADED_HEADLINE_FALLBACK = "Degraded panel — schema drift after 3-strike retry exhaustion."


def degrade_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return True when ``SYNTHPANEL_DRIFT_DEGRADE`` selects degraded-artifact mode.

    Reads :data:`os.environ` by default; pass *env* (any mapping) to
    override — the test seam. Unset or unrecognized values fall back to
    :data:`DEFAULT_DEGRADE_V1_0_0` rather than silently flipping
    behavior on a typo.
    """
    src: Mapping[str, str] = env if env is not None else os.environ
    raw = src.get(DRIFT_DEGRADE_ENV)
    if raw is None:
        return DEFAULT_DEGRADE_V1_0_0
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    logger.debug(
        "SYNTHPANEL_DRIFT_DEGRADE=%r is not a recognized truthy/falsy value; falling back to v1.0.0 default (%s).",
        raw,
        DEFAULT_DEGRADE_V1_0_0,
    )
    return DEFAULT_DEGRADE_V1_0_0


def exhausted_retry_outcome(
    *,
    partial_artifact: Mapping[str, Any],
    decision_being_informed: str,
    full_transcript_uri: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any] | ErrorEnvelope:
    """Return the contract-correct outcome after 3-strike retry exhaustion.

    With degrade enabled, returns a degraded ``panel_verdict`` dict
    carrying a ``schema_drift`` warn flag (alongside any flags the
    partial already raised). With degrade disabled (v1.0.0 default),
    returns a typed :class:`ErrorEnvelope` with ``error_code="SCHEMA_DRIFT"``
    and ``retry_safe=True`` — the caller can safely retry with
    different stimulus.

    The input *partial_artifact* is never mutated; the returned dict
    is always a fresh shallow-then-flag-list copy.
    """
    if degrade_enabled(env):
        return _degraded_artifact(
            partial_artifact=partial_artifact,
            decision_being_informed=decision_being_informed,
            full_transcript_uri=full_transcript_uri,
        )

    return ErrorEnvelope(
        error_code="SCHEMA_DRIFT",
        message=(
            "structured output exhausted 3-strike retry; the degraded "
            "artifact is not delivered because SYNTHPANEL_DRIFT_DEGRADE "
            "is off (v1.0.0 default). Set SYNTHPANEL_DRIFT_DEGRADE=1 to "
            "receive the degraded panel_verdict with a schema_drift flag "
            "instead. This will become the default at v1.1.0."
        ),
        field_path=None,
        schema_version=_SCHEMA_VERSION,
        retry_safe=True,
    )


def _degraded_artifact(
    *,
    partial_artifact: Mapping[str, Any],
    decision_being_informed: str,
    full_transcript_uri: str,
) -> dict[str, Any]:
    """Stamp the degraded-artifact contract onto *partial_artifact*.

    Idempotent on the ``schema_drift`` flag (set semantics on
    ``flags[]``) and overwrites the contract-required fields the caller
    is responsible for (transcript URI, decision echo, schema_version)
    so the artifact passes :func:`validate_response` even if the
    upstream partial omitted them.
    """
    artifact: dict[str, Any] = dict(partial_artifact)

    raw_flags = artifact.get("flags") or []
    flags: list[dict[str, Any]] = [dict(f) for f in raw_flags if isinstance(f, dict)]
    if not any(f.get("code") == _DRIFT_FLAG_CODE for f in flags):
        flags.append({"code": _DRIFT_FLAG_CODE, "severity": _DRIFT_FLAG_SEVERITY})
    artifact["flags"] = flags

    raw_ext = artifact.get("extension") or []
    artifact["extension"] = [dict(e) for e in raw_ext if isinstance(e, dict)]

    artifact.setdefault("headline", _DEGRADED_HEADLINE_FALLBACK)
    artifact.setdefault("convergence", 0.0)
    artifact.setdefault("dissent_count", 0)

    raw_verbatims = artifact.get("top_3_verbatims") or []
    artifact["top_3_verbatims"] = [dict(v) for v in raw_verbatims if isinstance(v, dict)]

    artifact["full_transcript_uri"] = full_transcript_uri

    meta = dict(artifact.get("meta") or {})
    meta["decision_being_informed"] = decision_being_informed
    artifact["meta"] = meta

    artifact["schema_version"] = _SCHEMA_VERSION

    return artifact


__all__ = [
    "DEFAULT_DEGRADE_V1_0_0",
    "DRIFT_DEGRADE_ENV",
    "degrade_enabled",
    "exhausted_retry_outcome",
]
