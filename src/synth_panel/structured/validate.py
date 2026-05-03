"""Validator-core for the v1.0.0 frozen MCP contract (AC-2).

Provides :func:`validate_request` and :func:`validate_response`, which return
``None`` when input conforms to the contract and a typed
:class:`ErrorEnvelope` (mirroring ``error_envelope`` in
``schemas/v1.0.0.json``) on failure.

Scope is intentionally narrow: this module enforces only the structural
fragment-level rules expressed in the v1.0.0 schema (decision_being_informed
shape, panel_verdict required fields, flag/severity enum membership). Tool
wiring lives elsewhere — AC-3 plugs ``validate_request`` into the MCP
``run_panel`` / ``run_quick_poll`` / ``extend_panel`` handlers, and AC-9
plugs :func:`apply_response_gate` (which calls :func:`validate_response`)
into the MCP response path so no v1.0.0 ``panel_verdict`` artifact can
leave the server unvalidated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from synth_panel.schemas import load

_SCHEMA_VERSION = "1.0.0"

_DECISION_TOOLS = frozenset({"run_panel", "run_quick_poll", "extend_panel"})
_NO_DECISION_TOOLS = frozenset({"run_prompt"})
_KNOWN_TOOLS = _DECISION_TOOLS | _NO_DECISION_TOOLS

_DECISION_FIELD = "decision_being_informed"

_schema_cache: Any = None


def _schema() -> Any:
    global _schema_cache
    if _schema_cache is None:
        _schema_cache = load(_SCHEMA_VERSION)
    return _schema_cache


@dataclass(frozen=True)
class ErrorEnvelope:
    """Typed error envelope mirroring ``error_envelope`` in v1.0.0.json."""

    error_code: str
    message: str
    field_path: str | None
    schema_version: str
    retry_safe: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["field_path"] is None:
            del d["field_path"]
        return d


def _err(
    code: str,
    message: str,
    *,
    field_path: str | None = None,
    retry_safe: bool = True,
) -> ErrorEnvelope:
    return ErrorEnvelope(
        error_code=code,
        message=message,
        field_path=field_path,
        schema_version=_SCHEMA_VERSION,
        retry_safe=retry_safe,
    )


def validate_request(tool: str, payload: dict[str, Any]) -> ErrorEnvelope | None:
    """Validate a request payload against the v1.0.0 contract for *tool*.

    Returns ``None`` when the payload conforms, otherwise an
    :class:`ErrorEnvelope` describing the first failure.
    """
    if tool not in _KNOWN_TOOLS:
        return _err(
            "INVALID_TOOL_ARG",
            f"Unknown tool {tool!r}; expected one of {sorted(_KNOWN_TOOLS)}.",
        )

    if not isinstance(payload, dict):
        return _err("INVALID_TOOL_ARG", "Request payload must be an object.")

    if tool in _NO_DECISION_TOOLS:
        if _DECISION_FIELD in payload:
            return _err(
                "INVALID_TOOL_ARG",
                f"Tool {tool!r} does not accept {_DECISION_FIELD!r}.",
                field_path=_DECISION_FIELD,
            )
        return None

    return _validate_decision_field(payload)


def _validate_decision_field(payload: dict[str, Any]) -> ErrorEnvelope | None:
    if _DECISION_FIELD not in payload or payload[_DECISION_FIELD] is None:
        return _err(
            "MISSING_DECISION",
            f"Required field {_DECISION_FIELD!r} is missing.",
            field_path=_DECISION_FIELD,
        )

    raw = payload[_DECISION_FIELD]
    if not isinstance(raw, str):
        return _err(
            "INVALID_TOOL_ARG",
            f"{_DECISION_FIELD!r} must be a string.",
            field_path=_DECISION_FIELD,
        )

    if "\n" in raw or "\r" in raw:
        return _err(
            "INVALID_TOOL_ARG",
            f"{_DECISION_FIELD!r} must not contain newline characters.",
            field_path=_DECISION_FIELD,
        )

    trimmed = raw.strip()
    if not trimmed:
        return _err(
            "MISSING_DECISION",
            f"Required field {_DECISION_FIELD!r} is empty after trimming.",
            field_path=_DECISION_FIELD,
        )

    fragment = _schema()["request_fragments"][_DECISION_FIELD]
    min_len = fragment["minLength"]
    max_len = fragment["maxLength"]
    n = len(trimmed)
    if n > max_len:
        return _err(
            "DECISION_TOO_LONG",
            f"{_DECISION_FIELD!r} is {n} chars (trimmed); max is {max_len}.",
            field_path=_DECISION_FIELD,
        )
    if n < min_len:
        return _err(
            "INVALID_TOOL_ARG",
            f"{_DECISION_FIELD!r} is {n} chars (trimmed); min is {min_len}.",
            field_path=_DECISION_FIELD,
        )
    return None


def validate_response(artifact: dict[str, Any]) -> ErrorEnvelope | None:
    """Validate a panel_verdict artifact against the v1.0.0 contract.

    Returns ``None`` when the artifact conforms, otherwise an
    :class:`ErrorEnvelope` describing the first failure. Failures here are
    ``retry_safe=False``: the server has already produced a non-conforming
    artifact, so a same-input retry is not a valid recovery.
    """
    if not isinstance(artifact, dict):
        return _err(
            "SCHEMA_DRIFT",
            "Response artifact must be an object.",
            retry_safe=False,
        )

    schema = _schema()
    verdict_schema = schema["panel_verdict"]

    for key in verdict_schema["required"]:
        if key not in artifact:
            return _err(
                "SCHEMA_DRIFT",
                f"panel_verdict missing required field {key!r}.",
                field_path=key,
                retry_safe=False,
            )

    if artifact.get("schema_version") != _SCHEMA_VERSION:
        return _err(
            "SCHEMA_DRIFT",
            f"schema_version must be {_SCHEMA_VERSION!r}; got {artifact.get('schema_version')!r}.",
            field_path="schema_version",
            retry_safe=False,
        )

    flags_enum = set(schema["flags_enum"]["enum"])
    severity_enum = set(schema["severity_enum"]["enum"])
    flags = artifact.get("flags", ())
    if not isinstance(flags, (list, tuple)):
        return _err(
            "SCHEMA_DRIFT",
            "panel_verdict.flags must be an array.",
            field_path="flags",
            retry_safe=False,
        )
    for i, flag in enumerate(flags):
        if not isinstance(flag, dict):
            return _err(
                "INVALID_FLAG",
                f"flags[{i}] must be an object.",
                field_path=f"flags[{i}]",
                retry_safe=False,
            )
        code = flag.get("code")
        if code not in flags_enum:
            return _err(
                "INVALID_FLAG",
                f"flags[{i}].code {code!r} is not a member of the flags enum.",
                field_path=f"flags[{i}].code",
                retry_safe=False,
            )
        sev = flag.get("severity")
        if sev not in severity_enum:
            return _err(
                "INVALID_FLAG",
                f"flags[{i}].severity {sev!r} is not a member of the severity enum.",
                field_path=f"flags[{i}].severity",
                retry_safe=False,
            )

    meta = artifact.get("meta")
    if not isinstance(meta, dict) or _DECISION_FIELD not in meta:
        return _err(
            "SCHEMA_DRIFT",
            f"panel_verdict.meta.{_DECISION_FIELD} missing.",
            field_path=f"meta.{_DECISION_FIELD}",
            retry_safe=False,
        )

    return None


def apply_response_gate(artifact: Any) -> Any:
    """AC-9 response gate — final validation before egress.

    Treats *artifact* as a v1.0.0 ``panel_verdict`` candidate iff it is a
    ``dict`` carrying a ``schema_version`` key. In that case
    :func:`validate_response` runs and, on failure, the artifact is
    replaced with the typed error envelope dict. Conformant artifacts
    are returned unchanged (same identity).

    Anything else — non-dicts, dicts without ``schema_version`` (legacy
    pre-contract MCP shapes such as ``{"results": [...], ...}``) — passes
    through untouched. The contract is enforced exactly where it claims
    to apply, without regressing shipped clients that still consume the
    pre-v1.0.0 response envelope.

    The gate is the single MCP egress chokepoint for the v1.0.0
    contract; treat every ``return json.dumps(...)`` site that may carry
    a panel_verdict as a place to call this function.
    """
    if not isinstance(artifact, dict):
        return artifact
    if "schema_version" not in artifact:
        return artifact
    err = validate_response(artifact)
    if err is None:
        return artifact
    return err.to_dict()


__all__ = [
    "ErrorEnvelope",
    "apply_response_gate",
    "validate_request",
    "validate_response",
]
