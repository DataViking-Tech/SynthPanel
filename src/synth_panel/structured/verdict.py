"""panel_verdict assembler for the v1.0.0 frozen contract (AC-6).

Builds the response artifact described by ``schemas/v1.0.0.json#/panel_verdict``
from a post-synthesis :class:`~synth_panel.orchestrator.PanelState` snapshot
plus the caller-supplied summary fields (headline, top verbatims, transcript
URI). Flag raising is delegated to AC-5
(:func:`~synth_panel.orchestrator._raise_flags`) so the contract is enforced
in exactly one place; non-enum signals ride on
``panel_state.extensions`` and surface under ``panel_verdict.extension[]``.

The assembler defends the bound rules the JSON Schema declares
(``headline`` ≤ 140 chars, ``top_3_verbatims`` ≤ 3 items, ``convergence``
in [0, 1], ``dissent_count`` ≥ 0) so it physically cannot ship a
non-conformant artifact, even if upstream code passes loose values.
The AC-9 response gate (:func:`validate_response`) re-checks egress;
this module is the producer-side counterpart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from synth_panel.orchestrator import Flag, PanelState, _raise_flags

_SCHEMA_VERSION = "1.0.0"
_HEADLINE_MAX_CHARS = 140
_VERBATIMS_MAX_ITEMS = 3
_DECISION_FIELD = "decision_being_informed"


def _coerce_decision(value: Any) -> str:
    """Reject the obvious decision-field violations at the producer boundary.

    The validator-core (AC-2/AC-3) rejects these on ingress, but we re-check
    here because nothing forces the caller to round-trip the request through
    the validator before assembling — the assembler is the single chokepoint
    for what enters ``meta.decision_being_informed``.
    """
    if not isinstance(value, str):
        raise ValueError(f"{_DECISION_FIELD!r} must be a string, got {type(value).__name__}")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{_DECISION_FIELD!r} must not contain newline characters")
    if not value.strip():
        raise ValueError(f"{_DECISION_FIELD!r} must be non-empty after trimming")
    return value


def _coerce_verbatims(items: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Clamp to ≤ 3 items and project to the schema's ``{persona_id, quote}`` shape.

    Drops unknown keys silently — the v1.0.0 schema has
    ``additionalProperties: false`` on each verbatim object, so passing
    them through would fail :func:`validate_response`.
    """
    out: list[dict[str, str]] = []
    for v in items:
        if len(out) >= _VERBATIMS_MAX_ITEMS:
            break
        try:
            out.append({"persona_id": str(v["persona_id"]), "quote": str(v["quote"])})
        except KeyError as exc:
            raise ValueError(f"verbatim missing required key: {exc.args[0]!r}") from None
    return out


def _coerce_convergence(state_value: float | None, override: float | None) -> float:
    """Pick the override if given, else state, else 0.0; clamp to [0, 1]."""
    raw = override if override is not None else state_value
    if raw is None:
        return 0.0
    return max(0.0, min(1.0, float(raw)))


def build_panel_verdict(
    *,
    decision_being_informed: str,
    panel_state: PanelState,
    headline: str,
    full_transcript_uri: str,
    top_3_verbatims: Iterable[Mapping[str, Any]] = (),
    dissent_count: int = 0,
    convergence: float | None = None,
    extra_flags: Iterable[Flag] = (),
) -> dict[str, Any]:
    """Assemble a v1.0.0-conformant ``panel_verdict`` dict.

    Parameters
    ----------
    decision_being_informed:
        The request-side decision string. Must be non-empty after trimming
        and contain no newline characters (mirrors the validator-core rule).
    panel_state:
        Post-synthesis snapshot. Drives flag raising (via AC-5) and supplies
        the default ``convergence`` value when the override is omitted.
    headline:
        Short summary line. Truncated to 140 characters per
        ``panel_verdict.headline.maxLength``.
    full_transcript_uri:
        Pointer to the persisted transcript (file://, s3://, …). Coerced to
        ``str`` — the schema only requires a string, not a parseable URI.
    top_3_verbatims:
        Up to three ``{persona_id, quote}`` mappings. Extra entries are
        dropped from the tail; unknown keys per entry are stripped.
    dissent_count:
        Number of panelists who dissented from the modal stance. Negative
        values are clamped to 0 to satisfy the schema's
        ``minimum: 0`` constraint.
    convergence:
        Optional override for ``panel_state.convergence``. ``None`` falls
        through to state; any value is clamped to ``[0.0, 1.0]``.
    extra_flags:
        Additional :class:`Flag` instances to append after the raised set
        (e.g. a schema-drift flag set by AC-9 just before egress). Each
        must already be enum-valid — :class:`Flag` rejects bad codes at
        construction.

    Returns
    -------
    dict
        A plain dict. Mutation is the caller's responsibility; the
        assembler holds no reference after return.
    """
    decision = _coerce_decision(decision_being_informed)

    raised: list[Flag] = list(_raise_flags(panel_state))
    raised.extend(extra_flags or ())
    flag_dicts = [{"code": f.code, "severity": f.severity} for f in raised]

    extension_dicts = [{"code": e.code, "message": e.message, "severity": e.severity} for e in panel_state.extensions]

    artifact: dict[str, Any] = {
        "headline": str(headline)[:_HEADLINE_MAX_CHARS],
        "convergence": _coerce_convergence(panel_state.convergence, convergence),
        "dissent_count": max(0, int(dissent_count)),
        "top_3_verbatims": _coerce_verbatims(top_3_verbatims),
        "flags": flag_dicts,
        "extension": extension_dicts,
        "full_transcript_uri": str(full_transcript_uri),
        "meta": {_DECISION_FIELD: decision},
        "schema_version": _SCHEMA_VERSION,
    }
    return artifact


__all__ = ["build_panel_verdict"]
