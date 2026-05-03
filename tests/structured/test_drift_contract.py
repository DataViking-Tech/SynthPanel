"""Drift contract tests (AC-8).

Post-3-strike retry exhaustion contract per SPEC.md §12.5:

- ``SYNTHPANEL_DRIFT_DEGRADE`` off (v1.0.0 default): typed
  :class:`~synth_panel.structured.validate.ErrorEnvelope` with
  ``error_code="SCHEMA_DRIFT"`` and ``retry_safe=True``.
- ``SYNTHPANEL_DRIFT_DEGRADE`` on (v1.1.0 default): degraded
  ``panel_verdict`` carrying
  ``flags=[{"code": "schema_drift", "severity": "warn"}]`` —
  **not** a typed error.
- Typed ``SCHEMA_DRIFT`` is reserved at the contract level for
  catastrophic re-validation failure, never as the carrier for
  ordinary 3-strike exhaustion when degrade is enabled.
"""

from __future__ import annotations

import pytest

from synth_panel.structured.retry import (
    DEFAULT_DEGRADE_V1_0_0,
    degrade_enabled,
    exhausted_retry_outcome,
)
from synth_panel.structured.validate import ErrorEnvelope, validate_response

_DECISION = "Should we ship the new pricing tier next quarter?"
_URI = "panel-result://result-test-drift"


def _partial_artifact(**overrides):
    base = {
        "headline": "Cohort splits on $79; price is the dominant objection.",
        "convergence": 0.42,
        "dissent_count": 2,
        "top_3_verbatims": [
            {"persona_id": "p1", "quote": "Too expensive for solo use."},
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Canonical AC-8 test (named in the bead acceptance criteria)
# ---------------------------------------------------------------------------


def test_exhausted_retry_returns_flagged_artifact_not_error() -> None:
    """When degrade is on, exhaustion returns a flagged artifact, not an error.

    This is the contract pivot: same engine event (3-strike exhaustion),
    two outcomes selected by ``SYNTHPANEL_DRIFT_DEGRADE``. With degrade
    on we deliver the partial signal under a ``schema_drift`` warn flag;
    only a catastrophic re-validation failure of the degraded artifact
    itself promotes to a typed ``SCHEMA_DRIFT`` error envelope.
    """
    out = exhausted_retry_outcome(
        partial_artifact=_partial_artifact(),
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    assert not isinstance(out, ErrorEnvelope), (
        "degrade=on must NOT promote ordinary exhaustion to a typed error; "
        "SCHEMA_DRIFT is reserved for catastrophic re-validation failure."
    )
    assert isinstance(out, dict)
    assert any(
        isinstance(f, dict) and f.get("code") == "schema_drift" and f.get("severity") == "warn" for f in out["flags"]
    ), f"degraded artifact must carry schema_drift/warn flag; got {out['flags']!r}"
    # Degraded artifact must still pass the v1.0.0 response contract.
    assert validate_response(out) is None


# ---------------------------------------------------------------------------
# Default behavior in v1.0.0: typed error
# ---------------------------------------------------------------------------


def test_v1_0_0_default_unset_returns_typed_error() -> None:
    """``SYNTHPANEL_DRIFT_DEGRADE`` unset → v1.0.0 default (off) → typed error."""
    assert DEFAULT_DEGRADE_V1_0_0 is False, "v1.0.0 ships with degrade off; flip to True only at v1.1.0."
    out = exhausted_retry_outcome(
        partial_artifact=_partial_artifact(),
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={},
    )
    assert isinstance(out, ErrorEnvelope)
    assert out.error_code == "SCHEMA_DRIFT"
    assert out.retry_safe is True
    assert out.schema_version == "1.0.0"


def test_drift_degrade_off_returns_typed_error() -> None:
    out = exhausted_retry_outcome(
        partial_artifact=_partial_artifact(),
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "off"},
    )
    assert isinstance(out, ErrorEnvelope)
    assert out.error_code == "SCHEMA_DRIFT"
    assert out.retry_safe is True


# ---------------------------------------------------------------------------
# degrade_enabled()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "on", "true", "yes", "ON", "True", "Yes"])
def test_degrade_enabled_truthy(val: str) -> None:
    assert degrade_enabled({"SYNTHPANEL_DRIFT_DEGRADE": val}) is True


@pytest.mark.parametrize("val", ["0", "off", "false", "no", "", "OFF", "False"])
def test_degrade_enabled_falsy(val: str) -> None:
    assert degrade_enabled({"SYNTHPANEL_DRIFT_DEGRADE": val}) is False


def test_degrade_enabled_unset_uses_v1_0_0_default() -> None:
    # v1.0.0 default is off (typed error). Will flip on for v1.1.0.
    assert degrade_enabled({}) is DEFAULT_DEGRADE_V1_0_0


def test_degrade_enabled_unrecognized_value_falls_back_to_default() -> None:
    """Garbage in the env var must not silently flip behavior either way."""
    assert degrade_enabled({"SYNTHPANEL_DRIFT_DEGRADE": "maybe"}) is DEFAULT_DEGRADE_V1_0_0


def test_degrade_enabled_reads_os_environ_when_env_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYNTHPANEL_DRIFT_DEGRADE", "1")
    assert degrade_enabled() is True
    monkeypatch.setenv("SYNTHPANEL_DRIFT_DEGRADE", "0")
    assert degrade_enabled() is False


# ---------------------------------------------------------------------------
# Idempotence + contract conformance of the degraded artifact
# ---------------------------------------------------------------------------


def test_degraded_artifact_idempotent_when_flag_already_present() -> None:
    """Don't double-stamp schema_drift if the upstream partial already added it."""
    partial = _partial_artifact(
        flags=[{"code": "schema_drift", "severity": "warn"}],
    )
    out = exhausted_retry_outcome(
        partial_artifact=partial,
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    assert isinstance(out, dict)
    drift_count = sum(1 for f in out["flags"] if isinstance(f, dict) and f.get("code") == "schema_drift")
    assert drift_count == 1


def test_degraded_artifact_preserves_existing_flags() -> None:
    partial = _partial_artifact(
        flags=[{"code": "low_convergence", "severity": "warn"}],
    )
    out = exhausted_retry_outcome(
        partial_artifact=partial,
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    assert isinstance(out, dict)
    codes = [f.get("code") for f in out["flags"] if isinstance(f, dict)]
    assert "low_convergence" in codes
    assert "schema_drift" in codes


def test_degraded_artifact_echoes_decision_verbatim() -> None:
    out = exhausted_retry_outcome(
        partial_artifact=_partial_artifact(),
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    assert isinstance(out, dict)
    assert out["meta"]["decision_being_informed"] == _DECISION


def test_degraded_artifact_stamps_schema_version() -> None:
    out = exhausted_retry_outcome(
        partial_artifact=_partial_artifact(),
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    assert isinstance(out, dict)
    assert out["schema_version"] == "1.0.0"


def test_degraded_artifact_does_not_mutate_input() -> None:
    """The input partial_artifact must not be mutated in place."""
    partial = _partial_artifact()
    snapshot = {k: (list(v) if isinstance(v, list) else v) for k, v in partial.items()}
    exhausted_retry_outcome(
        partial_artifact=partial,
        decision_being_informed=_DECISION,
        full_transcript_uri=_URI,
        env={"SYNTHPANEL_DRIFT_DEGRADE": "1"},
    )
    for k, v in snapshot.items():
        assert partial[k] == v, f"input partial_artifact mutated at key {k!r}"
    assert "flags" not in partial or partial.get("flags") == snapshot.get("flags")
