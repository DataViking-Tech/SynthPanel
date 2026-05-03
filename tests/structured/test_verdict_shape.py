"""Tests for the v1.0.0 panel_verdict assembler (AC-6).

The assembler builds a ``panel_verdict`` dict that conforms to
``schemas/v1.0.0.json#/panel_verdict``. It is the only sanctioned producer
of the artifact — the AC-9 response gate validates whatever this module
emits, so a violation here is a contract break, not a soft warning.
"""

from __future__ import annotations

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.orchestrator import (
    Flag,
    FlagExtension,
    PanelistResult,
    PanelState,
)
from synth_panel.structured.validate import validate_response
from synth_panel.structured.verdict import build_panel_verdict

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"


def _panelists(n: int) -> list[PanelistResult]:
    return [
        PanelistResult(
            persona_name=f"p{i}",
            responses=[{"response": "fine"}],
            usage=ZERO_USAGE,
        )
        for i in range(n)
    ]


def _state(**overrides) -> PanelState:
    """Healthy panel state — 8 panelists, no skew, no drift."""
    base = {
        "panelist_results": _panelists(8),
        "personas": [{"name": f"p{i}", "occupation": f"job{i}"} for i in range(8)],
        "convergence": 0.72,
        "schema_drift": False,
        "expected_categories": None,
        "observed_categories": None,
        "extensions": [],
    }
    base.update(overrides)
    return PanelState(**base)


# ---------------------------------------------------------------------------
# Canonical bead test (sy-b3n)
# ---------------------------------------------------------------------------


def test_verdict_echoes_decision_and_version() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="Strong support with pricing caveats.",
        full_transcript_uri="file:///tmp/run.jsonl",
    )

    assert artifact["meta"]["decision_being_informed"] == _VALID_DECISION
    assert artifact["schema_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_default_artifact_passes_response_validator() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    assert validate_response(artifact) is None


def test_artifact_contains_all_required_top_level_fields() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    required = {
        "headline",
        "convergence",
        "dissent_count",
        "top_3_verbatims",
        "flags",
        "extension",
        "full_transcript_uri",
        "meta",
        "schema_version",
    }
    assert set(artifact.keys()) >= required


# ---------------------------------------------------------------------------
# Bound enforcement
# ---------------------------------------------------------------------------


def test_headline_is_truncated_to_140_chars() -> None:
    long = "x" * 500
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline=long,
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    assert len(artifact["headline"]) <= 140


def test_top_3_verbatims_clamps_to_three_items() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
        top_3_verbatims=[{"persona_id": f"p{i}", "quote": f"q{i}"} for i in range(5)],
    )
    assert len(artifact["top_3_verbatims"]) == 3
    assert artifact["top_3_verbatims"][0] == {"persona_id": "p0", "quote": "q0"}


def test_top_3_verbatims_strips_unknown_keys() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
        top_3_verbatims=[
            {"persona_id": "p1", "quote": "great", "extra": "ignored"},
        ],
    )
    assert artifact["top_3_verbatims"] == [{"persona_id": "p1", "quote": "great"}]


def test_convergence_clamped_to_unit_interval() -> None:
    over = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(convergence=1.7),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    under = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(convergence=-0.2),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    assert over["convergence"] == 1.0
    assert under["convergence"] == 0.0


def test_convergence_defaults_to_zero_when_unknown() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(convergence=None),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    assert artifact["convergence"] == 0.0


def test_dissent_count_is_non_negative_integer() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
        dissent_count=-3,
    )
    assert artifact["dissent_count"] == 0
    assert isinstance(artifact["dissent_count"], int)


# ---------------------------------------------------------------------------
# Flags + extension routing
# ---------------------------------------------------------------------------


def test_small_n_state_emits_small_n_flag() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(panelist_results=_panelists(2), personas=[]),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    flag_codes = {f["code"] for f in artifact["flags"]}
    assert "small_n" in flag_codes
    # Conformance: every emitted flag uses an enum-valid (code, severity).
    assert validate_response(artifact) is None


def test_extensions_pass_through_to_extension_array() -> None:
    ext = FlagExtension(
        code="custom.cohort_imbalance",
        message="Two regions account for 90% of panel.",
        severity="warn",
    )
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(extensions=[ext]),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
    )
    assert artifact["extension"] == [
        {
            "code": "custom.cohort_imbalance",
            "message": "Two regions account for 90% of panel.",
            "severity": "warn",
        }
    ]


def test_extra_flags_appended_to_raised_set() -> None:
    artifact = build_panel_verdict(
        decision_being_informed=_VALID_DECISION,
        panel_state=_state(),
        headline="OK",
        full_transcript_uri="file:///tmp/run.jsonl",
        extra_flags=[Flag(code="schema_drift", severity="warn")],
    )
    flag_codes = [f["code"] for f in artifact["flags"]]
    assert "schema_drift" in flag_codes


# ---------------------------------------------------------------------------
# Decision-field defenses
# ---------------------------------------------------------------------------


def test_decision_must_be_non_empty_string() -> None:
    with pytest.raises(ValueError):
        build_panel_verdict(
            decision_being_informed="",
            panel_state=_state(),
            headline="OK",
            full_transcript_uri="file:///tmp/run.jsonl",
        )


def test_decision_with_newline_rejected() -> None:
    with pytest.raises(ValueError):
        build_panel_verdict(
            decision_being_informed="line one\nline two",
            panel_state=_state(),
            headline="OK",
            full_transcript_uri="file:///tmp/run.jsonl",
        )
