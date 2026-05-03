"""Tests for the post-synthesis flag raiser (sy-ac-5).

Covers all 7 ``flags_enum`` members in the v1.0.0 contract plus the
``extension[]`` sibling for non-enum signals. The canonical entry point
is ``test_small_n_raises_with_severity_warn`` (named in build-plan.md
AC-5); the remaining tests pin per-flag thresholds and the schema-bound
guarantees on :class:`Flag` / :class:`FlagExtension`.
"""

from __future__ import annotations

from typing import Any

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.orchestrator import (
    Flag,
    FlagExtension,
    PanelistResult,
    PanelState,
    _raise_flags,
)
from synth_panel.schemas import load


def _pr(name: str = "P", *, error: str | None = None, responses: list[dict[str, Any]] | None = None) -> PanelistResult:
    return PanelistResult(
        persona_name=name,
        responses=responses if responses is not None else [{"question": "q", "response": "ok"}],
        usage=ZERO_USAGE,
        error=error,
    )


def _diverse_personas(n: int) -> list[dict[str, Any]]:
    """Personas with distinct demographics so we don't accidentally trip
    the demographic_skew detector when probing other flags."""
    return [{"name": f"P{i}", "occupation": f"job-{i}", "age": 20 + i} for i in range(n)]


# ---------------------------------------------------------------------------
# Required test (build-plan.md AC-5)
# ---------------------------------------------------------------------------


def test_small_n_raises_with_severity_warn() -> None:
    """A panel with 4-7 completed panelists raises ``small_n`` at ``warn``."""
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(5)],
        personas=_diverse_personas(5),
    )

    flags = _raise_flags(state)

    small_n = [f for f in flags if f.code == "small_n"]
    assert len(small_n) == 1, f"expected exactly one small_n flag, got {flags!r}"
    assert small_n[0].severity == "warn"


# ---------------------------------------------------------------------------
# Per-flag coverage
# ---------------------------------------------------------------------------


def test_small_n_blocks_below_4() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(2)],
        personas=_diverse_personas(2),
    )
    flags = _raise_flags(state)
    assert Flag(code="small_n", severity="block") in flags


def test_no_small_n_when_n_at_threshold() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(8)],
        personas=_diverse_personas(8),
    )
    flags = _raise_flags(state)
    assert not any(f.code == "small_n" for f in flags)


def test_low_convergence_warn_band() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        convergence=0.40,
    )
    flags = _raise_flags(state)
    assert Flag(code="low_convergence", severity="warn") in flags


def test_low_convergence_block_band() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        convergence=0.10,
    )
    flags = _raise_flags(state)
    assert Flag(code="low_convergence", severity="block") in flags


def test_demographic_skew_when_all_share_a_field() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=[{"name": f"P{i}", "occupation": "data scientist", "age": 30 + i} for i in range(20)],
    )
    flags = _raise_flags(state)
    assert Flag(code="demographic_skew", severity="warn") in flags


def test_demographic_skew_silent_when_diverse() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
    )
    flags = _raise_flags(state)
    assert not any(f.code == "demographic_skew" for f in flags)


def test_persona_collision_on_duplicate_names() -> None:
    personas = [{"name": "Alice", "occupation": "a"}, {"name": "Alice", "occupation": "b"}, *_diverse_personas(8)]
    results = [_pr("Alice"), _pr("Alice"), *[_pr(f"P{i}") for i in range(8)]]
    state = PanelState(panelist_results=results, personas=personas)
    flags = _raise_flags(state)
    assert Flag(code="persona_collision", severity="warn") in flags


def test_out_of_distribution_when_unexpected_categories() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        expected_categories=["yes", "no"],
        observed_categories=["yes", "no", "maybe"],
    )
    flags = _raise_flags(state)
    assert Flag(code="out_of_distribution", severity="warn") in flags


def test_out_of_distribution_silent_when_subset() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        expected_categories=["yes", "no", "maybe"],
        observed_categories=["yes", "no"],
    )
    flags = _raise_flags(state)
    assert not any(f.code == "out_of_distribution" for f in flags)


def test_refusal_or_degenerate_warn_above_quarter() -> None:
    # 5/20 = 25% errors → warn band
    results = [_pr(f"P{i}") for i in range(15)] + [_pr(f"P{i}", error="refused") for i in range(15, 20)]
    state = PanelState(panelist_results=results, personas=_diverse_personas(20))
    flags = _raise_flags(state)
    refusal = [f for f in flags if f.code == "refusal_or_degenerate"]
    assert len(refusal) == 1
    assert refusal[0].severity == "warn"


def test_refusal_or_degenerate_block_above_half() -> None:
    results = [_pr(f"P{i}") for i in range(8)] + [_pr(f"P{i}", error="x") for i in range(8, 20)]
    state = PanelState(panelist_results=results, personas=_diverse_personas(20))
    flags = _raise_flags(state)
    refusal = [f for f in flags if f.code == "refusal_or_degenerate"]
    assert refusal and refusal[0].severity == "block"


def test_refusal_counts_per_response_failures() -> None:
    """A panelist whose every response errored is degenerate even without
    an outer ``error`` string set on the result."""
    bad = _pr(
        "P",
        responses=[
            {"question": "q1", "response": "[error: x]", "error": True},
            {"question": "q2", "response": None, "skipped_by_budget": True},
        ],
    )
    results = [bad, *[_pr(f"P{i}") for i in range(11)]]
    state = PanelState(panelist_results=results, personas=_diverse_personas(12))
    flags = _raise_flags(state)
    # 1/12 ≈ 8% → no flag
    assert not any(f.code == "refusal_or_degenerate" for f in flags)


def test_schema_drift_passthrough() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        schema_drift=True,
    )
    flags = _raise_flags(state)
    assert Flag(code="schema_drift", severity="warn") in flags


def test_clean_panel_emits_no_flags() -> None:
    state = PanelState(
        panelist_results=[_pr(f"P{i}") for i in range(20)],
        personas=_diverse_personas(20),
        convergence=0.85,
    )
    assert _raise_flags(state) == []


def test_empty_panel_emits_no_flags() -> None:
    """A zero-panelist state is degenerate but not yet classifiable —
    the verdict assembler will mark the run invalid separately. The
    raiser must not crash on it."""
    assert _raise_flags(PanelState()) == []


# ---------------------------------------------------------------------------
# Schema conformance
# ---------------------------------------------------------------------------


def test_all_seven_enum_members_are_reachable() -> None:
    """Constructing a maximally-bad panel triggers every contract enum
    member exactly once. This pins the raiser to the v1.0.0 contract:
    if the schema gains an 8th flag, this test fails until the raiser
    learns to produce it."""
    schema_codes = set(load(version="1.0.0")["flags_enum"]["enum"])

    state = PanelState(
        panelist_results=[_pr(f"P{i}", error="refused") for i in range(2)],
        personas=[{"name": "Dup", "occupation": "same"}, {"name": "Dup", "occupation": "same"}],
        convergence=0.05,
        expected_categories=["a"],
        observed_categories=["a", "z"],
        schema_drift=True,
    )
    raised = {f.code for f in _raise_flags(state)}
    assert raised == schema_codes


def test_flag_rejects_non_enum_code() -> None:
    with pytest.raises(ValueError, match="not_a_real_flag"):
        Flag(code="not_a_real_flag", severity="warn")


def test_flag_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="critical"):
        Flag(code="small_n", severity="critical")


def test_extension_rejects_enum_collision() -> None:
    """Non-enum extensions belong on ``extension[]``; reusing an enum
    member from there would silently shadow a real flag in the verdict
    assembler. Raise on construction so the mistake is loud."""
    with pytest.raises(ValueError, match="small_n"):
        FlagExtension(code="small_n", message="x", severity="warn")


def test_extension_rejects_invalid_severity() -> None:
    with pytest.raises(ValueError, match="critical"):
        FlagExtension(code="custom_thing", message="x", severity="critical")


def test_extension_rejects_empty_code() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        FlagExtension(code="", message="x", severity="info")


def test_extension_accepts_arbitrary_code() -> None:
    ext = FlagExtension(code="custom_thing", message="hi", severity="info")
    assert ext.code == "custom_thing"
    assert ext.severity == "info"
