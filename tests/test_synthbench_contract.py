"""Cross-repo contract test: SynthPanel submission payloads vs SynthBench's validator.

This is the guard for the product→benchmark data funnel (sp-ezz, gap-analysis
P0-2): a payload built by ``build_submission_payload`` from a realistic
calibrated panel run must pass SynthBench's *actual* ``validate_submission``
with zero ERROR-severity issues in tier 1 (schema + plausibility) AND
tier 2 (metric recomputation). Historically the payload failed tier 1 with
6+ schema errors, so no SynthPanel run ever validly reached the leaderboard.

Dependency note: the ``synthbench`` distribution on PyPI is an UNRELATED
project (synthetic ML datasets); the real harness is GitHub-only
(github.com/DataViking-Tech/SynthBench). Tests that need the real harness
use ``pytest.importorskip("synthbench.validation")`` — the submodule probe
skips gracefully both when synthbench is absent and when the wrong package
shadows the name. CI runs these tests for real in the dedicated
``synthbench-contract`` job (see .github/workflows/ci.yml), which installs
the harness from git.

The vendored-metric tests at the bottom always run: they pin the vendored
JSD / Kendall's tau-b implementations to the synthbench conventions using
the same fixture values as SynthBench's own tests/test_metrics.py.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from synth_panel.convergence import ConvergenceTracker, identify_tracked_questions
from synth_panel.instrument import parse_instrument
from synth_panel.synthbench_submit import (
    _vendored_jsd,
    _vendored_kendall_tau_b,
    build_submission_payload,
)

# GSS HAPPY aggregate proportions (approximate; the contract only needs a
# plausible, properly normalized human distribution).
_GSS_HAPPY_HUMAN = {"Very happy": 0.31, "Pretty happy": 0.56, "Not too happy": 0.13}


def _happiness_probe_questions() -> list[dict[str, Any]]:
    """Load the *bundled* happiness-probe pack — the instrument the README's
    ``--submit-to-synthbench`` recipe uses — and return its flat questions."""
    from synth_panel.mcp.data import _bundled_instrument_packs

    pack = _bundled_instrument_packs()["happiness-probe"]
    instrument = parse_instrument(dict(pack["instrument"]))
    return instrument.questions


def _calibrated_run_payload(
    *,
    extra_questions: list[dict[str, Any]] | None = None,
    baseline_question_key: str = "GSS_HAPPY",
) -> dict[str, Any]:
    """Simulate a calibrated 60-panelist run of the happiness-probe pack and
    build the submission payload exactly the way the CLI does:
    ``build_report`` → ``cumulative_distributions`` → ``build_submission_payload``.
    """
    questions = _happiness_probe_questions() + list(extra_questions or [])
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=20, min_n=0)

    # A realistic (non-degenerate, non-perfect) panel distribution. The rank
    # order deliberately differs from the human baseline (LLM panels tend to
    # over-index "Very happy") so kendall_tau < 1 and the tier-2 recompute
    # paths are exercised on non-trivial values.
    panel_counts = {"Very happy": 28, "Pretty happy": 23, "Not too happy": 9}
    satjob_cycle = ["Satisfied", "Dissatisfied", "Satisfied"]
    i = 0
    for category, count in panel_counts.items():
        for _ in range(count):
            record = {"HAPPY": category}
            if extra_questions:
                record["SATJOB"] = satjob_cycle[i % len(satjob_cycle)]
            tracker.record(record)
            i += 1

    baseline = {
        "dataset": "gss",
        "question_key": baseline_question_key,
        "human_distribution": dict(_GSS_HAPPY_HUMAN),
        "redistribution_policy": "full",
    }
    report = tracker.build_report(
        baseline=baseline,
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )
    model_distributions = tracker.cumulative_distributions()
    tracker.close()

    return build_submission_payload(
        panel_extra={"convergence": report, "run_invalid": False},
        calibration_spec="gss:HAPPY",
        baseline_payload=baseline,
        model_distributions=model_distributions,
        panelist_model="claude-haiku-4-5-20251001",
        instrument_name="happiness-probe",
        persona_pack_name="general-public",
    )


# ---------------------------------------------------------------------------
# The contract: payload passes SynthBench's real validate() (tiers 1 + 2)
# ---------------------------------------------------------------------------


def test_submission_payload_passes_synthbench_validator_tier1_and_tier2():
    sb_validation = pytest.importorskip(
        "synthbench.validation",
        reason="real synthbench harness not installed (GitHub-only; PyPI 'synthbench' is unrelated)",
    )

    payload = _calibrated_run_payload()
    # Round-trip through JSON first: the wire payload is what the server
    # validates, and this also catches non-JSON-serializable values.
    wire_payload = json.loads(json.dumps(payload))

    report = sb_validation.validate_submission(
        wire_payload,
        source="synthpanel --submit-to-synthbench",
        tier1=True,
        tier2=True,
    )
    assert report.errors == [], f"SynthBench validator rejected the payload:\n{report.format()}"
    assert report.ok


def test_multi_question_run_submits_only_the_calibrated_question():
    """An instrument with >1 bounded question uploads exactly one row —
    the one bound to the baseline — and still passes the validator
    (previously every question was stamped with the same human baseline,
    fabricating garbage JSD rows)."""
    sb_validation = pytest.importorskip(
        "synthbench.validation",
        reason="real synthbench harness not installed (GitHub-only; PyPI 'synthbench' is unrelated)",
    )

    extra = [
        {
            "key": "SATJOB",
            "text": "On the whole, how satisfied are you with the work you do?",
            "response_schema": {"type": "pick_one", "options": ["Satisfied", "Dissatisfied"]},
        }
    ]
    payload = _calibrated_run_payload(extra_questions=extra)

    assert [row["key"] for row in payload["per_question"]] == ["HAPPY"]
    assert payload["aggregate"]["n_questions"] == 1

    report = sb_validation.validate_submission(
        json.loads(json.dumps(payload)), source="synthpanel multi-question", tier1=True, tier2=True
    )
    assert report.errors == [], f"SynthBench validator rejected the payload:\n{report.format()}"


def test_tampered_aggregate_is_rejected_by_tier2():
    """Negative control: the validator this suite runs must actually be able
    to reject — an inflated composite_parity fails tier 2."""
    sb_validation = pytest.importorskip(
        "synthbench.validation",
        reason="real synthbench harness not installed (GitHub-only; PyPI 'synthbench' is unrelated)",
    )

    payload = _calibrated_run_payload()
    honest = payload["aggregate"]["composite_parity"]
    payload["aggregate"]["composite_parity"] = round(min(1.0, honest + 0.2), 6)  # fabricated
    report = sb_validation.validate_submission(payload, source="tampered", tier1=True, tier2=True)
    assert any(issue.code == "AGG_COMPOSITE" for issue in report.errors), report.format()


def test_per_question_metrics_are_tier2_identities():
    """jsd / kendall_tau are computed from the exact submitted distributions
    with synthbench's own functions, so the server-side recompute agrees to
    rounding precision (far inside the 3e-2 tier-2 tolerance)."""
    distributional = pytest.importorskip(
        "synthbench.metrics.distributional",
        reason="real synthbench harness not installed (GitHub-only; PyPI 'synthbench' is unrelated)",
    )
    ranking = pytest.importorskip("synthbench.metrics.ranking")

    payload = _calibrated_run_payload()
    assert payload["per_question"], "expected at least one submitted question"
    for row in payload["per_question"]:
        recomputed_jsd = distributional.jensen_shannon_divergence(row["human_distribution"], row["model_distribution"])
        recomputed_tau = ranking.kendall_tau_b(row["human_distribution"], row["model_distribution"])
        assert row["jsd"] == pytest.approx(recomputed_jsd, abs=1e-6)
        assert row["kendall_tau"] == pytest.approx(recomputed_tau, abs=1e-6)


def test_payload_version_matches_synthbench_harness_version():
    """When the real harness is importable, the top-level ``version`` stamp
    mirrors what ``synthbench run`` itself writes (its ``__version__``)."""
    synthbench = pytest.importorskip("synthbench")
    pytest.importorskip("synthbench.validation")

    payload = _calibrated_run_payload()
    assert payload["version"] == str(synthbench.__version__)


# ---------------------------------------------------------------------------
# Vendored metric parity (fallback path when synthbench is not installed)
# ---------------------------------------------------------------------------

# Distribution pairs covering the shapes SynthBench's own tests/test_metrics.py
# exercises: identical, disjoint, shifted mass, reversed rank order, ties,
# missing keys, empty distributions.
_METRIC_FIXTURE_PAIRS: list[tuple[dict[str, float], dict[str, float]]] = [
    ({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}),
    ({"a": 1.0}, {"b": 1.0}),
    ({"a": 0.3, "b": 0.7}, {"a": 0.4, "b": 0.6}),
    ({"a": 0.1, "b": 0.2, "c": 0.7}, {"a": 0.7, "b": 0.2, "c": 0.1}),
    ({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}),
    ({"a": 0.6, "b": 0.4}, {"a": 0.6, "b": 0.3, "c": 0.1}),
    (
        {"Very happy": 0.31, "Pretty happy": 0.56, "Not too happy": 0.13},
        {"Very happy": 19 / 60, "Pretty happy": 31 / 60, "Not too happy": 10 / 60},
    ),
    ({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.2, "b": 0.3, "c": 0.5}),
    ({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}),
    ({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.3, "c": 0.2}),
]


def test_vendored_metrics_match_synthbench_implementations():
    """Parity: the pure-Python fallbacks agree with synthbench's scipy-backed
    metrics on every fixture pair (both argument orders)."""
    distributional = pytest.importorskip(
        "synthbench.metrics.distributional",
        reason="real synthbench harness not installed (GitHub-only; PyPI 'synthbench' is unrelated)",
    )
    ranking = pytest.importorskip("synthbench.metrics.ranking")

    for p, q in _METRIC_FIXTURE_PAIRS:
        for x, y in ((p, q), (q, p)):
            assert _vendored_jsd(x, y) == pytest.approx(distributional.jensen_shannon_divergence(x, y), abs=1e-9), (
                x,
                y,
            )
            assert _vendored_kendall_tau_b(x, y) == pytest.approx(ranking.kendall_tau_b(x, y), abs=1e-9), (x, y)


# Known values mirroring SynthBench's tests/test_metrics.py — these always
# run so the fallback path is guarded even in environments without the
# real harness installed.


def test_vendored_jsd_known_values():
    p = {"a": 0.5, "b": 0.5}
    assert _vendored_jsd(p, p) == pytest.approx(0.0, abs=1e-10)
    # Disjoint supports hit the base-2 upper bound.
    assert _vendored_jsd({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0, abs=1e-6)
    # Symmetric.
    q = {"a": 0.2, "b": 0.8}
    assert _vendored_jsd(p, q) == pytest.approx(_vendored_jsd(q, p), abs=1e-10)
    # Bounded in [0, 1].
    assert 0.0 < _vendored_jsd(p, q) <= 1.0
    # Empty / zero-mass distribution → 1.0 (synthbench convention; NOTE this
    # deliberately differs from synth_panel.convergence.jensen_shannon_divergence,
    # which returns 0.0 for "no signal yet").
    assert _vendored_jsd({}, {"a": 1.0}) == 1.0
    assert _vendored_jsd({"a": 0.0}, {"a": 1.0}) == 1.0


def test_vendored_kendall_tau_known_values():
    p = {"a": 0.5, "b": 0.3, "c": 0.2}
    # Identical ranking → +1.
    assert _vendored_kendall_tau_b(p, p) == pytest.approx(1.0, abs=1e-10)
    # Same order, different magnitudes → +1.
    q_same_order = {"a": 0.6, "b": 0.25, "c": 0.15}
    assert _vendored_kendall_tau_b(p, q_same_order) == pytest.approx(1.0, abs=1e-10)
    # Perfect reversal → -1.
    q_reversed = {"a": 0.2, "b": 0.3, "c": 0.5}
    assert _vendored_kendall_tau_b(p, q_reversed) == pytest.approx(-1.0, abs=1e-10)
    # Constant input over the union support → 0.0 (scipy returns NaN;
    # synthbench maps it to 0.0).
    assert _vendored_kendall_tau_b({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, p) == 0.0
    # Fewer than 2 comparable options → 0.0.
    assert _vendored_kendall_tau_b({"a": 1.0}, {"a": 1.0}) == 0.0
    # Missing keys read as 0.0 mass (union support), matching synthbench:
    # {"a": .5, "b": .5} over {a, b, c} is [0.5, 0.5, 0.0] — not constant.
    assert _vendored_kendall_tau_b({"a": 0.5, "b": 0.5}, p) == pytest.approx(0.816496580927726, abs=1e-9)


# ---------------------------------------------------------------------------
# Payload shape guards that need no synthbench install
# ---------------------------------------------------------------------------


def test_payload_shape_matches_tier1_contract_without_synthbench():
    """The structural half of the contract, runnable everywhere: list-shaped
    per_question with the REQUIRED_PER_QUESTION fields, required config and
    aggregate keys, and a top-level version."""
    payload = _calibrated_run_payload()

    for key in ("benchmark", "version", "config", "aggregate", "per_question"):
        assert key in payload, f"missing required top-level key {key!r}"
    assert payload["benchmark"] == "synthbench"
    for key in ("dataset", "provider"):
        assert key in payload["config"], f"missing required config key {key!r}"
    for key in ("mean_jsd", "mean_kendall_tau", "composite_parity", "n_questions"):
        assert key in payload["aggregate"], f"missing required aggregate key {key!r}"

    per_question = payload["per_question"]
    assert isinstance(per_question, list) and per_question
    assert payload["aggregate"]["n_questions"] == len(per_question)
    for row in per_question:
        for key in ("key", "human_distribution", "model_distribution", "jsd", "kendall_tau"):
            assert key in row, f"missing required per-question key {key!r}"
        for dist_field in ("human_distribution", "model_distribution"):
            total = sum(row[dist_field].values())
            assert total == pytest.approx(1.0, abs=5e-3), f"{dist_field} sums to {total}"
