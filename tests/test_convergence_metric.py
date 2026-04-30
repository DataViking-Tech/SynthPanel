"""Tests for live convergence telemetry (sp-yaru).

Exercises the ConvergenceTracker directly (no LLM stack required) plus
the boundaries where the tracker integrates with CLI output and the
optional synthbench baseline dependency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from synth_panel.convergence import (
    DEFAULT_DIVERSITY_ENTROPY_THRESHOLD,
    DEFAULT_DIVERSITY_MIN_N,
    DEFAULT_EPSILON,
    ConvergenceTracker,
    SynthbenchUnavailableError,
    compute_calibration_jsd,
    derive_pick_one_schema_from_baseline,
    extract_categorical_responses,
    identify_tracked_questions,
    jensen_shannon_divergence,
    load_synthbench_baseline,
)
from synth_panel.structured.schemas import PICK_ONE_SCHEMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panelist(category_by_key: dict[str, str], tracked_keys: list[str]) -> Any:
    """Build a PanelistResult-shaped stub.

    ``category_by_key`` maps a question key to the extracted category
    label; anything missing becomes a text response the tracker will
    skip. Order of ``tracked_keys`` determines the main-question order.
    """
    responses = []
    for key in tracked_keys:
        if key in category_by_key:
            responses.append(
                {
                    "question": key,
                    "response": {"choice": category_by_key[key]},
                    "structured": True,
                }
            )
        else:
            responses.append({"question": key, "response": "free text"})
    return SimpleNamespace(responses=responses, error=None)


def _pick_one_question(key: str) -> dict[str, Any]:
    return {
        "text": f"Question {key}?",
        "key": key,
        "extraction_schema": "pick_one",
    }


# ---------------------------------------------------------------------------
# Acceptance-criteria-6 tests
# ---------------------------------------------------------------------------


def test_rolling_jsd_computed_after_k_panelists(capsys):
    """Distribution checks fire exactly every K completing panelists."""
    questions = [_pick_one_question("pricing")]
    tracked = identify_tracked_questions(questions)
    assert [k for _i, k, _q in tracked] == ["pricing"]

    tracker = ConvergenceTracker(tracked, check_every=5, epsilon=0.02, min_n=0, m_consecutive=2)
    for _ in range(4):
        tracker.record({"pricing": "A"})
    # Below K threshold — no check emitted yet.
    err_out = capsys.readouterr().err
    assert err_out == ""
    tracker.record({"pricing": "B"})

    err_out = capsys.readouterr().err.strip().splitlines()
    assert len(err_out) == 1
    payload = json.loads(err_out[0])
    assert payload["n_completed"] == 5
    assert "pricing" in payload["per_question"]
    reading = payload["per_question"]["pricing"]
    assert "jsd_last_batch" in reading
    assert "jsd_rolling_avg_3" in reading
    # JSD should be bounded in [0, 1].
    assert 0.0 <= reading["jsd_last_batch"] <= 1.0


def test_auto_stop_triggers_when_threshold_met():
    """Uniform distribution → rolling JSD → 0 → auto-stop fires."""
    questions = [_pick_one_question("choice")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(
        tracked,
        check_every=5,
        epsilon=0.05,
        min_n=10,
        m_consecutive=2,
        auto_stop=True,
    )
    stopped_at: int | None = None
    for i in range(100):
        # Deterministic cycle through 4 categories → perfectly uniform.
        cat = ["A", "B", "C", "D"][i % 4]
        if tracker.record({"choice": cat}):
            stopped_at = i + 1
            break

    assert stopped_at is not None, "tracker did not auto-stop on a stable uniform distribution"
    assert stopped_at >= 10, "auto-stop fired before min_n"
    assert tracker.auto_stopped is True
    assert tracker.overall_converged_at is not None
    assert tracker.overall_converged_at <= stopped_at


def test_auto_stop_respects_min_n():
    """A tiny min_n floor blocks premature auto-stop even when JSD is 0."""
    questions = [_pick_one_question("q")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(
        tracked,
        check_every=2,
        epsilon=DEFAULT_EPSILON,
        min_n=50,
        m_consecutive=2,
        auto_stop=True,
    )
    # Twenty identical records — JSD = 0 immediately, but min_n=50 should
    # keep auto_stop from firing.
    for _ in range(20):
        stopped = tracker.record({"q": "always-A"})
        assert stopped is False
    assert tracker.auto_stopped is False


def test_convergence_report_shape():
    """build_report returns the documented shape from the bead."""
    questions = [_pick_one_question("a"), _pick_one_question("b")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(
        tracked,
        check_every=3,
        epsilon=0.02,
        min_n=0,
        m_consecutive=2,
    )
    # Feed enough panelists to trigger at least a couple of checks so
    # converged_at has a chance to latch on question "a".
    for i in range(15):
        tracker.record({"a": "steady", "b": ["x", "y"][i % 2]})

    report = tracker.build_report()
    assert set(report.keys()) >= {
        "final_n",
        "check_every",
        "epsilon",
        "min_n",
        "m_consecutive",
        "auto_stopped",
        "overall_converged_at",
        "tracked_questions",
        "per_question",
        "human_baseline",
    }
    assert report["final_n"] == 15
    assert report["check_every"] == 3
    assert report["tracked_questions"] == ["a", "b"]
    for qkey in ("a", "b"):
        qrep = report["per_question"][qkey]
        assert qrep["final_n"] == 15
        assert isinstance(qrep["curve"], list)
        assert all(set(p.keys()) == {"n", "jsd"} for p in qrep["curve"])
        # Converged_at is either None or an integer <= final_n.
        ca = qrep["converged_at"]
        assert ca is None or (isinstance(ca, int) and ca <= 15)


def test_convergence_baseline_loaded_when_synthbench_available(monkeypatch):
    """When synthbench exposes a loader, the payload is returned verbatim."""
    fake_baseline = {"converged_at": 410, "curve": [{"n": 10, "jsd": 0.4}]}

    class _FakeSynthbench:
        @staticmethod
        def load_convergence_baseline(dataset: str, question_key: str | None = None) -> dict:
            assert dataset == "gss"
            assert question_key == "happiness"
            return dict(fake_baseline)

    monkeypatch.setitem(sys.modules, "synthbench", _FakeSynthbench())

    result = load_synthbench_baseline("gss:happiness")
    assert result["converged_at"] == 410
    assert result["dataset"] == "gss"
    assert result["question_key"] == "happiness"
    assert result["curve"] == fake_baseline["curve"]


def test_convergence_baseline_clear_error_when_synthbench_missing(monkeypatch):
    """Missing synthbench raises a typed error with install instructions."""
    # Force ``import synthbench`` to fail even if the user has it
    # installed in their dev env.
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "synthbench" or name.startswith("synthbench."):
            raise ImportError("No module named 'synthbench'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    # Clear any cached synthbench import from previous tests.
    for mod in list(sys.modules):
        if mod == "synthbench" or mod.startswith("synthbench."):
            monkeypatch.delitem(sys.modules, mod, raising=False)

    with pytest.raises(SynthbenchUnavailableError) as exc_info:
        load_synthbench_baseline("gss:happiness")
    msg = str(exc_info.value)
    assert "synthpanel[convergence]" in msg
    assert "pip install" in msg


# ---------------------------------------------------------------------------
# Support behaviour
# ---------------------------------------------------------------------------


def test_jensen_shannon_divergence_bounds():
    assert jensen_shannon_divergence({"a": 1}, {"a": 1}) == 0.0
    # Disjoint supports hit the JSD upper bound (1.0 in base-2).
    assert jensen_shannon_divergence({"a": 1}, {"b": 1}) == pytest.approx(1.0)
    # Empty → 0 (no-signal default).
    assert jensen_shannon_divergence({}, {"a": 1}) == 0.0


def test_identify_tracked_questions_ignores_free_text():
    questions = [
        {"text": "Free form?", "key": "free"},
        _pick_one_question("bounded"),
        {"text": "Enum?", "key": "enum", "response_schema": {"type": "enum", "options": ["x", "y"]}},
    ]
    tracked = identify_tracked_questions(questions)
    keys = [k for _i, k, _q in tracked]
    assert "free" not in keys
    assert "bounded" in keys
    assert "enum" in keys


def test_extract_categorical_skips_errored_responses():
    tracked = identify_tracked_questions([_pick_one_question("q")])
    pr = SimpleNamespace(
        responses=[
            {"question": "q", "response": "err", "error": True},
        ],
        error=None,
    )
    assert extract_categorical_responses(pr, tracked) == {}


# ---------------------------------------------------------------------------
# sp-5r88: derive_pick_one_schema_from_baseline
# ---------------------------------------------------------------------------


def test_derive_pick_one_schema_from_baseline_enum_keys_returns_sorted_enum_schema():
    baseline = {
        "human_distribution": {"chocolate": 0.5, "vanilla": 0.3, "strawberry": 0.2},
    }
    schema = derive_pick_one_schema_from_baseline(baseline)
    assert schema is not None
    # Deep copy — caller mutations must not leak back into the bundled schema.
    assert schema is not PICK_ONE_SCHEMA
    assert "enum" not in PICK_ONE_SCHEMA["properties"]["choice"]
    assert schema["properties"]["choice"]["enum"] == ["chocolate", "strawberry", "vanilla"]
    # Other properties of the bundled pick_one schema survive the derivation.
    assert schema["required"] == ["choice"]
    assert schema["properties"]["choice"]["type"] == "string"


@pytest.mark.parametrize(
    "distribution",
    [
        {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3, "5": 0.1},  # Likert strings
        {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.3, 5: 0.1},  # Likert ints
        {"3.5": 0.5, "4.5": 0.5},  # numeric-coercible floats
        {"red": 0.4, "2": 0.6},  # mixed: one numeric key poisons the lot
    ],
)
def test_derive_pick_one_schema_from_baseline_returns_none_for_numeric_keys(distribution):
    assert derive_pick_one_schema_from_baseline({"human_distribution": distribution}) is None


@pytest.mark.parametrize(
    "baseline",
    [
        {},  # no human_distribution at all
        {"human_distribution": None},
        {"human_distribution": {}},
        # Six enum keys — exceeds default max_options=5.
        {"human_distribution": {c: 1 / 6 for c in ["a", "b", "c", "d", "e", "f"]}},
    ],
)
def test_derive_pick_one_schema_from_baseline_returns_none_when_unfit(baseline):
    assert derive_pick_one_schema_from_baseline(baseline) is None


# ---------------------------------------------------------------------------
# sp-bldz: build_report calibration sub-object
# ---------------------------------------------------------------------------


def _record_happy_distribution(
    tracker: ConvergenceTracker,
    counts: dict[str, int],
    *,
    extra_keys: dict[str, str] | None = None,
) -> None:
    """Feed a tracker a known cumulative distribution for the ``happy`` key."""
    for category, count in counts.items():
        for _ in range(count):
            payload = {"happy": category}
            if extra_keys:
                payload.update(extra_keys)
            tracker.record(payload)


def test_build_report_attaches_calibration_subobject_with_provenance():
    questions = [_pick_one_question("happy")]
    tracked = identify_tracked_questions(questions)
    # check_every kept above n so the curve stays empty and the assertion
    # focuses on calibration shape, not rolling-jsd plumbing.
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    _record_happy_distribution(tracker, {"Very happy": 6, "Pretty happy": 3, "Not too happy": 1})
    baseline = {
        "dataset": "gss",
        "question_key": "HAPPY",
        "human_distribution": {
            "Very happy": 0.5,
            "Pretty happy": 0.4,
            "Not too happy": 0.1,
        },
    }

    report = tracker.build_report(
        baseline=baseline,
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )

    calib = report["per_question"]["happy"]["calibration"]
    assert set(calib.keys()) == {"jsd", "baseline_spec", "extractor", "auto_derived"}
    assert 0.0 <= calib["jsd"] <= 1.0
    assert calib["jsd"] == 0.008454  # rounded to 6 dp per wire format
    assert calib["baseline_spec"] == "gss:HAPPY"
    assert calib["extractor"] == "pick_one:auto-derived"
    assert calib["auto_derived"] is True
    # human_baseline still spliced in verbatim.
    assert report["human_baseline"] is baseline


def test_build_report_calibration_exact_match_yields_zero_jsd():
    questions = [_pick_one_question("happy")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    _record_happy_distribution(tracker, {"Very happy": 5, "Pretty happy": 3, "Not too happy": 2})
    baseline = {
        "human_distribution": {
            "Very happy": 0.5,
            "Pretty happy": 0.3,
            "Not too happy": 0.2,
        }
    }

    report = tracker.build_report(
        baseline=baseline,
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )

    assert report["per_question"]["happy"]["calibration"]["jsd"] == 0.0
    assert "alignment_error" not in report["per_question"]["happy"]["calibration"]


def test_build_report_calibration_disjoint_supports_emit_alignment_error():
    questions = [_pick_one_question("color")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    for _ in range(5):
        tracker.record({"color": "red"})
    for _ in range(5):
        tracker.record({"color": "blue"})

    baseline = {
        "human_distribution": {
            "Very happy": 0.5,
            "Pretty happy": 0.3,
            "Not too happy": 0.2,
        }
    }
    report = tracker.build_report(
        baseline=baseline,
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )

    calib = report["per_question"]["color"]["calibration"]
    assert calib["jsd"] == 1.0
    # Wire-format field per P-gate — sorted on both sides for determinism.
    assert calib["alignment_error"] == ("['blue', 'red'] vs ['Not too happy', 'Pretty happy', 'Very happy']")


def test_build_report_omits_calibration_when_spec_none():
    """Backward compatibility: omitting calibration_spec must not mutate the
    pre-change ``per_question[key]`` shape (structure.md §9, plan acceptance)."""
    questions = [_pick_one_question("happy")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    _record_happy_distribution(tracker, {"Very happy": 5, "Pretty happy": 5})

    report_default = tracker.build_report()
    report_with_baseline_only = tracker.build_report(
        baseline={"human_distribution": {"Very happy": 0.5, "Pretty happy": 0.5}}
    )

    for report in (report_default, report_with_baseline_only):
        entry = report["per_question"]["happy"]
        assert set(entry.keys()) == {"final_n", "converged_at", "curve", "support_size"}
        assert "calibration" not in entry


def test_build_report_omits_calibration_when_baseline_lacks_distribution():
    """``calibration_spec`` set but baseline carries no ``human_distribution``
    → calibration is silently absent (the loader is the only thing that can
    tell us the dataset shipped no distribution; we don't fabricate one)."""
    questions = [_pick_one_question("happy")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    _record_happy_distribution(tracker, {"Very happy": 3})

    report = tracker.build_report(
        baseline={"dataset": "gss", "question_key": "HAPPY"},
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )
    assert "calibration" not in report["per_question"]["happy"]


def test_build_report_calibration_matches_golden_fixture():
    """Pin the wire format. sp-pack-registry will fingerprint this shape;
    breaking changes here ripple downstream (structure.md §6, plan R3)."""
    fixture_path = Path(__file__).parent / "fixtures" / "convergence" / "calibration_report.json"
    expected = json.loads(fixture_path.read_text(encoding="utf-8"))["per_question"]

    questions = [_pick_one_question("happy"), _pick_one_question("color")]
    tracked = identify_tracked_questions(questions)
    tracker = ConvergenceTracker(tracked, check_every=100, min_n=0, m_consecutive=2)
    # 6 VH+red, 3 PH+blue, 1 NTH+red — drives "happy" support 3 & "color" support 2.
    sequence = (
        [{"happy": "Very happy", "color": "red"}] * 6
        + [{"happy": "Pretty happy", "color": "blue"}] * 3
        + [{"happy": "Not too happy", "color": "red"}] * 1
    )
    for record in sequence:
        tracker.record(record)

    baseline = {
        "human_distribution": {
            "Very happy": 0.5,
            "Pretty happy": 0.4,
            "Not too happy": 0.1,
        }
    }
    report = tracker.build_report(
        baseline=baseline,
        calibration_spec="gss:HAPPY",
        extractor_label="pick_one:auto-derived",
        auto_derived=True,
    )

    assert report["per_question"] == expected


def test_compute_calibration_jsd_wraps_local_implementation():
    """The wrapper must agree with the underlying JSD bit-for-bit so the
    structure.md §5 'do not import synthbench' decision stays observable."""
    p = {"a": 3.0, "b": 1.0}
    q = {"a": 0.5, "b": 0.5}
    assert compute_calibration_jsd(p, q) == jensen_shannon_divergence(p, q)
    # Disjoint supports hit the JSD upper bound.
    assert compute_calibration_jsd({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    # Empty model distribution → 0.0 (no panelist signal yet).
    assert compute_calibration_jsd({}, {"a": 1.0}) == 0.0


# ---------------------------------------------------------------------------
# Diversity warning tests (GH-291)
# ---------------------------------------------------------------------------


def _make_tracker(key: str = "q1", **kwargs: Any) -> ConvergenceTracker:
    questions = [_pick_one_question(key)]
    tracked = identify_tracked_questions(questions)
    return ConvergenceTracker(tracked, check_every=5, epsilon=0.02, min_n=0, m_consecutive=2, **kwargs)


def _feed(tracker: ConvergenceTracker, key: str, distribution: dict[str, int]) -> None:
    """Feed synthetic responses into the tracker according to distribution counts."""
    for choice, count in distribution.items():
        for _ in range(count):
            tracker.record({key: choice})


def test_diversity_warning_low_entropy():
    """90% concentration on one choice should produce a diversity warning."""
    tracker = _make_tracker(diversity_min_n=20)
    _feed(tracker, "q1", {"Foo": 27, "Bar": 1, "Baz": 1, "Qux": 1})
    report = tracker.build_report()
    warnings = report["diversity_warnings"]
    assert len(warnings) == 1
    assert "q1" in warnings[0]
    assert "90%" in warnings[0]
    assert "Foo" in warnings[0]


def test_diversity_no_warning_high_entropy():
    """Near-uniform distribution across 4 options should not trigger a warning."""
    tracker = _make_tracker(diversity_min_n=20)
    _feed(tracker, "q1", {"A": 10, "B": 10, "C": 10, "D": 10})
    report = tracker.build_report()
    assert report["diversity_warnings"] == []


def test_diversity_no_warning_below_min_n():
    """Fewer responses than diversity_min_n should never produce a warning."""
    tracker = _make_tracker(diversity_min_n=20)
    # Feed only 10 responses, all identical — would warn if min_n weren't checked.
    _feed(tracker, "q1", {"OnlyOne": 10})
    report = tracker.build_report()
    assert report["diversity_warnings"] == []


def test_diversity_custom_threshold_fires_on_moderate_concentration():
    """A raised threshold (0.8) should flag moderately-concentrated distributions."""
    tracker = _make_tracker(diversity_min_n=10, diversity_entropy_threshold=0.8)
    # 60/40 split — entropy ~0.97 bits / 1.0 = ~0.97 normalized. Still above 0.8.
    # Use a more concentrated split: 80/20 across 4 options.
    _feed(tracker, "q1", {"A": 16, "B": 2, "C": 1, "D": 1})
    report = tracker.build_report()
    assert len(report["diversity_warnings"]) == 1


def test_diversity_warnings_key_always_present():
    """build_report() always includes 'diversity_warnings', even when nothing fires."""
    tracker = _make_tracker(diversity_min_n=20)
    report = tracker.build_report()
    assert "diversity_warnings" in report
    assert isinstance(report["diversity_warnings"], list)


def test_diversity_defaults_exported():
    """The default constants must be importable and have sane values."""
    assert 0 < DEFAULT_DIVERSITY_ENTROPY_THRESHOLD < 1
    assert DEFAULT_DIVERSITY_MIN_N > 0
