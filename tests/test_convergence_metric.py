"""Tests for live convergence telemetry (sp-yaru).

Exercises the ConvergenceTracker directly (no LLM stack required) plus
the boundaries where the tracker integrates with CLI output and the
optional synthbench baseline dependency.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from synth_panel.convergence import (
    DEFAULT_EPSILON,
    ConvergenceTracker,
    SynthbenchUnavailableError,
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
