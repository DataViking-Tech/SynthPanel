"""Tests for sp-xw2z6o: per-question failure budget.

Three layers, mirroring tests/test_cost_gate.py:

* ``QuestionFailureBudget`` unit semantics — threshold math (int vs
  fractional), idempotence, thread safety, snapshot shape.
* Orchestrator integration — once a question's budget is tripped,
  subsequent panelists skip that question via ``skipped_by_budget``
  responses; their other questions still run.
* CLI integration — ``panel run --question-failure-budget`` produces a
  valid JSON envelope with ``disabled_questions``,
  ``question_failure_budget`` snapshot, and a non-inflated
  ``failure_stats`` (budget skips don't count as failures or as
  missing answers, per #314 framing).
"""

from __future__ import annotations

import json
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.main import main
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import ConversationMessage
from synth_panel.question_budget import QuestionFailureBudget
from synth_panel.runtime import TurnSummary

# ---------------------------------------------------------------------------
# QuestionFailureBudget unit tests
# ---------------------------------------------------------------------------


class TestQuestionFailureBudgetUnit:
    def test_rejects_non_positive_total(self):
        with pytest.raises(ValueError):
            QuestionFailureBudget(2, 0)
        with pytest.raises(ValueError):
            QuestionFailureBudget(2, -1)

    def test_rejects_zero_or_negative_int_budget(self):
        with pytest.raises(ValueError):
            QuestionFailureBudget(0, 5)
        with pytest.raises(ValueError):
            QuestionFailureBudget(-1, 5)

    def test_rejects_out_of_range_fraction(self):
        with pytest.raises(ValueError):
            QuestionFailureBudget(0.0, 5)
        with pytest.raises(ValueError):
            QuestionFailureBudget(1.0, 5)
        with pytest.raises(ValueError):
            QuestionFailureBudget(1.5, 5)

    def test_rejects_bool_budget(self):
        with pytest.raises(TypeError):
            QuestionFailureBudget(True, 5)  # type: ignore[arg-type]

    def test_int_budget_disables_at_count(self):
        b = QuestionFailureBudget(2, 10)
        # First failure: still under threshold.
        assert b.record_failure(0) is False
        assert b.is_disabled(0) is False
        # Second failure: trips disable.
        assert b.record_failure(0) is True
        assert b.is_disabled(0) is True

    def test_threshold_count_is_exposed(self):
        assert QuestionFailureBudget(3, 10).threshold_count == 3

    def test_fractional_budget_rounds_up(self):
        """0.25 of 4 panelists → ceil(1.0) = 1; first failure trips."""
        b = QuestionFailureBudget(0.25, 4)
        assert b.threshold_count == 1
        assert b.fraction == pytest.approx(0.25)
        assert b.record_failure(0) is True

    def test_fractional_budget_with_larger_panel(self):
        """0.25 of 8 → ceil(2.0) = 2; first under, second trips."""
        b = QuestionFailureBudget(0.25, 8)
        assert b.threshold_count == 2
        assert b.record_failure(0) is False
        assert b.record_failure(0) is True

    def test_fractional_budget_floor_on_small_panel(self):
        """Tiny fraction on small panel still requires at least 1 failure."""
        b = QuestionFailureBudget(0.1, 5)
        assert b.threshold_count == 1

    def test_disable_is_idempotent(self):
        b = QuestionFailureBudget(2, 5)
        b.record_failure(0)
        b.record_failure(0)
        assert b.is_disabled(0) is True
        # Further failures don't change the snapshot or undo the disable.
        b.record_failure(0)
        assert b.is_disabled(0) is True

    def test_independent_questions_track_separately(self):
        b = QuestionFailureBudget(2, 10)
        b.record_failure(0)
        b.record_failure(1)
        assert b.is_disabled(0) is False
        assert b.is_disabled(1) is False
        # Trip Q0 only — Q1 stays alive.
        b.record_failure(0)
        assert b.is_disabled(0) is True
        assert b.is_disabled(1) is False

    def test_disabled_details_carry_question_text(self):
        b = QuestionFailureBudget(1, 5)
        b.record_failure(3, question_text="What's the capital of Mars?")
        details = b.disabled_details()
        assert len(details) == 1
        assert details[0]["question_index"] == 3
        assert details[0]["question_text"] == "What's the capital of Mars?"
        assert details[0]["failures_at_disable"] == 1
        assert details[0]["threshold_count"] == 1

    def test_snapshot_machine_readable(self):
        b = QuestionFailureBudget(0.25, 4)
        b.record_failure(2, question_text="Q3")
        snap = b.snapshot()
        assert snap["threshold_count"] == 1
        assert snap["threshold_fraction"] == pytest.approx(0.25)
        assert snap["total_panelists"] == 4
        assert snap["disabled_count"] == 1
        assert snap["disabled"][0]["question_index"] == 2
        assert snap["failure_counts"][2] == 1

    def test_thread_safety_concurrent_records(self):
        """Concurrent record() calls converge to a consistent disable state.

        Once the question is disabled, ``record_failure`` short-circuits
        without incrementing the counter (subsequent panelists shouldn't
        keep paying to log into a disabled question), so we can only
        assert that the counter at disable time is exactly the threshold.
        Without the lock a torn read/write would either over- or
        under-count and miss the disable trigger entirely.
        """
        b = QuestionFailureBudget(50, 10_000)
        n_threads = 32
        per_thread = 50
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(per_thread):
                b.record_failure(0)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert b.is_disabled(0) is True
        details = b.disabled_details()
        assert len(details) == 1
        # The disable must trigger at exactly the threshold count — racing
        # increments past the threshold without disabling would mean the
        # lock lost the trigger.
        assert details[0]["failures_at_disable"] == 50


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def _system(p: dict[str, Any]) -> str:
    return f"You are {p['name']}"


def _question(q: dict[str, Any]) -> str:
    return q["text"]


def _ok_turn_summary(text: str = "fine") -> TurnSummary:
    usage = CostTokenUsage(input_tokens=10, output_tokens=5)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(
        assistant_messages=[msg],
        iterations=1,
        usage=usage,
    )


@patch("synth_panel.orchestrator.AgentRuntime")
def test_orchestrator_skips_disabled_question_for_later_panelists(
    mock_runtime_cls: MagicMock,
) -> None:
    """One question always fails; budget=2 means panelists 1-2 fail it,
    panelists 3+ skip it via skipped_by_budget."""
    questions = [{"text": "Q1"}, {"text": "BAD_Q"}, {"text": "Q3"}]

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()

        def run_turn(prompt: str) -> TurnSummary:
            if prompt == "BAD_Q":
                raise RuntimeError("schema mismatch")
            return _ok_turn_summary("ok")

        runtime.run_turn.side_effect = run_turn
        return runtime

    mock_runtime_cls.side_effect = runtime_factory

    personas = [{"name": f"P{i}"} for i in range(5)]
    budget = QuestionFailureBudget(2, total_panelists=5)
    client = MagicMock()

    results, _reg, _sessions = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,  # serialize so disable order is deterministic
        question_budget=budget,
    )

    assert len(results) == 5
    assert budget.is_disabled(1) is True
    assert budget.is_disabled(0) is False
    assert budget.is_disabled(2) is False

    # First two panelists hit the bad question and recorded an error.
    for pr in results[:2]:
        bad_resp = pr.responses[1]
        assert bad_resp.get("error") is True
        assert bad_resp.get("skipped_by_budget") is not True

    # Remaining panelists short-circuit Q1 (the BAD_Q).
    for pr in results[2:]:
        bad_resp = pr.responses[1]
        assert bad_resp.get("skipped_by_budget") is True
        assert bad_resp.get("error") is not True
        assert bad_resp.get("response") is None

    # Other questions still ran for every panelist.
    for pr in results:
        assert pr.responses[0].get("response") == "ok"
        assert pr.responses[2].get("response") == "ok"


@patch("synth_panel.orchestrator.AgentRuntime")
def test_orchestrator_no_budget_means_no_skip_marker(
    mock_runtime_cls: MagicMock,
) -> None:
    """Sanity: omitting question_budget keeps the existing fast path —
    every question runs for every panelist, even on repeated failure."""

    def runtime_factory(*args, **kwargs):
        runtime = MagicMock()

        def run_turn(prompt: str) -> TurnSummary:
            if prompt == "BAD":
                raise RuntimeError("oops")
            return _ok_turn_summary()

        runtime.run_turn.side_effect = run_turn
        return runtime

    mock_runtime_cls.side_effect = runtime_factory
    client = MagicMock()
    personas = [{"name": f"P{i}"} for i in range(3)]
    questions = [{"text": "OK"}, {"text": "BAD"}]

    results, _reg, _sessions = run_panel_parallel(
        client=client,
        personas=personas,
        questions=questions,
        model="claude-sonnet-4-6",
        system_prompt_fn=_system,
        question_prompt_fn=_question,
        max_workers=1,
    )

    assert len(results) == 3
    for pr in results:
        # Every panelist hit the bad question — no skip markers.
        bad_resp = pr.responses[1]
        assert bad_resp.get("error") is True
        assert "skipped_by_budget" not in bad_resp


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def _selective_failure_summary(prompt: str) -> TurnSummary:
    """Fail when the prompt matches the bad-question text."""
    if "BROKEN" in prompt:
        raise RuntimeError("model rejected schema")
    return _ok_turn_summary("good")


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_question_budget_short_circuits_and_emits_disabled_questions(
    mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """End-to-end: instrument with one always-failing question, budget=2.
    Run completes, JSON contains disabled_questions + question_failure_budget,
    failure_stats reports skipped_by_budget separately and does NOT count
    those skips as failures or as missing pairs (per #314 framing)."""

    runtime = MagicMock()
    runtime.run_turn.side_effect = _selective_failure_summary
    mock_runtime_cls.return_value = runtime

    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text(
        textwrap.dedent(
            """
            personas:
              - name: P1
              - name: P2
              - name: P3
              - name: P4
              - name: P5
            """
        ).strip()
    )
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text(
        textwrap.dedent(
            """
            instrument:
              version: 1
              questions:
                - text: Tell me about your morning.
                - text: BROKEN_QUESTION_THE_MODEL_HATES
                - text: What's your favorite tool?
            """
        ).strip()
    )

    code = main(
        [
            "--model",
            "claude-sonnet-4-6",
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(personas_file),
            "--instrument",
            str(survey_file),
            "--question-failure-budget",
            "2",
            "--max-concurrent",
            "1",
            "--no-synthesis",
        ]
    )
    captured = capsys.readouterr()

    assert code == 0, f"run with disabled question should still exit 0; got {code}\nstderr={captured.err!r}"

    payload = json.loads(captured.out)

    # Acceptance: disabled-question state is surfaced in JSON.
    disabled = payload.get("disabled_questions") or []
    assert len(disabled) == 1
    assert disabled[0]["question_index"] == 1
    assert "BROKEN" in (disabled[0]["question_text"] or "")
    assert disabled[0]["failures_at_disable"] == 2
    assert disabled[0]["threshold_count"] == 2

    # Snapshot of the budget itself for dashboards.
    snap = payload.get("question_failure_budget")
    assert snap is not None
    assert snap["threshold_count"] == 2
    assert snap["disabled_count"] == 1

    # Acceptance (#314 framing): budget skips do NOT inflate failure stats.
    fs = payload["failure_stats"]
    # Two real failures (P1 + P2 hit the bad question); P3-P5 skipped it.
    assert fs["errored_pairs"] == 2
    assert fs["skipped_by_budget"] == 3
    # Only the two ok questions per panelist count toward total_pairs:
    # 5 panelists * 2 ok questions + 2 panelists * 1 errored question = 12.
    assert fs["total_pairs"] == 12
    # Failure rate = 2 / 12 ≈ 0.167 — well below the default 0.5 threshold.
    assert fs["failure_rate"] == pytest.approx(2 / 12, rel=1e-6)
    assert payload.get("run_invalid") is False

    # Banner mentions the disabled question.
    assert "Disabled mid-run" in captured.err
    assert "Q2" in captured.err  # 1-indexed display


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_fractional_budget(
    mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Fractional --question-failure-budget 0.25 with 4 panelists → 1 failure trips."""

    runtime = MagicMock()
    runtime.run_turn.side_effect = _selective_failure_summary
    mock_runtime_cls.return_value = runtime

    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: A\n  - name: B\n  - name: C\n  - name: D\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: BROKEN\n    - text: Hi\n")

    code = main(
        [
            "--model",
            "claude-sonnet-4-6",
            "--output-format",
            "json",
            "panel",
            "run",
            "--personas",
            str(personas_file),
            "--instrument",
            str(survey_file),
            "--question-failure-budget",
            "0.25",
            "--max-concurrent",
            "1",
            "--no-synthesis",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    disabled = payload.get("disabled_questions") or []
    assert len(disabled) == 1
    assert disabled[0]["threshold_fraction"] == pytest.approx(0.25)
    # ceil(0.25 * 4) = 1 → first failure trips → 3 panelists skip
    assert payload["failure_stats"]["skipped_by_budget"] == 3
    assert payload["failure_stats"]["errored_pairs"] == 1


def test_cli_rejects_invalid_budget_values(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: P1\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: Q?\n")

    base = [
        "panel",
        "run",
        "--personas",
        str(personas_file),
        "--instrument",
        str(survey_file),
        "--no-synthesis",
        "--question-failure-budget",
    ]

    code = main([*base, "0"])
    assert code == 1
    err = capsys.readouterr().err
    assert "question-failure-budget" in err.lower()

    code = main([*base, "1.5"])
    assert code == 1
    err = capsys.readouterr().err
    assert "question-failure-budget" in err.lower()

    code = main([*base, "abc"])
    assert code == 1
    err = capsys.readouterr().err
    assert "question-failure-budget" in err.lower()
