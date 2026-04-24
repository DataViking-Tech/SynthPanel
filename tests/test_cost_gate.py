"""Tests for sp-utnk: mid-run --max-cost gate.

Covers three layers:

* ``CostGate`` unit semantics — projection math, halt-on-trip idempotency,
  thread safety, snapshot shape.
* Orchestrator integration — gate trips cancel pending futures and return
  the already-completed prefix of panelists.
* CLI integration — ``panel run --max-cost`` produces a valid partial
  JSON result with ``run_invalid``, ``cost_exceeded``, ``halted_at_panelist``
  and a non-zero exit code.
"""

from __future__ import annotations

import json
import textwrap
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import CostGate
from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.llm.models import CompletionResponse, StopReason, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.main import main
from synth_panel.orchestrator import run_panel_parallel
from synth_panel.persistence import ConversationMessage
from synth_panel.runtime import TurnSummary

# ---------------------------------------------------------------------------
# CostGate unit tests
# ---------------------------------------------------------------------------


class TestCostGateUnit:
    def test_rejects_non_positive_max_cost(self):
        with pytest.raises(ValueError):
            CostGate(max_cost_usd=0, total_panelists=10)
        with pytest.raises(ValueError):
            CostGate(max_cost_usd=-1.0, total_panelists=10)

    def test_rejects_non_positive_total(self):
        with pytest.raises(ValueError):
            CostGate(max_cost_usd=1.0, total_panelists=0)

    def test_no_halt_before_any_completion(self):
        gate = CostGate(max_cost_usd=1.00, total_panelists=10)
        assert gate.should_halt() is False
        assert gate.projected_total() == 0.0
        assert gate.completed == 0
        assert gate.running_cost == 0.0

    def test_single_completion_projects_linearly(self):
        """Projection = running / completed * total."""
        gate = CostGate(max_cost_usd=10.00, total_panelists=10)
        # First panelist costs $0.50 → projected total = 0.50 / 1 * 10 = $5.00
        halted = gate.record(0.50)
        assert halted is False
        assert gate.projected_total() == pytest.approx(5.00)
        assert gate.completed == 1
        assert gate.running_cost == pytest.approx(0.50)

    def test_halts_when_projection_exceeds_ceiling(self):
        gate = CostGate(max_cost_usd=1.00, total_panelists=10)
        # First panelist costs $0.20 → projected = 0.20 / 1 * 10 = $2.00 > $1.00
        halted = gate.record(0.20)
        assert halted is True
        assert gate.should_halt() is True
        assert gate.halted_projection == pytest.approx(2.00)

    def test_halt_is_idempotent(self):
        """Once halted the gate stays halted and records don't flip it back."""
        gate = CostGate(max_cost_usd=1.00, total_panelists=10)
        gate.record(0.20)  # trips
        first_proj = gate.halted_projection
        # Later sample with lower amortized cost still leaves gate halted.
        gate.record(0.0)
        assert gate.should_halt() is True
        # ``halted_projection`` captures the first trip, not later state.
        assert gate.halted_projection == first_proj

    def test_projection_tightens_with_more_samples(self):
        """Late samples change the projection but do not un-halt."""
        gate = CostGate(max_cost_usd=1.00, total_panelists=4)
        # $0.10 * 4 = $0.40 — safe
        assert gate.record(0.10) is False
        # running=$0.20, completed=2 → projected = 0.20 / 2 * 4 = $0.40 — safe
        assert gate.record(0.10) is False
        assert gate.should_halt() is False
        # Spike the third panelist: running=$0.80, completed=3 →
        # projected = 0.80 / 3 * 4 ≈ $1.067 > $1.00 — trips.
        assert gate.record(0.60) is True
        assert gate.should_halt() is True

    def test_negative_cost_clamped_to_zero(self):
        gate = CostGate(max_cost_usd=1.0, total_panelists=10)
        gate.record(-5.0)
        assert gate.running_cost == 0.0
        assert gate.should_halt() is False

    def test_snapshot_is_machine_readable(self):
        gate = CostGate(max_cost_usd=1.00, total_panelists=10)
        gate.record(0.20)  # halts
        snap = gate.snapshot()
        assert snap["halted"] is True
        assert snap["completed"] == 1
        assert snap["total_panelists"] == 10
        assert snap["max_cost_usd"] == pytest.approx(1.00)
        assert snap["running_cost_usd"] == pytest.approx(0.20)
        assert snap["projected_total_usd"] == pytest.approx(2.00)
        assert snap["halted_projection_usd"] == pytest.approx(2.00)

    def test_snapshot_when_not_halted(self):
        gate = CostGate(max_cost_usd=100.00, total_panelists=10)
        gate.record(0.01)
        snap = gate.snapshot()
        assert snap["halted"] is False
        assert snap["halted_projection_usd"] is None

    def test_thread_safety_concurrent_records(self):
        """Concurrent record() calls must not lose accounting."""
        gate = CostGate(max_cost_usd=1_000_000.0, total_panelists=10_000)
        n_threads = 32
        per_thread = 200
        barrier = threading.Barrier(n_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(per_thread):
                gate.record(0.001)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_completed = n_threads * per_thread
        assert gate.completed == expected_completed
        assert gate.running_cost == pytest.approx(expected_completed * 0.001, rel=1e-9)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def _heavy_response() -> CompletionResponse:
    """A response with enough tokens to have a non-trivial priced cost."""
    # sonnet pricing: $3/MTok input, $15/MTok output → 100k/50k ≈ $0.30 + $0.75 = $1.05 per panelist.
    return CompletionResponse(
        id="resp",
        model="claude-sonnet-4-6",
        content=[TextBlock(text="ok")],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=100_000, output_tokens=50_000),
    )


def _light_response() -> CompletionResponse:
    return CompletionResponse(
        id="resp",
        model="claude-sonnet-4-6",
        content=[TextBlock(text="ok")],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _system(p: dict[str, Any]) -> str:
    return f"You are {p['name']}"


def _question(q: dict[str, Any]) -> str:
    return q["text"]


class TestOrchestratorCostGateIntegration:
    def test_gate_halts_and_returns_partial_prefix(self):
        """When projection trips after first panelist, pending are cancelled.

        10 panelists, each priced around $1.05 at sonnet rates → projection
        after the first panelist is ~$10.50, which vastly exceeds the $1.00
        ceiling. The gate trips and we get back only the completed prefix.
        """
        client = MagicMock()
        client.send = MagicMock(return_value=_heavy_response())

        personas = [{"name": f"P{i}"} for i in range(10)]
        questions = [{"text": "Q1"}]
        gate = CostGate(max_cost_usd=1.00, total_panelists=10)

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="claude-sonnet-4-6",
            system_prompt_fn=_system,
            question_prompt_fn=_question,
            max_workers=1,  # serialize so cancellations actually bite
            cost_gate=gate,
        )

        assert gate.should_halt() is True
        # At least one completed (to trip the gate); fewer than all 10 (gate
        # cancelled the rest).
        assert 1 <= len(results) <= 9
        # Every returned result must be a real completion (not a cancel fallback).
        for r in results:
            assert r.error is None

    def test_gate_does_not_halt_when_cost_is_well_under_ceiling(self):
        """Light usage should let the full panel complete."""
        client = MagicMock()
        client.send = MagicMock(return_value=_light_response())

        personas = [{"name": f"P{i}"} for i in range(5)]
        questions = [{"text": "Q"}]
        # A generous ceiling — 5 panelists * light response << $100
        gate = CostGate(max_cost_usd=100.00, total_panelists=5)

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="claude-sonnet-4-6",
            system_prompt_fn=_system,
            question_prompt_fn=_question,
            cost_gate=gate,
        )

        assert gate.should_halt() is False
        assert len(results) == 5

    def test_no_gate_means_no_behaviour_change(self):
        """Sanity: omitting cost_gate keeps the existing fast path."""
        client = MagicMock()
        client.send = MagicMock(return_value=_heavy_response())

        personas = [{"name": f"P{i}"} for i in range(3)]
        questions = [{"text": "Q"}]

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="claude-sonnet-4-6",
            system_prompt_fn=_system,
            question_prompt_fn=_question,
        )

        assert len(results) == 3


# ---------------------------------------------------------------------------
# CLI integration — exercises --max-cost through main() with a mocked
# AgentRuntime so we can inject expensive usage without real API calls.
# ---------------------------------------------------------------------------


def _expensive_turn_summary(text: str = "expensive response") -> TurnSummary:
    """Turn summary with enough tokens to trip a $1 ceiling at sonnet rates.

    Sonnet: $3/MTok input, $15/MTok output.
    100k input + 50k output → $0.30 + $0.75 = $1.05 per turn.
    """
    usage = CostTokenUsage(input_tokens=100_000, output_tokens=50_000)
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


def _cheap_turn_summary(text: str = "cheap response") -> TurnSummary:
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
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_max_cost_emits_valid_partial_json(
    mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """End-to-end: --max-cost trips, CLI emits parseable JSON with the
    expected ``run_invalid`` / ``cost_exceeded`` / ``halted_at_panelist``
    fields and exits 2.
    """
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _expensive_turn_summary()
    mock_runtime_cls.return_value = mock_runtime

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
              - name: P6
              - name: P7
              - name: P8
              - name: P9
              - name: P10
            """
        ).strip()
    )
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: What do you think?\n")

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
            "--max-cost",
            "1.00",
            "--max-concurrent",
            "1",
            "--no-synthesis",
        ]
    )

    captured = capsys.readouterr()
    assert code == 2, f"expected exit 2 on cost-gate halt; got {code}\nstdout={captured.out!r}\nstderr={captured.err!r}"

    payload = json.loads(captured.out)
    assert payload.get("run_invalid") is True
    assert payload.get("cost_exceeded") is True
    assert payload.get("abort_reason") == "cost_exceeded"
    halted_at = payload.get("halted_at_panelist")
    assert isinstance(halted_at, int) and halted_at >= 1

    # Cost-gate diagnostics surfaced at top level.
    gate_snap = payload.get("cost_gate")
    assert gate_snap is not None
    assert gate_snap["halted"] is True
    assert gate_snap["completed"] == halted_at
    assert gate_snap["max_cost_usd"] == pytest.approx(1.00)
    assert gate_snap["projected_total_usd"] > 1.00

    # Partial result array present — every returned persona has a response
    # or an error record, and the count matches the halt point.
    rounds = payload.get("rounds") or []
    assert len(rounds) == 1
    round_results = rounds[0].get("results") or []
    assert len(round_results) == halted_at, (
        f"round result count {len(round_results)} should equal halted_at_panelist={halted_at}"
    )
    for r in round_results:
        assert "persona" in r
        assert "responses" in r

    # Operators grep stderr for the halt banner in CI.
    assert "cost" in captured.err.lower() and "halt" in captured.err.lower()


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_below_ceiling_completes_normally(
    mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Generous --max-cost must not disturb a normal panel run."""
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _cheap_turn_summary()
    mock_runtime_cls.return_value = mock_runtime

    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n  - name: Carol\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: Q?\n")

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
            "--max-cost",
            "100.00",
            "--no-synthesis",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload.get("run_invalid") is False
    assert "cost_exceeded" not in payload
    assert payload.get("persona_count") == 3


def test_cli_rejects_non_positive_max_cost(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: P1\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: Q?\n")

    code = main(
        [
            "panel",
            "run",
            "--personas",
            str(personas_file),
            "--instrument",
            str(survey_file),
            "--max-cost",
            "0",
            "--no-synthesis",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "max-cost" in err.lower()
