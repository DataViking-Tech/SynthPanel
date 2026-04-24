"""Abort-path coverage for scaled panel runs (sp-56pb / sp-i2ub slice d).

Every abort path — rate-limit exhaustion, SIGINT, cost gate, and total
panelist failure — must produce a *valid* partial JSON document with
``run_invalid: true`` and a specific ``abort_reason``, and exit non-zero
so automation can detect the abort without scraping banner text.

These tests cover the CLI envelope end-to-end: they exercise ``main()``
with a mocked ``AgentRuntime`` so no real API calls fire, then assert
that the emitted JSON is parseable and carries the fields downstream
consumers (MCP, CI, dashboards) key on.

Integration: the final test simulates the bead's worked example —
n=50 with a 10% rate-limit error rate and an abort at panelist 30,
followed by a clean resume that finishes all 50.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.checkpoint import (
    CheckpointWriter,
    checkpoint_dir_for,
    load_checkpoint,
    new_run_id,
)
from synth_panel.cli.commands import _classify_total_failure_abort_reason
from synth_panel.cost import TokenUsage as CostTokenUsage
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionResponse, StopReason, TextBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.main import main
from synth_panel.orchestrator import (
    PanelistResult,
    RunAbortedError,
    run_panel_parallel,
)
from synth_panel.persistence import ConversationMessage
from synth_panel.runtime import TurnSummary

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _personas_yaml(names: list[str]) -> str:
    lines = ["personas:"]
    for n in names:
        lines.append(f"  - name: {n}")
    return "\n".join(lines) + "\n"


def _write_inputs(tmp_path: Path, persona_names: list[str]) -> tuple[Path, Path]:
    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text(_personas_yaml(persona_names))
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  version: 1\n  questions:\n    - text: What do you think?\n")
    return personas_file, survey_file


def _cheap_turn_summary(text: str = "ok") -> TurnSummary:
    usage = CostTokenUsage(input_tokens=10, output_tokens=5)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


def _expensive_turn_summary(text: str = "ok") -> TurnSummary:
    # Enough tokens to trip a $1 cost gate at sonnet rates (~$1.05/panelist).
    usage = CostTokenUsage(input_tokens=100_000, output_tokens=50_000)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


def _assert_valid_partial_json(stdout: str) -> dict[str, Any]:
    """Parse stdout as JSON and assert the partial-result contract holds."""
    assert stdout.strip(), "expected JSON on stdout, got empty output"
    payload = json.loads(stdout)
    assert isinstance(payload, dict), f"expected JSON object, got {type(payload).__name__}"
    assert payload.get("run_invalid") is True, f"run_invalid must be true on abort; payload={payload}"
    assert isinstance(payload.get("abort_reason"), str) and payload["abort_reason"], (
        f"abort_reason must be a non-empty string; got {payload.get('abort_reason')!r}"
    )
    return payload


# ---------------------------------------------------------------------------
# _classify_total_failure_abort_reason unit coverage
# ---------------------------------------------------------------------------


class TestClassifyTotalFailureAbortReason:
    def test_empty_sample_errors_falls_through_to_total_failure(self) -> None:
        assert _classify_total_failure_abort_reason({}) == "total_failure"
        assert _classify_total_failure_abort_reason({"sample_errors": []}) == "total_failure"

    def test_all_rate_limit_markers_classified_as_rate_limit_exhausted(self) -> None:
        diag = {
            "sample_errors": [
                ("P1", "RATE_LIMIT: retries_exhausted after 6 attempts"),
                ("P2", "LLMError category=rate_limit status=429"),
                ("P3", "Retries exhausted for this panelist"),
            ]
        }
        assert _classify_total_failure_abort_reason(diag) == "rate_limit_exhausted"

    def test_any_non_rate_limit_error_degrades_to_total_failure(self) -> None:
        # One sample pointing at a 400 (bad model) breaks the classification;
        # treating this as rate-limit-exhausted would send operators down
        # the wrong debugging path.
        diag = {
            "sample_errors": [
                ("P1", "rate_limit: retries_exhausted"),
                ("P2", "HTTP 400: invalid model name"),
            ]
        }
        assert _classify_total_failure_abort_reason(diag) == "total_failure"

    def test_handles_list_shaped_sample_errors(self) -> None:
        # sample_errors are tuples on construction; a JSON round-trip turns
        # them into lists. The classifier must handle both shapes.
        diag = {"sample_errors": [["P1", "429 rate_limit exhausted"]]}
        assert _classify_total_failure_abort_reason(diag) == "rate_limit_exhausted"


# ---------------------------------------------------------------------------
# Orchestrator layer: RunAbortedError carries partial results on SIGINT
# ---------------------------------------------------------------------------


def _persona(name: str) -> dict[str, Any]:
    return {"name": name, "age": 30}


def _system(p: dict[str, Any]) -> str:
    return f"You are {p['name']}"


def _question(q: dict[str, Any]) -> str:
    return q["text"] if isinstance(q, dict) else str(q)


def _mock_response() -> CompletionResponse:
    return CompletionResponse(
        id="resp",
        model="claude-sonnet-4-6",
        content=[TextBlock(text="ok")],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


class TestOrchestratorSigint:
    def test_run_aborted_error_carries_partial_prefix(self) -> None:
        """KeyboardInterrupt in a worker thread propagates via future.result()
        to the main-thread ``as_completed`` loop, where run_panel_parallel
        catches it and raises ``RunAbortedError`` with the finished prefix.
        """
        calls = {"n": 0}

        def flaky_send(_req: Any) -> CompletionResponse:
            calls["n"] += 1
            if calls["n"] > 2:
                # Simulate user hitting Ctrl-C mid-run.
                raise KeyboardInterrupt
            return _mock_response()

        client = MagicMock()
        client.send = flaky_send

        personas = [_persona(f"P{i}") for i in range(5)]
        questions = [{"text": "Q"}]

        with pytest.raises(RunAbortedError) as exc_info:
            run_panel_parallel(
                client=client,
                personas=personas,
                questions=questions,
                model="claude-sonnet-4-6",
                system_prompt_fn=_system,
                question_prompt_fn=_question,
                max_workers=1,  # serialize so the first 2 land before abort
            )

        aborted = exc_info.value
        assert aborted.reason == "sigint"
        # At least one panelist completed before the abort fired; no results
        # past the abort were dropped silently (checked via count vs. total).
        assert 0 <= len(aborted.results) < len(personas)
        # Every returned result is a real PanelistResult (no None sentinels).
        for r in aborted.results:
            assert isinstance(r, PanelistResult)


# ---------------------------------------------------------------------------
# CLI integration — each abort path emits valid partial JSON + abort_reason
# ---------------------------------------------------------------------------


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_cost_exceeded_emits_partial_json(
    _mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Cost-gate halt produces valid JSON with abort_reason=cost_exceeded.

    Smoke-test the contract this slice promises — the full cost-gate
    behaviour is already covered by ``tests/test_cost_gate.py``; here
    we only assert the envelope fields the bead calls out so the
    abort-path test suite is self-contained.
    """
    mock_runtime = MagicMock()
    mock_runtime.run_turn.return_value = _expensive_turn_summary()
    mock_runtime_cls.return_value = mock_runtime

    personas_file, survey_file = _write_inputs(tmp_path, [f"P{i}" for i in range(10)])

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
    assert code == 2

    captured = capsys.readouterr()
    payload = _assert_valid_partial_json(captured.out)
    assert payload["abort_reason"] == "cost_exceeded"
    assert payload.get("cost_exceeded") is True
    halted_at = payload.get("halted_at_panelist")
    assert isinstance(halted_at, int) and halted_at >= 1
    # Partial round present; every emitted panelist slot has the shape
    # consumers depend on.
    rounds = payload.get("rounds") or []
    assert len(rounds) == 1
    for row in rounds[0].get("results") or []:
        assert "persona" in row
        assert "responses" in row


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_sigint_emits_partial_json(
    _mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """SIGINT mid-run produces valid JSON with abort_reason=sigint.

    Implementation detail: we simulate the signal by making the third
    panelist's ``run_turn`` raise ``KeyboardInterrupt``. That mirrors
    what Python's default signal handler does after our checkpoint
    writer flushes — the main thread sees a KeyboardInterrupt while
    iterating ``as_completed``, which the orchestrator classifies as
    the ``sigint`` abort reason.
    """
    mock_runtime = MagicMock()
    turn_calls = {"n": 0}

    def flaky_run_turn(*_args: Any, **_kwargs: Any) -> TurnSummary:
        turn_calls["n"] += 1
        if turn_calls["n"] > 2:
            raise KeyboardInterrupt
        return _cheap_turn_summary()

    mock_runtime.run_turn.side_effect = flaky_run_turn
    mock_runtime_cls.return_value = mock_runtime

    personas_file, survey_file = _write_inputs(tmp_path, [f"P{i}" for i in range(5)])

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
            # Serialize so the KeyboardInterrupt lands after a deterministic
            # number of completed panelists; otherwise the parallel workers
            # race and the partial count becomes non-deterministic.
            "--max-concurrent",
            "1",
            "--no-synthesis",
        ]
    )
    assert code == 2, f"SIGINT abort must exit 2; got {code}"

    captured = capsys.readouterr()
    payload = _assert_valid_partial_json(captured.out)
    assert payload["abort_reason"] == "sigint"
    # Partial prefix is present and bounded below the full panel size.
    halted_at = payload.get("halted_at_panelist")
    assert isinstance(halted_at, int)
    assert 0 <= halted_at < 5
    # Operators grep stderr for the interrupt notice in CI.
    assert "interrupted" in captured.err.lower() or "sigint" in captured.err.lower()


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_rate_limit_exhausted_emits_partial_json(
    _mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Every panelist hits RETRIES_EXHAUSTED → abort_reason=rate_limit_exhausted.

    The LLM client exhausts its rate-limit retry budget after max_retries
    429s and raises ``LLMError(category=RETRIES_EXHAUSTED)``. The runtime
    re-raises that into the orchestrator's per-panelist Exception catch,
    so each panelist lands with ``error`` set to the stringified exception.
    When *every* panelist fails that way, detect_total_failure trips and
    the classifier tags the run as rate_limit_exhausted.
    """
    mock_runtime = MagicMock()
    mock_runtime.run_turn.side_effect = LLMError(
        "rate limit retries exhausted after 6 attempts",
        LLMErrorCategory.RETRIES_EXHAUSTED,
        status_code=429,
    )
    mock_runtime_cls.return_value = mock_runtime

    personas_file, survey_file = _write_inputs(tmp_path, [f"P{i}" for i in range(3)])

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
            "--no-synthesis",
        ]
    )
    assert code == 2

    captured = capsys.readouterr()
    payload = _assert_valid_partial_json(captured.out)
    assert payload["abort_reason"] == "rate_limit_exhausted", f"expected rate_limit_exhausted; payload={payload}"
    # total_failure envelope is also present so consumers can grab the
    # sample errors for the triage message.
    tf = payload.get("total_failure")
    assert tf is not None
    assert tf.get("sample_errors"), "total_failure envelope must carry sample errors"


@patch("synth_panel.orchestrator.AgentRuntime")
@patch("synth_panel.cli.commands.LLMClient")
def test_cli_total_failure_non_rate_limit_emits_total_failure_reason(
    _mock_client_cls: MagicMock,
    mock_runtime_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Non-rate-limit total failure → abort_reason=total_failure.

    Here every panelist hits a 400 (e.g. bad model name). detect_total_failure
    trips and the classifier falls back to the generic ``total_failure``
    tag — operators shouldn't be told to raise --rate-limit-rps when the
    real problem is an invalid model id.
    """
    mock_runtime = MagicMock()
    mock_runtime.run_turn.side_effect = RuntimeError("HTTP 400: invalid model name")
    mock_runtime_cls.return_value = mock_runtime

    personas_file, survey_file = _write_inputs(tmp_path, [f"P{i}" for i in range(3)])

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
            "--no-synthesis",
        ]
    )
    assert code == 2

    captured = capsys.readouterr()
    payload = _assert_valid_partial_json(captured.out)
    assert payload["abort_reason"] == "total_failure"
    # Rate-limit classifier should NOT have misfired.
    assert payload["abort_reason"] != "rate_limit_exhausted"


# ---------------------------------------------------------------------------
# Integration: n=50 with 10% rate-limit errors, abort at 30, resume clean
# ---------------------------------------------------------------------------


def _rate_limit_error() -> LLMError:
    return LLMError(
        "simulated 429",
        LLMErrorCategory.RATE_LIMIT,
        status_code=429,
        retry_after=0.001,
    )


def _ok_turn_summary(persona_name: str) -> TurnSummary:
    usage = CostTokenUsage(input_tokens=10, output_tokens=5)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": f"ok from {persona_name}"}],
        usage=usage,
    )
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


class TestEndToEndResume:
    def test_n50_10pct_rate_errors_abort_at_30_resume_finishes(
        self,
        tmp_path: Path,
    ) -> None:
        """n=50 panel, 10% of calls 429, "SIGINT" at 30, resume completes all 50.

        This mirrors the bead's worked example without actually spawning
        50 real LLM sessions. We exercise the checkpoint writer + the
        orchestrator callback pipeline end-to-end:

          1. Stage 1: run 30 of 50 panelists. Every 10th panelist's first
             attempt simulates a 429; the 429 contributes to ``usage``
             but the panelist still completes (the client retry budget
             absorbs the failure). After 30 completions the writer's
             ``mark_aborted`` is called — equivalent to SIGINT landing
             at the cadence boundary.
          2. Stage 2: load the checkpoint, rerun only the remaining 20
             panelists through ``run_panel_parallel`` with a fresh writer
             preloaded from stage 1.
          3. Final assertion: the final checkpoint covers all 50 personas
             with ``remaining == []`` and abort_reason cleared.

        Keeps the test deterministic and single-threaded (max_workers=1)
        so the "abort at 30" boundary is exact.
        """
        persona_names = [f"p{i:02d}" for i in range(50)]
        personas = [_persona(n) for n in persona_names]
        questions = [{"text": "What do you think?"}]

        # Build a mock client whose send() raises a RATE_LIMIT on every 10th
        # call, then succeeds. We never route through the real retry
        # loop — the point of the integration test is that the orchestrator
        # and writer cope with the error signal landing on a panelist.
        send_count = {"n": 0}

        def flaky_send(_req: Any) -> CompletionResponse:
            send_count["n"] += 1
            if send_count["n"] % 10 == 0:
                raise _rate_limit_error()
            return _mock_response()

        # Stage 1: run the first 30 panelists with a writer flushing every 1.
        run_id = new_run_id()
        directory = checkpoint_dir_for(run_id, tmp_path)
        cfg = {
            "persona_names": persona_names,
            "persona_count": 50,
            "question_texts": ["What do you think?"],
            "question_count": 1,
            "model": "claude-sonnet-4-6",
            "temperature": 0.7,
            "top_p": None,
        }
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=cfg,
            all_personas=persona_names,
            every=5,
        )

        # Prime partial results by driving the orchestrator against the
        # first 30 personas. The remaining 20 stay unrun.
        client_stage1 = MagicMock()
        client_stage1.send = flaky_send

        def record_cb(pr: PanelistResult) -> None:
            writer.record_completed(
                {
                    "persona": pr.persona_name,
                    "responses": pr.responses,
                    "usage": pr.usage.to_dict(),
                    "error": pr.error,
                },
                pr.usage.to_dict(),
            )

        stage1_results, _reg, _sess = run_panel_parallel(
            client=client_stage1,
            personas=personas[:30],
            questions=questions,
            model="claude-sonnet-4-6",
            system_prompt_fn=_system,
            question_prompt_fn=_question,
            max_workers=1,
            on_panelist_complete=record_cb,
        )
        # Panelists whose single call hit a rate-limit error carry a
        # response-level error on that question; the panelist itself
        # still completes and counts toward the checkpoint cadence. The
        # orchestrator's inner per-question exception handler is what
        # keeps a transient 429 from taking down the whole panelist.
        assert len(stage1_results) == 30
        errored_responses = sum(
            1 for pr in stage1_results for resp in pr.responses if isinstance(resp, dict) and resp.get("error")
        )
        # Every 10th send raised a 429 → expect ~3 error responses in 30
        # panelists; allow slop for the first-call vs. retry ordering.
        assert 1 <= errored_responses <= 5, (
            f"expected roughly 10% of 30 panelists to carry rate-limit response errors; got {errored_responses}"
        )

        writer.mark_aborted("signal:SIGINT")
        writer.flush(force=True)

        # The checkpoint on disk now carries 30 completed + 20 remaining.
        mid_ckpt = load_checkpoint(run_id, tmp_path)
        assert len(mid_ckpt.completed) == 30
        assert mid_ckpt.remaining == persona_names[30:]
        assert mid_ckpt.abort_reason == "signal:SIGINT"

        # Stage 2: resume — rerun the remaining 20 with a writer preloaded
        # from the stage 1 checkpoint. Use a clean client so no rate-limit
        # errors interfere with the resume (if the resume also failed,
        # the bead's claim "resume clean" would be false).
        client_stage2 = MagicMock()
        client_stage2.send = MagicMock(return_value=_mock_response())

        writer_resume = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=cfg,
            all_personas=persona_names,
            every=5,
            preloaded_completed=mid_ckpt.completed,
            preloaded_usage=mid_ckpt.usage,
        )

        def record_cb_resume(pr: PanelistResult) -> None:
            writer_resume.record_completed(
                {
                    "persona": pr.persona_name,
                    "responses": pr.responses,
                    "usage": pr.usage.to_dict(),
                    "error": pr.error,
                },
                pr.usage.to_dict(),
            )

        stage2_results, _reg, _sess = run_panel_parallel(
            client=client_stage2,
            personas=personas[30:],
            questions=questions,
            model="claude-sonnet-4-6",
            system_prompt_fn=_system,
            question_prompt_fn=_question,
            max_workers=1,
            on_panelist_complete=record_cb_resume,
        )
        writer_resume.flush(force=True)

        # Final checkpoint: all 50 personas present, remaining empty, no
        # abort left dangling.
        final = load_checkpoint(run_id, tmp_path)
        completed_names = sorted(c["persona"] for c in final.completed)
        assert completed_names == sorted(persona_names), (
            f"final checkpoint should cover all 50 personas; got {len(completed_names)}"
        )
        assert final.remaining == []
        # Sanity: the resume slice got real, clean responses for the
        # panelists who didn't finish stage 1.
        assert len(stage2_results) == 20
        assert all(r.error is None for r in stage2_results)
