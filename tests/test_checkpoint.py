"""Panelist-level checkpointing tests (sp-hsk3 / sp-i2ub slice b).

Covers:
  - Periodic flush every K completed panelists.
  - Signal-triggered flush on SIGINT.
  - Resume skips already-completed panelists and merges results.
  - Config fingerprint drift is rejected on resume.
  - Abort reason is preserved and surfaced on resume.
  - Usage accumulation across resume boundary.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.checkpoint import (
    CheckpointDriftError,
    CheckpointFormatError,
    CheckpointNotFoundError,
    CheckpointWriter,
    PanelCheckpoint,
    _merge_usage,
    checkpoint_dir_for,
    ensure_config_matches,
    fingerprint_config,
    load_checkpoint,
    new_run_id,
    save_checkpoint,
)
from synth_panel.llm.models import (
    CompletionResponse,
    StopReason,
    TextBlock,
)
from synth_panel.llm.models import (
    TokenUsage as LLMTokenUsage,
)
from synth_panel.orchestrator import PanelistResult, run_panel_parallel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "ok") -> CompletionResponse:
    return CompletionResponse(
        id="resp",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.send = MagicMock(return_value=_make_response())
    return client


def _persona(name: str) -> dict[str, Any]:
    return {"name": name, "age": 30}


def _system_prompt(persona: dict[str, Any]) -> str:
    return f"You are {persona['name']}."


def _question_prompt(question: dict[str, Any]) -> str:
    return question["text"] if isinstance(question, dict) else str(question)


def _make_config(persona_names: list[str]) -> dict[str, Any]:
    return {
        "persona_names": persona_names,
        "persona_count": len(persona_names),
        "question_texts": ["What do you think?"],
        "question_count": 1,
        "model": "sonnet",
        "temperature": 0.7,
        "top_p": None,
    }


# ---------------------------------------------------------------------------
# Fingerprint + config drift
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_is_deterministic(self) -> None:
        config = _make_config(["Alice", "Bob"])
        assert fingerprint_config(config) == fingerprint_config(dict(config))

    def test_fingerprint_sensitive_to_persona_list(self) -> None:
        a = fingerprint_config(_make_config(["Alice", "Bob"]))
        b = fingerprint_config(_make_config(["Alice", "Carol"]))
        assert a != b

    def test_fingerprint_sensitive_to_model(self) -> None:
        cfg_a = _make_config(["Alice"])
        cfg_b = dict(cfg_a)
        cfg_b["model"] = "haiku"
        assert fingerprint_config(cfg_a) != fingerprint_config(cfg_b)

    def test_ensure_config_matches_accepts_identical(self) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-x",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        ensure_config_matches(ckpt, cfg)  # no raise

    def test_ensure_config_matches_rejects_drift(self) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-x",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        drifted = dict(cfg)
        drifted["model"] = "haiku"
        with pytest.raises(CheckpointDriftError):
            ensure_config_matches(ckpt, drifted)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice", "Bob"])
        ckpt = PanelCheckpoint(
            run_id="run-abc",
            created_at="2026-04-24T00:00:00Z",
            updated_at="2026-04-24T00:00:00Z",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
            completed=[{"persona": "Alice", "responses": [], "usage": {}, "error": None}],
            remaining=["Bob"],
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        directory = checkpoint_dir_for("run-abc", tmp_path)
        save_checkpoint(ckpt, directory)
        loaded = load_checkpoint("run-abc", tmp_path)
        assert loaded.run_id == "run-abc"
        assert loaded.completed[0]["persona"] == "Alice"
        assert loaded.remaining == ["Bob"]
        assert loaded.usage["input_tokens"] == 10
        assert loaded.config_fingerprint == ckpt.config_fingerprint

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CheckpointNotFoundError):
            load_checkpoint("run-does-not-exist", tmp_path)

    def test_load_malformed_raises(self, tmp_path: Path) -> None:
        directory = checkpoint_dir_for("run-bad", tmp_path)
        directory.mkdir(parents=True)
        (directory / "state.json").write_text("not valid json")
        with pytest.raises(CheckpointFormatError):
            load_checkpoint("run-bad", tmp_path)

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-atomic",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        directory = checkpoint_dir_for("run-atomic", tmp_path)
        save_checkpoint(ckpt, directory)
        save_checkpoint(ckpt, directory)  # rewrite should not leave tmp files
        leftovers = [p for p in directory.iterdir() if p.name.startswith(".ckpt-")]
        assert leftovers == []


# ---------------------------------------------------------------------------
# Merge usage helper
# ---------------------------------------------------------------------------


class TestMergeUsage:
    def test_integers_sum(self) -> None:
        merged = _merge_usage(
            {"input_tokens": 10, "output_tokens": 5},
            {"input_tokens": 3, "output_tokens": 2},
        )
        assert merged["input_tokens"] == 13
        assert merged["output_tokens"] == 7

    def test_provider_cost_sums(self) -> None:
        merged = _merge_usage(
            {"provider_reported_cost": 0.10},
            {"provider_reported_cost": 0.05},
        )
        assert merged["provider_reported_cost"] == pytest.approx(0.15)

    def test_provider_cost_both_none_stays_absent(self) -> None:
        merged = _merge_usage({"input_tokens": 1}, {"output_tokens": 1})
        assert "provider_reported_cost" not in merged


# ---------------------------------------------------------------------------
# CheckpointWriter cadence + remaining
# ---------------------------------------------------------------------------


class TestCheckpointWriter:
    def test_flushes_every_k_panelists(self, tmp_path: Path) -> None:
        run_id = "run-cadence"
        directory = checkpoint_dir_for(run_id, tmp_path)
        cfg = _make_config(["a", "b", "c", "d", "e"])
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=cfg,
            all_personas=["a", "b", "c", "d", "e"],
            every=2,
        )
        for name in ["a", "b"]:
            writer.record_completed(
                {"persona": name, "responses": [], "usage": {"input_tokens": 1}, "error": None},
                {"input_tokens": 1},
            )
        # 2 completions at every=2 → should have flushed
        assert (directory / "state.json").exists()
        loaded = load_checkpoint(run_id, tmp_path)
        assert {c["persona"] for c in loaded.completed} == {"a", "b"}
        assert loaded.remaining == ["c", "d", "e"]
        assert loaded.usage["input_tokens"] == 2

    def test_flush_no_op_before_first_record(self, tmp_path: Path) -> None:
        run_id = "run-empty"
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a"]),
            all_personas=["a"],
            every=25,
        )
        writer.flush()
        assert not (directory / "state.json").exists()

    def test_force_flush_writes_even_without_records(self, tmp_path: Path) -> None:
        run_id = "run-force"
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a"]),
            all_personas=["a"],
            every=25,
        )
        writer.flush(force=True)
        assert (directory / "state.json").exists()

    def test_idempotent_duplicate_records(self, tmp_path: Path) -> None:
        run_id = "run-dupe"
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a", "b"]),
            all_personas=["a", "b"],
            every=1,
        )
        record = {"persona": "a", "responses": [], "usage": {"input_tokens": 1}, "error": None}
        writer.record_completed(record, {"input_tokens": 1})
        writer.record_completed(record, {"input_tokens": 1})  # duplicate
        loaded = load_checkpoint(run_id, tmp_path)
        assert sum(1 for c in loaded.completed if c["persona"] == "a") == 1
        # usage should only have counted once
        assert loaded.usage["input_tokens"] == 1

    def test_preloaded_state_preserved_and_extended(self, tmp_path: Path) -> None:
        run_id = "run-preload"
        directory = checkpoint_dir_for(run_id, tmp_path)
        preloaded = [{"persona": "a", "responses": [], "usage": {"input_tokens": 10}, "error": None}]
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a", "b", "c"]),
            all_personas=["a", "b", "c"],
            every=1,
            preloaded_completed=preloaded,
            preloaded_usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert writer.remaining() == ["b", "c"]
        writer.record_completed(
            {"persona": "b", "responses": [], "usage": {"input_tokens": 3}, "error": None},
            {"input_tokens": 3},
        )
        loaded = load_checkpoint(run_id, tmp_path)
        assert {c["persona"] for c in loaded.completed} == {"a", "b"}
        assert loaded.remaining == ["c"]
        assert loaded.usage["input_tokens"] == 13
        assert loaded.usage["output_tokens"] == 5


# ---------------------------------------------------------------------------
# Orchestrator callback integration
# ---------------------------------------------------------------------------


class TestOrchestratorCallback:
    def test_callback_invoked_for_every_panelist(self, tmp_path: Path) -> None:
        client = _mock_client()
        personas = [_persona(n) for n in ("Alice", "Bob", "Carol")]
        questions = [{"text": "What do you think?"}]
        seen: list[str] = []
        lock = threading.Lock()

        def cb(pr: PanelistResult) -> None:
            with lock:
                seen.append(pr.persona_name)

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            on_panelist_complete=cb,
        )
        assert len(results) == 3
        assert set(seen) == {"Alice", "Bob", "Carol"}

    def test_callback_exceptions_do_not_kill_run(self) -> None:
        client = _mock_client()
        personas = [_persona("Alice"), _persona("Bob")]
        questions = [{"text": "Q"}]

        def cb(pr: PanelistResult) -> None:
            raise RuntimeError("boom")

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            on_panelist_complete=cb,
        )
        assert len(results) == 2
        assert all(r.error is None for r in results)


# ---------------------------------------------------------------------------
# End-to-end: run → "SIGINT" → resume completes all panelists
# ---------------------------------------------------------------------------


def _simulate_run_with_abort(
    *,
    tmp_path: Path,
    persona_names: list[str],
    abort_after: int,
) -> tuple[str, list[PanelistResult]]:
    """Drive a partial run whose checkpoint gets flushed after ``abort_after``.

    Mirrors what happens when SIGINT fires mid-run: the writer has already
    flushed up through ``abort_after`` because we set ``every=1``. We then
    stop the run without completing the remaining panelists. This is the
    equivalent of a SIGINT landing at the same cadence boundary, but lets
    the test stay single-threaded and deterministic.
    """
    run_id = new_run_id()
    directory = checkpoint_dir_for(run_id, tmp_path)
    cfg = _make_config(persona_names)
    writer = CheckpointWriter(
        run_id=run_id,
        directory=directory,
        config=cfg,
        all_personas=persona_names,
        every=1,
    )
    partial: list[PanelistResult] = []
    for name in persona_names[:abort_after]:
        pr = PanelistResult(
            persona_name=name,
            responses=[{"question": "Q", "response": f"A from {name}"}],
            usage=LLMTokenUsage(input_tokens=7, output_tokens=3).__class__
            and LLMTokenUsage(input_tokens=7, output_tokens=3),
        )
        # We're reusing LLMTokenUsage, but PanelistResult.usage is
        # synth_panel.cost.TokenUsage — convert.
        from synth_panel.cost import TokenUsage as CostTokenUsage

        pr.usage = CostTokenUsage(input_tokens=7, output_tokens=3)
        partial.append(pr)
        writer.record_completed(
            {
                "persona": name,
                "responses": pr.responses,
                "usage": pr.usage.to_dict(),
                "error": None,
            },
            pr.usage.to_dict(),
        )
    writer.mark_aborted("signal:SIGINT")
    writer.flush(force=True)
    return run_id, partial


class TestResumeEndToEnd:
    def test_sigint_at_7_resume_finishes_all_10(self, tmp_path: Path) -> None:
        persona_names = [f"p{i:02d}" for i in range(10)]
        run_id, _partial = _simulate_run_with_abort(tmp_path=tmp_path, persona_names=persona_names, abort_after=7)

        # Resume path: load checkpoint, verify config, rerun remaining.
        ckpt = load_checkpoint(run_id, tmp_path)
        assert ckpt.abort_reason == "signal:SIGINT"
        assert {c["persona"] for c in ckpt.completed} == set(persona_names[:7])
        assert ckpt.remaining == persona_names[7:]

        cfg = _make_config(persona_names)
        ensure_config_matches(ckpt, cfg)

        # Rerun only the remaining personas via orchestrator with a fresh
        # writer that carries the preloaded slice.
        remaining_personas = [_persona(n) for n in ckpt.remaining]
        client = _mock_client()
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=cfg,
            all_personas=persona_names,
            every=25,
            preloaded_completed=ckpt.completed,
            preloaded_usage=ckpt.usage,
        )
        assert writer.remaining() == persona_names[7:]

        def cb(pr: PanelistResult) -> None:
            writer.record_completed(
                {
                    "persona": pr.persona_name,
                    "responses": pr.responses,
                    "usage": pr.usage.to_dict(),
                    "error": pr.error,
                },
                pr.usage.to_dict(),
            )

        results, _reg, _sessions = run_panel_parallel(
            client=client,
            personas=remaining_personas,
            questions=[{"text": "What do you think?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            on_panelist_complete=cb,
        )
        writer.flush(force=True)
        assert sorted(r.persona_name for r in results) == sorted(persona_names[7:])

        # Final checkpoint: contains all 10 panelists.
        final = load_checkpoint(run_id, tmp_path)
        assert sorted(c["persona"] for c in final.completed) == sorted(persona_names)
        assert final.remaining == []

    def test_resume_rejects_config_drift(self, tmp_path: Path) -> None:
        run_id, _ = _simulate_run_with_abort(tmp_path=tmp_path, persona_names=["a", "b", "c"], abort_after=1)
        ckpt = load_checkpoint(run_id, tmp_path)
        drifted = _make_config(["a", "b", "c"])
        drifted["model"] = "haiku"  # user changed model between runs
        with pytest.raises(CheckpointDriftError):
            ensure_config_matches(ckpt, drifted)


# ---------------------------------------------------------------------------
# Signal handler trap (main thread only)
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_sigint_triggers_flush_and_propagates(self, tmp_path: Path) -> None:
        """Send SIGINT to ourselves → writer flushes, KeyboardInterrupt fires."""
        run_id = "run-sig"
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a", "b", "c"]),
            all_personas=["a", "b", "c"],
            every=25,
        )
        writer.install_signal_handlers()
        try:
            # Record one completion so flush has something to write.
            writer.record_completed(
                {"persona": "a", "responses": [], "usage": {"input_tokens": 1}, "error": None},
                {"input_tokens": 1},
            )
            with pytest.raises(KeyboardInterrupt):
                os.kill(os.getpid(), signal.SIGINT)
                # The default SIGINT handler raises KeyboardInterrupt
                # synchronously after our handler returns. Give the
                # interpreter a tick to deliver the signal.
                time.sleep(0.05)
            assert (directory / "state.json").exists()
            loaded = load_checkpoint(run_id, tmp_path)
            assert loaded.abort_reason == "signal:SIGINT"
            assert {c["persona"] for c in loaded.completed} == {"a"}
        finally:
            writer.remove_signal_handlers()

    def test_install_handlers_idempotent(self, tmp_path: Path) -> None:
        run_id = "run-sig-idem"
        writer = CheckpointWriter(
            run_id=run_id,
            directory=checkpoint_dir_for(run_id, tmp_path),
            config=_make_config(["a"]),
            all_personas=["a"],
            every=1,
        )
        writer.install_signal_handlers()
        writer.install_signal_handlers()  # no-op, no raise
        writer.remove_signal_handlers()
        writer.remove_signal_handlers()  # no-op, no raise


# ---------------------------------------------------------------------------
# CLI helpers smoke test
# ---------------------------------------------------------------------------


class TestCLIHelpers:
    def test_panelist_result_round_trips_through_dict(self) -> None:
        from synth_panel.cli.commands import (
            _panelist_result_from_dict,
            _panelist_result_to_ckpt_dict,
        )
        from synth_panel.cost import TokenUsage as CostTokenUsage

        pr = PanelistResult(
            persona_name="Alice",
            responses=[{"question": "Q", "response": "A"}],
            usage=CostTokenUsage(input_tokens=10, output_tokens=5),
            error=None,
            model="sonnet",
        )
        record = _panelist_result_to_ckpt_dict(pr, fallback_model="sonnet")
        # Survives a JSON round trip (important for on-disk persistence).
        record = json.loads(json.dumps(record))
        restored = _panelist_result_from_dict(record)
        assert restored.persona_name == "Alice"
        assert restored.usage.input_tokens == 10
        assert restored.usage.output_tokens == 5
        assert restored.responses == [{"question": "Q", "response": "A"}]
        assert restored.model == "sonnet"

    def test_build_run_config_fingerprint_is_stable(self) -> None:
        from synth_panel.cli.commands import _build_run_config_fingerprint

        cfg_a = _build_run_config_fingerprint(
            personas=[{"name": "Alice"}, {"name": "Bob"}],
            questions=[{"text": "Q"}],
            model="sonnet",
            persona_models=None,
            temperature=0.7,
            top_p=None,
            response_schema=None,
            extract_schema=None,
            template_vars=None,
        )
        cfg_b = _build_run_config_fingerprint(
            personas=[{"name": "Alice"}, {"name": "Bob"}],
            questions=[{"text": "Q"}],
            model="sonnet",
            persona_models=None,
            temperature=0.7,
            top_p=None,
            response_schema=None,
            extract_schema=None,
            template_vars=None,
        )
        assert fingerprint_config(cfg_a) == fingerprint_config(cfg_b)
