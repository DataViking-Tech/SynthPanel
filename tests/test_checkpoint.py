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
    CheckpointSchemaTooNewError,
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
from synth_panel.metadata_migrations import (
    CURRENT_SCHEMA_VERSION,
    migrate_to_current,
    migrate_v1_to_v2,
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


# ---------------------------------------------------------------------------
# sy-ws76: --resume <run-id> standalone CLI entry point
# ---------------------------------------------------------------------------


class TestCheckpointCliArgs:
    """``cli_args`` is the field that backs bare ``--resume <id>`` flow."""

    def test_cli_args_round_trips_through_dict(self) -> None:
        cfg = _make_config(["a"])
        ckpt = PanelCheckpoint(
            run_id="run-cli-args",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
            cli_args={"personas": "/tmp/p.yaml", "instrument": "/tmp/i.yaml"},
        )
        d = ckpt.to_dict()
        assert d["cli_args"] == {"personas": "/tmp/p.yaml", "instrument": "/tmp/i.yaml"}
        restored = PanelCheckpoint.from_dict(json.loads(json.dumps(d)))
        assert restored.cli_args == {"personas": "/tmp/p.yaml", "instrument": "/tmp/i.yaml"}

    def test_cli_args_round_trips_through_disk(self, tmp_path: Path) -> None:
        cfg = _make_config(["a"])
        ckpt = PanelCheckpoint(
            run_id="run-cli-disk",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
            cli_args={
                "personas": "examples/personas.yaml",
                "instrument": "examples/survey.yaml",
                "personas_merge": ["a.yaml", "b.yaml"],
                "personas_merge_on_collision": "error",
            },
        )
        directory = checkpoint_dir_for("run-cli-disk", tmp_path)
        save_checkpoint(ckpt, directory)
        loaded = load_checkpoint("run-cli-disk", tmp_path)
        assert loaded.cli_args is not None
        assert loaded.cli_args["personas"] == "examples/personas.yaml"
        assert loaded.cli_args["instrument"] == "examples/survey.yaml"
        assert loaded.cli_args["personas_merge"] == ["a.yaml", "b.yaml"]
        assert loaded.cli_args["personas_merge_on_collision"] == "error"

    def test_legacy_checkpoint_without_cli_args_still_loads(self, tmp_path: Path) -> None:
        # Pre-sy-ws76 checkpoints predate the cli_args field. They must
        # still load — backwards compat for the existing on-disk format.
        directory = checkpoint_dir_for("run-legacy", tmp_path)
        directory.mkdir(parents=True)
        cfg = _make_config(["a"])
        legacy = {
            "version": 1,
            "run_id": "run-legacy",
            "created_at": "now",
            "updated_at": "now",
            "config_fingerprint": fingerprint_config(cfg),
            "config": cfg,
            "completed": [],
            "remaining": ["a"],
            "usage": {},
            "abort_reason": None,
            # No cli_args field at all.
        }
        (directory / "state.json").write_text(json.dumps(legacy))
        loaded = load_checkpoint("run-legacy", tmp_path)
        assert loaded.cli_args is None

    def test_writer_persists_cli_args(self, tmp_path: Path) -> None:
        run_id = "run-writer-cli-args"
        directory = checkpoint_dir_for(run_id, tmp_path)
        writer = CheckpointWriter(
            run_id=run_id,
            directory=directory,
            config=_make_config(["a"]),
            all_personas=["a"],
            every=1,
            cli_args={"personas": "examples/p.yaml", "instrument": "examples/i.yaml"},
        )
        writer.flush(force=True)
        loaded = load_checkpoint(run_id, tmp_path)
        assert loaded.cli_args == {"personas": "examples/p.yaml", "instrument": "examples/i.yaml"}

    def test_build_resume_cli_args_captures_fields(self) -> None:
        from argparse import Namespace

        from synth_panel.cli.commands import _build_resume_cli_args

        ns = Namespace(
            personas="examples/p.yaml",
            instrument="examples/i.yaml",
            personas_merge=["a.yaml", "b.yaml"],
            personas_merge_on_collision="error",
        )
        snapshot = _build_resume_cli_args(ns)
        assert snapshot["personas"] == "examples/p.yaml"
        assert snapshot["instrument"] == "examples/i.yaml"
        assert snapshot["personas_merge"] == ["a.yaml", "b.yaml"]
        assert snapshot["personas_merge_on_collision"] == "error"

    def test_apply_resume_cli_args_fills_missing_only(self) -> None:
        from argparse import Namespace

        from synth_panel.cli.commands import _apply_resume_cli_args

        # User omitted both --personas and --instrument.
        ns = Namespace(
            personas=None,
            instrument=None,
            personas_merge=[],
            personas_merge_on_collision="dedup",
        )
        _apply_resume_cli_args(
            ns,
            {
                "personas": "saved/p.yaml",
                "instrument": "saved/i.yaml",
                "personas_merge": ["m.yaml"],
                "personas_merge_on_collision": "error",
            },
        )
        assert ns.personas == "saved/p.yaml"
        assert ns.instrument == "saved/i.yaml"
        assert ns.personas_merge == ["m.yaml"]
        assert ns.personas_merge_on_collision == "error"

    def test_apply_resume_cli_args_user_explicit_wins(self) -> None:
        from argparse import Namespace

        from synth_panel.cli.commands import _apply_resume_cli_args

        # User passed --personas explicitly; saved value must NOT clobber.
        ns = Namespace(
            personas="user/explicit.yaml",
            instrument=None,
            personas_merge=[],
            personas_merge_on_collision="dedup",
        )
        _apply_resume_cli_args(
            ns,
            {"personas": "saved/p.yaml", "instrument": "saved/i.yaml"},
        )
        assert ns.personas == "user/explicit.yaml"  # unchanged
        assert ns.instrument == "saved/i.yaml"  # filled in


# ---------------------------------------------------------------------------
# End-to-end CLI: synthpanel panel run --resume <id> (no --personas/--instrument)
# ---------------------------------------------------------------------------


def _personas_yaml() -> str:
    return "personas:\n  - name: Alice\n  - name: Bob\n  - name: Carol\n"


def _instrument_yaml() -> str:
    return "instrument:\n  questions:\n    - text: Q1\n"


def _seed_partial_checkpoint(
    *,
    tmp_path: Path,
    run_id: str,
    persona_names: list[str],
    completed_names: list[str],
    personas_path: str,
    instrument_path: str,
) -> Path:
    """Drop a checkpoint on disk that mirrors a SIGINT'd CLI run.

    Uses ``_build_run_config_fingerprint`` so the saved fingerprint
    matches what a real ``handle_panel_run`` will compute for the same
    personas + questions on resume.
    """
    from synth_panel.cli.commands import _build_run_config_fingerprint

    cfg = _build_run_config_fingerprint(
        personas=[{"name": n} for n in persona_names],
        questions=[{"text": "Q1"}],
        model="sonnet",
        persona_models=None,
        temperature=None,
        top_p=None,
        response_schema=None,
        extract_schema=None,
        template_vars=None,
    )
    directory = checkpoint_dir_for(run_id, tmp_path)
    completed = [
        {
            "persona": n,
            "responses": [{"question": "Q1", "response": f"answer from {n}"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "cost": "$0.0001",
            "error": None,
        }
        for n in completed_names
    ]
    ckpt = PanelCheckpoint(
        run_id=run_id,
        created_at="2026-04-30T00:00:00Z",
        updated_at="2026-04-30T00:00:00Z",
        config_fingerprint=fingerprint_config(cfg),
        config=cfg,
        completed=completed,
        remaining=[n for n in persona_names if n not in completed_names],
        usage={"input_tokens": 7 * len(completed_names), "output_tokens": 3 * len(completed_names)},
        abort_reason="signal:SIGINT",
        cli_args={
            "personas": personas_path,
            "instrument": instrument_path,
            "personas_merge": [],
            "personas_merge_on_collision": "dedup",
        },
    )
    save_checkpoint(ckpt, directory)
    return directory


class TestResumeFromCheckpointCli:
    """Bare ``synthpanel panel run --resume <id>`` recovers paths and skips done panelists.

    These tests dispatch ``handle_panel_run`` directly rather than ``main()``
    so they don't trigger ``setup_logging`` and pollute downstream
    capsys-based tests in other files.
    """

    @staticmethod
    def _parsed(argv: list[str]):
        from synth_panel.cli.parser import build_parser

        return build_parser().parse_args(argv)

    @staticmethod
    def _fmt():
        from synth_panel.cli.output import OutputFormat

        return OutputFormat("text")

    def test_resume_without_personas_or_instrument_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        from synth_panel.cli.commands import handle_panel_run
        from synth_panel.cost import TokenUsage as CostTokenUsage
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        # Anchor checkpoint root + creds so the CLI doesn't need real env.
        ckpt_root = tmp_path / "ckpts"
        monkeypatch.setenv("SYNTHPANEL_CHECKPOINT_ROOT", str(ckpt_root))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_personas_yaml())
        instrument_file = tmp_path / "survey.yaml"
        instrument_file.write_text(_instrument_yaml())

        run_id = "run-resume-cli"
        _seed_partial_checkpoint(
            tmp_path=ckpt_root,
            run_id=run_id,
            persona_names=["Alice", "Bob", "Carol"],
            completed_names=["Alice"],  # 1 done, 2 remaining
            personas_path=str(personas_file),
            instrument_path=str(instrument_file),
        )

        # Mock the orchestrator: only Bob+Carol should be dispatched.
        registry = WorkerRegistry()
        dispatched_personas: list[list[str]] = []

        def fake_run_panel_parallel(*args, **kwargs):
            personas = kwargs["personas"]
            dispatched_personas.append([p.get("name") for p in personas])
            return (
                [
                    PanelistResult(
                        persona_name=p["name"],
                        responses=[{"question": "Q1", "response": f"fresh answer {p['name']}"}],
                        usage=CostTokenUsage(input_tokens=5, output_tokens=2),
                    )
                    for p in personas
                ],
                registry,
                {},
            )

        args = self._parsed(
            [
                "--model",
                "sonnet",
                "panel",
                "run",
                "--resume",
                run_id,
                "--no-synthesis",
            ]
        )
        with (
            patch("synth_panel.cli.commands.run_panel_parallel", side_effect=fake_run_panel_parallel),
            patch("synth_panel.cli.commands.LLMClient"),
        ):
            code = handle_panel_run(args, self._fmt())

        assert code == 0, "resume without --personas/--instrument should succeed"
        # Only the 2 remaining panelists were dispatched (Alice was skipped).
        assert dispatched_personas == [["Bob", "Carol"]]

    def test_missing_personas_without_resume_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli.commands import handle_panel_run

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        instrument_file = tmp_path / "survey.yaml"
        instrument_file.write_text(_instrument_yaml())

        args = self._parsed(
            [
                "panel",
                "run",
                "--instrument",
                str(instrument_file),
            ]
        )
        code = handle_panel_run(args, self._fmt())
        assert code == 1
        err = capsys.readouterr().err
        assert "--personas is required" in err

    def test_resume_drift_refuses_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli.commands import handle_panel_run

        ckpt_root = tmp_path / "ckpts"
        monkeypatch.setenv("SYNTHPANEL_CHECKPOINT_ROOT", str(ckpt_root))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_personas_yaml())
        instrument_file = tmp_path / "survey.yaml"
        instrument_file.write_text(_instrument_yaml())

        run_id = "run-drift"
        _seed_partial_checkpoint(
            tmp_path=ckpt_root,
            run_id=run_id,
            persona_names=["Alice", "Bob", "Carol"],
            completed_names=["Alice"],
            personas_path=str(personas_file),
            instrument_path=str(instrument_file),
        )

        # Ask for a different model on resume — original was 'sonnet'.
        args = self._parsed(
            [
                "--model",
                "haiku",
                "panel",
                "run",
                "--resume",
                run_id,
            ]
        )
        code = handle_panel_run(args, self._fmt())
        assert code == 1
        err = capsys.readouterr().err
        assert "drift" in err.lower()
        assert "--allow-drift" in err

    def test_resume_with_allow_drift_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from unittest.mock import patch

        from synth_panel.cli.commands import handle_panel_run
        from synth_panel.cost import TokenUsage as CostTokenUsage
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        ckpt_root = tmp_path / "ckpts"
        monkeypatch.setenv("SYNTHPANEL_CHECKPOINT_ROOT", str(ckpt_root))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text(_personas_yaml())
        instrument_file = tmp_path / "survey.yaml"
        instrument_file.write_text(_instrument_yaml())

        run_id = "run-drift-allowed"
        _seed_partial_checkpoint(
            tmp_path=ckpt_root,
            run_id=run_id,
            persona_names=["Alice", "Bob", "Carol"],
            completed_names=["Alice"],
            personas_path=str(personas_file),
            instrument_path=str(instrument_file),
        )

        registry = WorkerRegistry()

        def fake_run_panel_parallel(*args, **kwargs):
            personas = kwargs["personas"]
            return (
                [
                    PanelistResult(
                        persona_name=p["name"],
                        responses=[{"question": "Q1", "response": "x"}],
                        usage=CostTokenUsage(input_tokens=1, output_tokens=1),
                    )
                    for p in personas
                ],
                registry,
                {},
            )

        args = self._parsed(
            [
                "--model",
                "haiku",  # drift: original was sonnet
                "panel",
                "run",
                "--resume",
                run_id,
                "--allow-drift",
                "--no-synthesis",
            ]
        )
        with (
            patch("synth_panel.cli.commands.run_panel_parallel", side_effect=fake_run_panel_parallel),
            patch("synth_panel.cli.commands.LLMClient"),
        ):
            code = handle_panel_run(args, self._fmt())

        assert code == 0
        err = capsys.readouterr().err
        assert "drift" in err.lower()
        assert "statistically inconsistent" in err.lower()


# ---------------------------------------------------------------------------
# sy-z8p6: schema_version field + migration registry
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    """schema_version field and migrate_to_current() migration chain."""

    def test_current_version_written_on_save(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-ver",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        d = ckpt.to_dict()
        assert d["schema_version"] == CURRENT_SCHEMA_VERSION
        assert "version" not in d

    def test_current_version_persisted_to_disk(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-ver-disk",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        directory = checkpoint_dir_for("run-ver-disk", tmp_path)
        save_checkpoint(ckpt, directory)
        raw = json.loads((directory / "state.json").read_text())
        assert raw["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_v1_checkpoint_migrates_on_load(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice", "Bob"])
        directory = checkpoint_dir_for("run-v1", tmp_path)
        directory.mkdir(parents=True)
        v1_data = {
            "version": 1,
            "run_id": "run-v1",
            "created_at": "2026-04-24T00:00:00Z",
            "updated_at": "2026-04-24T00:00:00Z",
            "config_fingerprint": fingerprint_config(cfg),
            "config": cfg,
            "completed": [{"persona": "Alice", "responses": [], "usage": {}, "error": None}],
            "remaining": ["Bob"],
            "usage": {"input_tokens": 5},
            "abort_reason": None,
            # No cli_args, no schema_version
        }
        (directory / "state.json").write_text(json.dumps(v1_data))
        loaded = load_checkpoint("run-v1", tmp_path)
        assert loaded.run_id == "run-v1"
        assert loaded.cli_args is None
        assert loaded.completed[0]["persona"] == "Alice"
        assert loaded.remaining == ["Bob"]

    def test_schema_too_new_raises(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice"])
        directory = checkpoint_dir_for("run-future", tmp_path)
        directory.mkdir(parents=True)
        future_data = {
            "schema_version": CURRENT_SCHEMA_VERSION + 1,
            "run_id": "run-future",
            "created_at": "now",
            "updated_at": "now",
            "config_fingerprint": fingerprint_config(cfg),
            "config": cfg,
        }
        (directory / "state.json").write_text(json.dumps(future_data))
        with pytest.raises(CheckpointSchemaTooNewError, match="newer than this synthpanel"):
            load_checkpoint("run-future", tmp_path)

    def test_migrate_v1_to_v2_adds_defaults(self) -> None:
        v1 = {
            "version": 1,
            "run_id": "x",
            "created_at": "now",
            "updated_at": "now",
            "config_fingerprint": "abc",
            "config": {},
        }
        result = migrate_v1_to_v2(v1)
        assert result["schema_version"] == 2
        assert result["cli_args"] is None
        assert result["completed"] == []
        assert result["remaining"] == []
        assert result["usage"] == {}
        assert result["abort_reason"] is None

    def test_migrate_v1_to_v2_preserves_existing_fields(self) -> None:
        v1 = {
            "version": 1,
            "run_id": "x",
            "created_at": "now",
            "updated_at": "now",
            "config_fingerprint": "abc",
            "config": {},
            "completed": [{"persona": "Alice"}],
            "remaining": ["Bob"],
            "usage": {"input_tokens": 7},
            "abort_reason": "signal:SIGINT",
        }
        result = migrate_v1_to_v2(v1)
        assert result["completed"] == [{"persona": "Alice"}]
        assert result["remaining"] == ["Bob"]
        assert result["usage"] == {"input_tokens": 7}
        assert result["abort_reason"] == "signal:SIGINT"

    def test_migrate_to_current_noop_at_current_version(self) -> None:
        data = {"schema_version": CURRENT_SCHEMA_VERSION, "run_id": "x"}
        assert migrate_to_current(data) is data

    def test_migrate_to_current_raises_on_future_version(self) -> None:
        data = {"schema_version": CURRENT_SCHEMA_VERSION + 1}
        with pytest.raises(ValueError, match="newer than this synthpanel"):
            migrate_to_current(data)

    def test_round_trip_preserves_schema_version(self, tmp_path: Path) -> None:
        cfg = _make_config(["Alice"])
        ckpt = PanelCheckpoint(
            run_id="run-rt-ver",
            created_at="now",
            updated_at="now",
            config_fingerprint=fingerprint_config(cfg),
            config=cfg,
        )
        directory = checkpoint_dir_for("run-rt-ver", tmp_path)
        save_checkpoint(ckpt, directory)
        loaded = load_checkpoint("run-rt-ver", tmp_path)
        # Re-saving the loaded checkpoint still writes the current version.
        d = loaded.to_dict()
        assert d["schema_version"] == CURRENT_SCHEMA_VERSION
