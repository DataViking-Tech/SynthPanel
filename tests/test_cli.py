"""Tests for the synthpanel CLI framework."""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cli.commands import (
    _build_rounds_shape,
    _load_instrument,
    _load_personas,
    _load_schema,
    _merge_persona_lists,
)
from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.parser import build_parser
from synth_panel.cli.repl import SessionState
from synth_panel.cli.slash import SLASH_COMMANDS, dispatch_slash
from synth_panel.cost import ZERO_USAGE, TokenUsage
from synth_panel.main import main
from synth_panel.prompts import persona_system_prompt
from synth_panel.runtime import TurnSummary

# --- Parser tests ---


class TestParser:
    def test_no_args_gives_no_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_global_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.model is None
        assert args.permission_mode == "full-access"
        assert args.config is None
        assert args.output_format == "text"

    def test_global_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--model",
                "opus",
                "--permission-mode",
                "read-only",
                "--output-format",
                "json",
                "--config",
                "/tmp/cfg.toml",
            ]
        )
        assert args.model == "opus"
        assert args.permission_mode == "read-only"
        assert args.output_format == "json"
        assert args.config == "/tmp/cfg.toml"

    def test_prompt_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["prompt", "hello", "world"])
        assert args.command == "prompt"
        assert args.text == ["hello", "world"]

    def test_panel_run_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.command == "panel"
        assert args.panel_command == "run"
        assert args.personas == "p.yaml"
        assert args.instrument == "i.yaml"

    def test_panel_run_personas_merge_default_empty(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.personas_merge == []

    def test_panel_run_personas_merge_repeatable(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--personas-merge",
                "extra1.yaml",
                "--personas-merge",
                "extra2.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.personas_merge == ["extra1.yaml", "extra2.yaml"]

    def test_panel_run_synthesis_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
                "--no-synthesis",
                "--synthesis-model",
                "opus",
                "--synthesis-prompt",
                "Summarize briefly.",
            ]
        )
        assert args.no_synthesis is True
        assert args.synthesis_model == "opus"
        assert args.synthesis_prompt == "Summarize briefly."

    def test_panel_run_synthesis_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.no_synthesis is False
        assert args.synthesis_model is None
        assert args.synthesis_prompt is None

    def test_invalid_permission_mode(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--permission-mode", "invalid"])

    def test_invalid_output_format(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--output-format", "xml"])


# --- Output tests ---


class TestOutput:
    def test_text_format(self):
        buf = io.StringIO()
        emit(OutputFormat.TEXT, message="hello", file=buf)
        assert buf.getvalue() == "hello\n"

    def test_text_format_with_usage(self):
        buf = io.StringIO()
        emit(
            OutputFormat.TEXT,
            message="hello",
            usage={"input_tokens": 10, "output_tokens": 20},
            file=buf,
        )
        output = buf.getvalue()
        assert "hello" in output
        assert "input=10" in output
        assert "output=20" in output

    def test_json_format(self):
        buf = io.StringIO()
        emit(OutputFormat.JSON, message="hi", file=buf)
        data = json.loads(buf.getvalue())
        assert data["message"] == "hi"

    def test_json_format_with_usage(self):
        buf = io.StringIO()
        emit(
            OutputFormat.JSON,
            message="hi",
            usage={"input_tokens": 5, "output_tokens": 10},
            file=buf,
        )
        data = json.loads(buf.getvalue())
        assert data["usage"]["input_tokens"] == 5

    def test_ndjson_format(self):
        buf = io.StringIO()
        emit(OutputFormat.NDJSON, message="hi", file=buf)
        data = json.loads(buf.getvalue())
        assert data["type"] == "message"
        assert data["text"] == "hi"


# --- Slash command tests ---


class TestSlashCommands:
    def _make_state(self) -> SessionState:
        return SessionState(model="sonnet")

    def test_help_lists_all_commands(self, capsys):
        state = self._make_state()
        dispatch_slash("/help", state, OutputFormat.TEXT)
        out = capsys.readouterr().out
        for name in SLASH_COMMANDS:
            assert f"/{name}" in out

    def test_status_shows_state(self, capsys):
        state = self._make_state()
        state.turn_count = 3
        dispatch_slash("/status", state, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "Turn count: 3" in out
        assert "sonnet" in out

    def test_model_show(self, capsys):
        state = self._make_state()
        dispatch_slash("/model", state, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "sonnet" in out

    def test_model_switch(self, capsys):
        state = self._make_state()
        dispatch_slash("/model opus", state, OutputFormat.TEXT)
        assert state.model == "opus"

    def test_clear_requires_confirm(self, capsys):
        state = self._make_state()
        state.turn_count = 5
        dispatch_slash("/clear", state, OutputFormat.TEXT)
        assert state.turn_count == 5  # not cleared
        out = capsys.readouterr().out
        assert "--confirm" in out

    def test_clear_with_confirm(self, capsys):
        state = self._make_state()
        state.turn_count = 5
        dispatch_slash("/clear --confirm", state, OutputFormat.TEXT)
        assert state.turn_count == 0

    def test_unknown_command(self, capsys):
        state = self._make_state()
        dispatch_slash("/foobar", state, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert "Unknown command" in out
        assert "/foobar" in out

    def test_json_output(self, capsys):
        state = self._make_state()
        dispatch_slash("/status", state, OutputFormat.JSON)
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "Turn count" in data["message"]


# --- Mock helpers ---


def _mock_turn_summary(text: str = "Hello!") -> TurnSummary:
    """Create a TurnSummary with a text response."""
    from synth_panel.persistence import ConversationMessage

    usage = TokenUsage(input_tokens=10, output_tokens=20)
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


# --- main() integration tests (with mocked LLM) ---


class TestMain:
    @patch("synth_panel.cli.commands.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_prompt_command(self, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("Hi there!")
        mock_runtime_cls.return_value = mock_runtime

        code = main(["prompt", "hello", "world"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Hi there!" in out

    @patch("synth_panel.cli.commands.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_prompt_json_output(self, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("response text")
        mock_runtime_cls.return_value = mock_runtime

        code = main(["--output-format", "json", "prompt", "test"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["message"] == "response text"
        assert "cost" in data
        assert "usage" in data

    @patch("synth_panel.cli.commands.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_prompt_error_returns_nonzero(self, mock_client_cls, mock_runtime_cls, capsys):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.side_effect = RuntimeError("API failure")
        mock_runtime_cls.return_value = mock_runtime

        code = main(["prompt", "hello"])
        assert code == 1


# --- YAML loading tests ---


class TestYAMLLoading:
    def test_load_personas_with_key(self, tmp_path):
        p = tmp_path / "personas.yaml"
        p.write_text("personas:\n  - name: Alice\n    age: 30\n    occupation: Engineer\n")
        personas = _load_personas(str(p))
        assert len(personas) == 1
        assert personas[0]["name"] == "Alice"

    def test_load_personas_as_list(self, tmp_path):
        p = tmp_path / "personas.yaml"
        p.write_text("- name: Bob\n  age: 25\n")
        personas = _load_personas(str(p))
        assert len(personas) == 1
        assert personas[0]["name"] == "Bob"

    def test_load_personas_invalid(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="Invalid personas file"):
            _load_personas(str(p))

    def test_load_personas_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _load_personas("/nonexistent/path.yaml")

    def test_merge_persona_lists_appends(self, tmp_path):
        extra = tmp_path / "extra.yaml"
        extra.write_text("personas:\n  - name: Carol\n    age: 40\n")
        base = [{"name": "Alice", "age": 30}]
        merged = _merge_persona_lists(base, [str(extra)])
        assert [p["name"] for p in merged] == ["Alice", "Carol"]

    def test_merge_persona_lists_multiple_files_in_order(self, tmp_path):
        f1 = tmp_path / "f1.yaml"
        f1.write_text("personas:\n  - name: Bob\n")
        f2 = tmp_path / "f2.yaml"
        f2.write_text("personas:\n  - name: Dan\n")
        base = [{"name": "Alice"}]
        merged = _merge_persona_lists(base, [str(f1), str(f2)])
        assert [p["name"] for p in merged] == ["Alice", "Bob", "Dan"]

    def test_merge_persona_lists_name_collision_overrides(self, tmp_path):
        override = tmp_path / "override.yaml"
        override.write_text("personas:\n  - name: Alice\n    age: 99\n")
        base = [{"name": "Alice", "age": 30}, {"name": "Bob"}]
        merged = _merge_persona_lists(base, [str(override)])
        assert len(merged) == 2
        assert merged[0] == {"name": "Alice", "age": 99}
        assert merged[1] == {"name": "Bob"}

    def test_merge_persona_lists_unnamed_always_appended(self, tmp_path):
        extra = tmp_path / "extra.yaml"
        extra.write_text("- occupation: Anonymous\n- occupation: Anonymous\n")
        base: list = []
        merged = _merge_persona_lists(base, [str(extra)])
        assert len(merged) == 2

    def test_merge_persona_lists_missing_file(self):
        with pytest.raises(FileNotFoundError):
            _merge_persona_lists([], ["/nonexistent/extra.yaml"])

    def test_load_instrument_with_key(self, tmp_path):
        p = tmp_path / "survey.yaml"
        p.write_text("instrument:\n  questions:\n    - text: What?\n")
        instr = _load_instrument(str(p))
        assert instr.questions == [{"text": "What?"}]
        assert len(instr.rounds) == 1

    def test_load_instrument_questions_key(self, tmp_path):
        p = tmp_path / "survey.yaml"
        p.write_text("questions:\n  - text: What?\n")
        instr = _load_instrument(str(p))
        assert instr.questions == [{"text": "What?"}]
        assert len(instr.rounds) == 1

    def test_load_instrument_invalid(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="Invalid instrument"):
            _load_instrument(str(p))

    def test_persona_system_prompt(self):
        persona = {
            "name": "Alice",
            "age": 30,
            "occupation": "Engineer",
            "background": "10 years in software",
            "personality_traits": ["curious", "methodical"],
        }
        prompt = persona_system_prompt(persona)
        assert "Alice" in prompt
        assert "30" in prompt
        assert "Engineer" in prompt
        assert "curious" in prompt
        assert "methodical" in prompt


# --- Panel run tests ---


def _mock_synthesis_result():
    """Create a mock SynthesisResult for CLI integration tests."""
    from synth_panel.cost import CostEstimate
    from synth_panel.synthesis import SynthesisResult

    return SynthesisResult(
        summary="Test synthesis summary",
        themes=["theme1"],
        agreements=["agree1"],
        disagreements=[],
        surprises=[],
        recommendation="Do X",
        usage=ZERO_USAGE,
        cost=CostEstimate(),
        model="sonnet",
    )


class TestPanelRun:
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_basic(self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("I think...")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: What do you think?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Alice" in out
        assert "I think..." in out
        assert "SYNTHESIS" in out
        assert "Test synthesis summary" in out

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_json(self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Question?\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["persona_count"] == 1
        assert data["question_count"] == 1
        # Rounds-shaped output (sp-zg4): single-round wraps per-persona
        # results inside a single round entry.
        assert "rounds" in data
        assert len(data["rounds"]) == 1
        assert len(data["rounds"][0]["results"]) == 1
        assert data["synthesis"] is not None
        assert data["synthesis"]["summary"] == "Test synthesis summary"
        assert "panelist_cost" in data

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_no_synthesis(self, mock_client_cls, mock_runtime_cls, capsys, tmp_path):
        """--no-synthesis skips the synthesis step entirely."""
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main(
            [
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
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["synthesis"] is None
        assert data["total_cost"] == data["panelist_cost"]

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_multi_question_emits_full_cost_shape(
        self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path
    ):
        """Multi-question CLI runs must surface the full cost shape (sp-027).

        The synthbench publish pipeline reads ``total_cost``, ``total_usage``,
        ``panelist_cost`` and ``panelist_usage`` off the rounds-shaped JSON
        output. A regression in any one of them silently zeroes a
        leaderboard row's $/100Q column, so this test pins the contract.
        """
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1?\n    - text: Q2?\n    - text: Q3?\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)

        # All four cost fields must be present at the top level of the
        # rounds-shaped output — no silent drops, no Nones.
        for key in ("total_cost", "total_usage", "panelist_cost", "panelist_usage"):
            assert key in data, f"missing {key} in multi-question output"
            assert data[key] is not None, f"{key} unexpectedly None"

        # panelist_usage is the dict shape produced by TokenUsage.to_dict()
        # — it must include the four token buckets even when zero.
        for bucket in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            assert bucket in data["panelist_usage"]
            assert bucket in data["total_usage"]

        # Cost strings are formatted USD (e.g. "$0.0001"). Reject the
        # accidental empty / placeholder shapes that would mask a regression.
        assert data["total_cost"].startswith("$")
        assert data["panelist_cost"].startswith("$")

        # The metadata block surfaces the same data for downstream consumers
        # (synthbench publish reads either path); both must agree.
        assert "metadata" in data
        assert "cost" in data["metadata"]
        assert "total_cost_usd" in data["metadata"]["cost"]

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_synthesis_model(self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path):
        """--synthesis-model is passed through to synthesize_panel."""
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--synthesis-model",
                "opus",
            ]
        )
        assert code == 0
        mock_synth.assert_called_once()
        _, kwargs = mock_synth.call_args
        assert kwargs["model"] == "opus"

    def test_panel_run_missing_personas(self, capsys, tmp_path):
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                "/nonexistent.yaml",
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 1

    def test_panel_run_missing_instrument(self, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: X\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                "/nonexistent.yaml",
            ]
        )
        assert code == 1

    # --- sp-2hg: silent-failure-on-universal-provider-errors ------------

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_all_errored_returns_nonzero_and_skips_synthesis(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-2hg: when every question errors the run must exit non-zero,
        surface a fatal banner, and NOT call synthesize_panel."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Q1", "response": "[error: boom]", "error": True},
                        {"question": "Q2", "response": "[error: boom]", "error": True},
                    ],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[
                        {"question": "Q1", "response": "[error: boom]", "error": True},
                        {"question": "Q2", "response": "[error: boom]", "error": True},
                    ],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n    - text: Q2\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 2
        err = capsys.readouterr().err
        assert "PANEL RUN INVALID" in err
        assert "4/4" in err
        # Synthesis MUST not run on a fully-errored panel
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_strict_flag_bails_on_any_error(self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path):
        """sp-2hg: --strict turns any error into a fatal run (exit 3)."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Q1", "response": "good answer"},
                        {"question": "Q2", "response": "good answer"},
                    ],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[
                        {"question": "Q1", "response": "[error: boom]", "error": True},
                        {"question": "Q2", "response": "good answer"},
                    ],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n    - text: Q2\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--strict",
            ]
        )
        assert code == 3
        err = capsys.readouterr().err
        assert "--strict" in err
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_partial_errors_below_threshold_still_synthesizes(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-2hg: a single-error panel (25% failure) should still synthesize
        when --strict is NOT set and the failure rate is below the default
        threshold (0.5). Exit code is 0."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Q1", "response": "good"},
                        {"question": "Q2", "response": "good"},
                    ],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[
                        {"question": "Q1", "response": "[error: x]", "error": True},
                        {"question": "Q2", "response": "good"},
                    ],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n    - text: Q2\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        mock_synth.assert_called_once()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_json_output_includes_failure_stats(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-2hg: structured output must carry failure_stats + run_invalid
        so MCP/CI consumers can gate without parsing banners."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Q1", "response": "[error: x]", "error": True},
                    ],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["run_invalid"] is True
        assert data["failure_stats"]["errored_pairs"] == 1
        assert data["failure_stats"]["total_pairs"] == 1
        assert data["failure_stats"]["failure_rate"] == 1.0
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_total_failure_names_model_and_upstream_error(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-efip: a knowingly-bad model name that 400s on every call must
        exit non-zero and name the failing model + upstream status on stderr.

        Regression guard: previously the CLI returned a normally-shaped
        result with 0-token panelists and a misleading "panel complete"
        exit code when every request was rejected upstream.
        """
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        upstream_err = "OpenRouter API error 400: invalid model 'haiku:0.25' is not a recognized identifier"
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Q1", "response": f"[error: {upstream_err}]", "error": True},
                    ],
                    usage=ZERO_USAGE,
                    model="haiku:0.25",
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[
                        {"question": "Q1", "response": f"[error: {upstream_err}]", "error": True},
                    ],
                    usage=ZERO_USAGE,
                    model="haiku:0.25",
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--model",
                "haiku:0.25",
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        # Banner on stderr must name the failing model and the 400 status.
        assert "PANEL RUN INVALID" in captured.err
        assert "haiku:0.25" in captured.err
        assert "400" in captured.err
        # JSON payload must carry the structured diagnostic for MCP/CI.
        data = json.loads(captured.out)
        assert data["run_invalid"] is True
        assert data["total_failure"]["panelists"] == 2
        assert data["total_failure"]["models"] == ["haiku:0.25"]
        assert data["total_failure"]["sample_errors"], "sample_errors must be populated"
        first_persona, first_err = data["total_failure"]["sample_errors"][0]
        assert first_persona in {"Alice", "Bob"}
        assert "400" in first_err
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_total_failure_when_every_panelist_exploded(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-efip: wholesale panelist exceptions (pr.error set, responses
        empty) must also trigger the total-failure banner even if the
        aggregate failure rate somehow rounds to the threshold."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[],
                    usage=ZERO_USAGE,
                    error="OpenRouter API error 400: bad model 'haiku:0.25'",
                    model="haiku:0.25",
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q1\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "PANEL RUN INVALID" in captured.err
        assert "haiku:0.25" in captured.err
        assert "400" in captured.err
        mock_synth.assert_not_called()


# --- sp-efip: total-failure detector unit tests ---------------------------


class TestDetectTotalFailure:
    """Unit coverage for the shared helper that the CLI + MCP both use."""

    def test_detects_all_error_responses(self):
        from synth_panel._runners import detect_total_failure
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[{"question": "Q", "response": "[error: 400]", "error": True}],
                usage=ZERO_USAGE,
                model="haiku:0.25",
            ),
        ]
        failure = detect_total_failure(results)
        assert failure is not None
        assert failure["panelists"] == 1
        assert failure["models"] == ["haiku:0.25"]
        assert failure["sample_errors"][0][0] == "Alice"
        assert "400" in failure["sample_errors"][0][1]

    def test_detects_wholesale_panelist_error(self):
        from synth_panel._runners import detect_total_failure
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[],
                usage=ZERO_USAGE,
                error="provider 400",
                model="bad-model",
            ),
        ]
        failure = detect_total_failure(results)
        assert failure is not None
        assert failure["models"] == ["bad-model"]
        assert failure["sample_errors"][0] == ("Alice", "provider 400")

    def test_returns_none_when_any_panelist_has_usable_data(self):
        from synth_panel._runners import detect_total_failure
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[{"question": "Q", "response": "[error: boom]", "error": True}],
                usage=ZERO_USAGE,
                model="m1",
            ),
            PanelistResult(
                persona_name="Bob",
                responses=[{"question": "Q", "response": "a real answer"}],
                usage=TokenUsage(input_tokens=10, output_tokens=5),
                model="m1",
            ),
        ]
        assert detect_total_failure(results) is None

    def test_empty_results_is_total_failure(self):
        from synth_panel._runners import detect_total_failure

        failure = detect_total_failure([])
        assert failure is not None
        assert failure["panelists"] == 0


# --- sp-efip: run_panel_sync raises on total failure ----------------------


class TestRunPanelSyncTotalFailure:
    """The shared runner used by MCP + SDK must raise, not silently return
    a 'success' envelope, when every panelist failed."""

    @patch("synth_panel._runners.run_panel_parallel")
    def test_raises_when_every_panelist_errored(self, mock_run):
        from synth_panel._runners import PanelTotalFailureError, run_panel_sync
        from synth_panel.orchestrator import PanelistResult

        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q", "response": "[error: 400]", "error": True}],
                    usage=ZERO_USAGE,
                    model="haiku:0.25",
                ),
            ],
            None,
            {},
        )

        client = MagicMock()
        with pytest.raises(PanelTotalFailureError) as excinfo:
            run_panel_sync(
                client=client,
                personas=[{"name": "Alice"}],
                questions=[{"text": "Q"}],
                model="haiku:0.25",
                synthesis=False,
            )
        assert "haiku:0.25" in str(excinfo.value)
        assert "400" in str(excinfo.value)
        diag = excinfo.value.diagnostic
        assert diag["models"] == ["haiku:0.25"]


# --- sp-bjt4: missing-input refusal detection -----------------------------


class TestMissingInputDetection:
    """Polite refusals ("I don't see the content you mentioned") don't trip
    _analyze_failures because the panelist returned a clean response. This
    suite exercises the secondary heuristic that gates run_invalid on
    ≥50% of panelists reporting missing/unavailable input."""

    def test_detector_flags_above_threshold(self):
        from synth_panel.cli.commands import _detect_missing_input_refusals
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[
                    {"question": "Summarize the page", "response": "I don't see the page content you mentioned."},
                ],
                usage=ZERO_USAGE,
            ),
            PanelistResult(
                persona_name="Bob",
                responses=[
                    {"question": "Summarize the page", "response": "No content was provided for me to analyze."},
                ],
                usage=ZERO_USAGE,
            ),
            PanelistResult(
                persona_name="Carol",
                responses=[
                    {"question": "Summarize the page", "response": "The copy here reads clearly and I think it works."},
                ],
                usage=ZERO_USAGE,
            ),
        ]

        stats = _detect_missing_input_refusals(results)
        assert stats["considered"] == 3
        assert stats["refusing"] == 2
        assert stats["refusal_rate"] == pytest.approx(2 / 3)
        assert stats["refusing_personas"] == ["Alice", "Bob"]

    def test_detector_ignores_follow_ups(self):
        """Primary-response refusals count; follow-ups piggyback and are skipped
        so we don't double-count a single refusal."""
        from synth_panel.cli.commands import _detect_missing_input_refusals
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[
                    {"question": "Primary Q", "response": "Sounds great, I'd pay $10/mo."},
                    {
                        "question": "Follow-up",
                        "response": "I haven't been provided with more details.",
                        "follow_up": True,
                    },
                ],
                usage=ZERO_USAGE,
            ),
        ]
        stats = _detect_missing_input_refusals(results)
        assert stats["considered"] == 1
        assert stats["refusing"] == 0
        assert stats["refusal_rate"] == 0.0

    def test_detector_excludes_errored_panelists(self):
        """A wholesale panelist failure is already flagged by _analyze_failures;
        including it here would double-count and distort the rate."""
        from synth_panel.cli.commands import _detect_missing_input_refusals
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[],
                usage=ZERO_USAGE,
                error="provider 503",
            ),
            PanelistResult(
                persona_name="Bob",
                responses=[
                    {"question": "Q", "response": "I don't see the content you mentioned."},
                ],
                usage=ZERO_USAGE,
            ),
        ]
        stats = _detect_missing_input_refusals(results)
        assert stats["considered"] == 1
        assert stats["refusing"] == 1
        assert stats["refusal_rate"] == 1.0

    def test_detector_returns_zero_rate_when_empty(self):
        from synth_panel.cli.commands import _detect_missing_input_refusals

        stats = _detect_missing_input_refusals([])
        assert stats["considered"] == 0
        assert stats["refusing"] == 0
        assert stats["refusal_rate"] == 0.0
        assert stats["refusing_personas"] == []

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_sets_run_invalid_on_majority_missing_input(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """End-to-end: three clean panelists, two refuse for missing input
        → run_invalid=True, warning surfaced, synthesis skipped."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[
                        {"question": "Review the page", "response": "I don't see the page content you mentioned."},
                    ],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[
                        {"question": "Review the page", "response": "No content was provided to analyze."},
                    ],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Carol",
                    responses=[
                        {"question": "Review the page", "response": "It's a solid value prop overall."},
                    ],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n  - name: Carol\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Review the page\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["run_invalid"] is True
        assert data["missing_input_stats"]["refusing"] == 2
        assert data["missing_input_stats"]["considered"] == 3
        assert data["missing_input_stats"]["refusal_rate"] == pytest.approx(2 / 3)
        assert sorted(data["missing_input_stats"]["refusing_personas"]) == ["Alice", "Bob"]
        # Top-level warning is what CI gates key on when they don't parse stats.
        assert any("missing_input_refusals" in w for w in data["warnings"])
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_stays_valid_below_threshold(self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path):
        """One refusal out of three (~33%) is below the 50% threshold — the
        run is still valid and synthesis proceeds."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q", "response": "I don't see the content you mentioned."}],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Bob",
                    responses=[{"question": "Q", "response": "Reasonably priced."}],
                    usage=ZERO_USAGE,
                ),
                PanelistResult(
                    persona_name="Carol",
                    responses=[{"question": "Q", "response": "Great value."}],
                    usage=ZERO_USAGE,
                ),
            ],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n  - name: Carol\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["run_invalid"] is False
        assert data["missing_input_stats"]["refusing"] == 1
        assert data["missing_input_stats"]["considered"] == 3
        assert not any("missing_input_refusals" in w for w in data.get("warnings", []))
        mock_synth.assert_called_once()


# --- sp-7vp: --save flag --------------------------------------------------


class TestPanelRunSave:
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_save_flag_writes_result_and_prints_id(
        self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path, monkeypatch
    ):
        """--save stores results to disk and prints the result ID to stderr."""
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        # Redirect results to tmp_path so we don't pollute ~/.synthpanel
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: What do you think?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--save",
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        assert "Result saved: result-" in captured.err

        # Extract the result ID from stderr
        for line in captured.err.splitlines():
            if line.startswith("Result saved: "):
                result_id = line.split("Result saved: ")[1].strip()
                break
        else:
            pytest.fail("Result ID not found in stderr")

        # Verify the file was created
        result_file = tmp_path / "data" / "results" / f"{result_id}.json"
        assert result_file.exists()

        # Verify the file content is valid JSON with expected fields
        data = json.loads(result_file.read_text())
        assert data["persona_count"] == 1
        assert data["question_count"] == 1
        assert "results" in data
        assert "total_usage" in data
        assert "total_cost" in data

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_save_result_works_with_analyze(
        self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, tmp_path, monkeypatch
    ):
        """Saved result can be loaded by 'synthpanel analyze'."""
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("Strongly agree")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 25\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Do you agree?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--save",
            ]
        )
        assert code == 0

        # Extract result ID
        captured = capsys.readouterr()
        result_id = None
        for line in captured.err.splitlines():
            if line.startswith("Result saved: "):
                result_id = line.split("Result saved: ")[1].strip()
                break
        assert result_id is not None

        # Now run analyze on the saved result
        code = main(["analyze", result_id])
        assert code == 0

    def test_save_flag_parser(self):
        """--save flag is accepted by the parser."""
        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml", "--save"])
        assert args.save is True

    def test_save_flag_default_false(self):
        """--save defaults to False."""
        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert args.save is False


# --- sp-5on.5: panel synthesize re-synthesis subcommand -----------------


class TestPanelSynthesize:
    """sp-5on.5: ``panel synthesize`` replays synthesis against a saved result."""

    def test_panel_synthesize_parser(self):
        parser = build_parser()
        args = parser.parse_args(
            ["panel", "synthesize", "result-abc", "--synthesis-model", "opus", "--synthesis-prompt", "Be brief."]
        )
        assert args.command == "panel"
        assert args.panel_command == "synthesize"
        assert args.result == "result-abc"
        assert args.synthesis_model == "opus"
        assert args.synthesis_prompt == "Be brief."

    def test_panel_synthesize_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["panel", "synthesize", "result-xyz"])
        assert args.synthesis_model is None
        assert args.synthesis_prompt is None

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_synthesize_reloads_result_and_saves_sidecar(
        self, mock_client_cls, mock_synth, capsys, tmp_path, monkeypatch
    ):
        """Loads saved result by ID, invokes synthesize_panel, writes sidecar."""
        mock_synth.return_value = _mock_synthesis_result()
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))

        # Seed a saved panel result on disk
        results_dir = tmp_path / "data" / "results"
        results_dir.mkdir(parents=True)
        result_id = "result-20260101-120000-abcdef"
        (results_dir / f"{result_id}.json").write_text(
            json.dumps(
                {
                    "created_at": "2026-01-01T12:00:00+00:00",
                    "model": "haiku",
                    "persona_count": 1,
                    "question_count": 1,
                    "total_usage": {"input_tokens": 5, "output_tokens": 10},
                    "total_cost": "$0.0001",
                    "results": [
                        {
                            "persona": "Alice",
                            "responses": [
                                {"question": "What do you think?", "response": "I like it."},
                            ],
                            "usage": {"input_tokens": 5, "output_tokens": 10},
                            "cost": "$0.0001",
                            "error": None,
                        }
                    ],
                }
            )
        )

        code = main(["panel", "synthesize", result_id, "--synthesis-model", "opus"])
        assert code == 0

        # synthesize_panel should have been called with the rebuilt panelist_results
        mock_synth.assert_called_once()
        _, kwargs = mock_synth.call_args
        assert kwargs["model"] == "opus"
        assert kwargs["panelist_model"] == "haiku"
        panelist_results_arg = mock_synth.call_args.args[1]
        assert len(panelist_results_arg) == 1
        assert panelist_results_arg[0].persona_name == "Alice"
        # Questions reconstructed from response dicts when not saved explicitly
        questions_arg = mock_synth.call_args.args[2]
        assert questions_arg == [{"text": "What do you think?"}]

        # Sidecar was written alongside the original result
        sidecars = list(results_dir.glob(f"{result_id}.synthesis-*.json"))
        assert len(sidecars) == 1
        sidecar_data = json.loads(sidecars[0].read_text())
        assert sidecar_data["source_result_id"] == result_id
        assert sidecar_data["synthesis"]["summary"] == "Test synthesis summary"
        assert sidecar_data["synthesis_prompt_override"] is False

        # Original result file must remain untouched
        original = json.loads((results_dir / f"{result_id}.json").read_text())
        assert "synthesis" not in original

        # Text output surfaces the synthesis
        out = capsys.readouterr()
        assert "SYNTHESIS" in out.out
        assert "Test synthesis summary" in out.out
        assert "Saved synthesis:" in out.err

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_synthesize_passes_custom_prompt(self, mock_client_cls, mock_synth, capsys, tmp_path, monkeypatch):
        """``--synthesis-prompt`` flows through as custom_prompt."""
        mock_synth.return_value = _mock_synthesis_result()
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))

        result_file = tmp_path / "inline-result.json"
        result_file.write_text(
            json.dumps(
                {
                    "model": "sonnet",
                    "persona_count": 1,
                    "question_count": 1,
                    "results": [
                        {
                            "persona": "Bob",
                            "responses": [{"question": "Q?", "response": "A."}],
                            "usage": {"input_tokens": 1, "output_tokens": 2},
                        }
                    ],
                }
            )
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "synthesize",
                str(result_file),
                "--synthesis-prompt",
                "Summarize tersely.",
            ]
        )
        assert code == 0
        _, kwargs = mock_synth.call_args
        assert kwargs["custom_prompt"] == "Summarize tersely."

        payload = json.loads(capsys.readouterr().out)
        assert payload["source_result_id"] == result_file.stem
        assert payload["synthesis"]["summary"] == "Test synthesis summary"
        assert payload["saved_as"].startswith(f"{result_file.stem}.synthesis-")

    def test_panel_synthesize_missing_result_returns_error(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path / "data"))
        code = main(["panel", "synthesize", "result-does-not-exist"])
        assert code == 1
        assert "not found" in capsys.readouterr().err


# --- sp-f4t: default model resolution -----------------------------------


class TestDefaultModelResolution:
    """sp-f4t: ``--model`` default must respect available credentials and
    surface the chosen model so the user can cancel/override."""

    def test_resolve_default_model_prefers_anthropic(self, monkeypatch):
        from synth_panel.cli.commands import _resolve_default_model

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        alias, source = _resolve_default_model()
        assert alias == "sonnet"
        assert source == "ANTHROPIC_API_KEY"

    def test_resolve_default_model_falls_through_to_gemini(self, monkeypatch):
        from synth_panel.cli.commands import _resolve_default_model

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemkey")
        alias, source = _resolve_default_model()
        assert alias == "gemini-2.5-flash"
        assert source == "GEMINI_API_KEY"

    def test_resolve_default_model_falls_back_to_sonnet_when_no_creds(self, monkeypatch):
        from synth_panel.cli.commands import _resolve_default_model

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        alias, source = _resolve_default_model()
        assert alias == "sonnet"
        assert source is None

    def test_resolve_default_model_respects_preference_order(self, monkeypatch):
        """Anthropic beats Gemini even if both keys are set."""
        from synth_panel.cli.commands import _resolve_default_model

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gemkey")
        alias, source = _resolve_default_model()
        assert alias == "sonnet"
        assert source == "ANTHROPIC_API_KEY"

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_announces_default_model_on_stderr(
        self, mock_client_cls, mock_run, mock_synth, monkeypatch, capsys, tmp_path
    ):
        """When --model is omitted, the chosen default is printed to stderr
        so an operator can cancel and re-run with an explicit override."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "XAI_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "gemkey")

        registry = WorkerRegistry()
        mock_run.return_value = (
            [PanelistResult(persona_name="A", responses=[{"question": "Q", "response": "ok"}], usage=ZERO_USAGE)],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        err = capsys.readouterr().err
        assert "gemini-2.5-flash" in err
        assert "GEMINI_API_KEY" in err
        assert "Override with --model" in err

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_explicit_model_suppresses_announcement(
        self, mock_client_cls, mock_run, mock_synth, monkeypatch, capsys, tmp_path
    ):
        """When --model is explicit, no auto-select message is printed."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        monkeypatch.setenv("GEMINI_API_KEY", "gemkey")

        registry = WorkerRegistry()
        mock_run.return_value = (
            [PanelistResult(persona_name="A", responses=[{"question": "Q", "response": "ok"}], usage=ZERO_USAGE)],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q\n")

        code = main(
            [
                "--model",
                "gemini-2.5-flash-lite",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        err = capsys.readouterr().err
        assert "--model not specified" not in err


# --- sp-1hb: CLI variable passing ---------------------------------------


class TestVarSubstitution:
    """sp-1hb: ``--var KEY=VALUE`` and ``--vars-file`` substitute template
    placeholders in instrument question text."""

    def test_collect_template_vars_parses_cli_flags(self):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        ns = _ap.Namespace(vars=["candidates=Alpha, Beta", "theme=pricing"], vars_file=None)
        result = _collect_template_vars(ns)
        assert result == {"candidates": "Alpha, Beta", "theme": "pricing"}

    def test_collect_template_vars_allows_equals_in_value(self):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        ns = _ap.Namespace(vars=["equation=a=b+c"], vars_file=None)
        assert _collect_template_vars(ns) == {"equation": "a=b+c"}

    def test_collect_template_vars_rejects_malformed_flag(self):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        ns = _ap.Namespace(vars=["novalue"], vars_file=None)
        with pytest.raises(ValueError, match="KEY=VALUE"):
            _collect_template_vars(ns)

    def test_collect_template_vars_rejects_empty_key(self):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        ns = _ap.Namespace(vars=["=hello"], vars_file=None)
        with pytest.raises(ValueError, match="empty key"):
            _collect_template_vars(ns)

    def test_collect_template_vars_merges_file_and_cli_flags(self, tmp_path):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        f = tmp_path / "vars.yaml"
        f.write_text("candidates:\n  - Alpha\n  - Beta\ntheme: memorability\n")
        ns = _ap.Namespace(vars=["theme=pricing"], vars_file=str(f))
        result = _collect_template_vars(ns)
        # List flattened to comma-joined string
        assert result["candidates"] == "Alpha, Beta"
        # CLI overrides file
        assert result["theme"] == "pricing"

    def test_collect_template_vars_rejects_non_mapping_file(self, tmp_path):
        import argparse as _ap

        from synth_panel.cli.commands import _collect_template_vars

        f = tmp_path / "bad.yaml"
        f.write_text("- just\n- a\n- list\n")
        ns = _ap.Namespace(vars=None, vars_file=str(f))
        with pytest.raises(ValueError, match="YAML mapping"):
            _collect_template_vars(ns)

    def test_apply_vars_to_instrument_mutates_round_questions(self):
        from synth_panel.cli.commands import _apply_vars_to_instrument
        from synth_panel.instrument import parse_instrument

        inst = parse_instrument(
            {
                "version": 3,
                "rounds": [
                    {
                        "name": "intro",
                        "questions": [
                            {"text": "Evaluate: {candidates}. Reaction?"},
                        ],
                    },
                    {
                        "name": "followup",
                        "depends_on": "intro",
                        "questions": [
                            {"text": "Which of {candidates} do you remember?"},
                        ],
                    },
                ],
            }
        )
        _apply_vars_to_instrument(inst, {"candidates": "Alpha, Beta, Gamma"})
        assert inst.rounds[0].questions[0]["text"] == ("Evaluate: Alpha, Beta, Gamma. Reaction?")
        assert inst.rounds[1].questions[0]["text"] == ("Which of Alpha, Beta, Gamma do you remember?")

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_cli_var_substitutes_bundled_placeholder(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """End-to-end: a question containing ``{candidates}`` is passed to
        ``run_panel_parallel`` already rendered when ``--var candidates=...``
        is supplied."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="A", responses=[{"question": "rendered", "response": "ok"}], usage=ZERO_USAGE
                )
            ],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: 'Evaluate: {candidates}'\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--var",
                "candidates=Alpha, Beta, Gamma",
            ]
        )
        assert code == 0
        # Verify the orchestrator saw the rendered question text
        _, kwargs = mock_run.call_args
        questions_passed = kwargs["questions"]
        assert len(questions_passed) == 1
        assert questions_passed[0]["text"] == "Evaluate: Alpha, Beta, Gamma"

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_malformed_var_returns_error(self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--var",
                "badentry",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "KEY=VALUE" in err
        mock_run.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_aborts_on_unsubstituted_placeholder(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-6yi: missing --var aborts the run before any LLM call."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: 'Rate {features} for {problem}'\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "unsubstituted placeholders" in err
        assert "features" in err
        assert "problem" in err
        assert "--var features=" in err
        mock_run.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_allow_unresolved_warns_and_proceeds(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-6yi: --allow-unresolved keeps literal braces and emits a warning."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        registry = WorkerRegistry()
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="A",
                    responses=[{"question": "Rate {features}", "response": "ok"}],
                    usage=ZERO_USAGE,
                )
            ],
            registry,
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: 'Rate {features}'\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--allow-unresolved",
            ]
        )
        assert code == 0
        err = capsys.readouterr().err
        assert "Warning" in err
        assert "features" in err
        _, kwargs = mock_run.call_args
        assert kwargs["questions"][0]["text"] == "Rate {features}"

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_partial_vars_still_aborts_on_remaining_placeholder(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
        """sp-6yi: supplying some --var values still fails when others are missing."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text(
            "instrument:\n  questions:\n"
            "    - text: 'Rate {features}'\n"
            "      follow_ups:\n"
            "        - 'And for {problem}?'\n"
        )

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--var",
                "features=X, Y",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "problem" in err
        assert "features" not in err.split("placeholders:", 1)[-1].split(".")[0]
        mock_run.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_landing_page_pack_aborts_on_missing_landing_page(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path, monkeypatch
    ):
        """sp-anje: the exact Round 3 self-audit repro must fail fast.

        Running the bundled ``landing-page-comprehension`` pack with only
        an irrelevant ``--var product=anything`` previously let every
        persona see a literal ``{landing_page}`` block. The guard must
        abort with ``landing_page`` (and ``alt_cta``) named in stderr
        and no LLM call issued.
        """
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: A\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                "landing-page-comprehension",
                "--var",
                "product=anything",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "unsubstituted placeholders" in err
        assert "landing_page" in err
        assert "alt_cta" in err
        mock_run.assert_not_called()


# --- Pack CLI tests ---


class TestPackCommands:
    @pytest.fixture(autouse=True)
    def _data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
        self._tmp = tmp_path

    def test_pack_list_shows_bundled(self, capsys):
        code = main(["pack", "list"])
        assert code == 0
        out = capsys.readouterr().out
        assert "developer" in out
        assert "Developers" in out

    def test_pack_import_and_list(self, capsys):
        pfile = self._tmp / "personas.yaml"
        pfile.write_text("name: My Pack\npersonas:\n  - name: Alice\n    age: 30\n")
        code = main(["pack", "import", str(pfile)])
        assert code == 0
        out = capsys.readouterr().out
        assert "My Pack" in out
        assert "1 personas" in out

    def test_pack_import_with_custom_name_and_id(self, capsys):
        pfile = self._tmp / "p.yaml"
        pfile.write_text("personas:\n  - name: Bob\n")
        code = main(["pack", "import", str(pfile), "--name", "Custom", "--id", "my-id"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Custom" in out
        assert "my-id" in out

    def test_pack_import_invalid_file(self, capsys):
        code = main(["pack", "import", "/nonexistent.yaml"])
        assert code == 1

    def test_pack_import_validation_error(self, capsys):
        pfile = self._tmp / "bad.yaml"
        pfile.write_text("personas:\n  - age: 30\n")
        code = main(["pack", "import", str(pfile)])
        assert code == 1
        assert "Validation error" in capsys.readouterr().err

    def test_pack_export_stdout(self, capsys):
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("Export Test", [{"name": "Eve"}], pack_id="exp")
        code = main(["pack", "export", "exp"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Eve" in out
        assert "Export Test" in out

    def test_pack_export_to_file(self, capsys):
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("File Export", [{"name": "Dan"}], pack_id="fexp")
        outfile = self._tmp / "out.yaml"
        code = main(["pack", "export", "fexp", "-o", str(outfile)])
        assert code == 0
        content = outfile.read_text()
        assert "Dan" in content

    def test_pack_export_nonexistent(self, capsys):
        code = main(["pack", "export", "nope"])
        assert code == 1

    def test_pack_show_prints_yaml_to_stdout(self, capsys):
        """sp-oem: `pack show <id>` is an alias for stdout export."""
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("Show Test", [{"name": "Zoe"}], pack_id="shtest")
        code = main(["pack", "show", "shtest"])
        assert code == 0
        out = capsys.readouterr().out
        assert "Zoe" in out
        assert "Show Test" in out

    def test_pack_show_matches_pack_export_stdout(self, capsys):
        """sp-oem: `pack show <id>` must produce the same output as
        `pack export <id>` with no output file - the whole point of the
        alias is that they are interchangeable for inspection use."""
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("Parity Test", [{"name": "Iris", "age": 40}], pack_id="parity")
        main(["pack", "export", "parity"])
        export_out = capsys.readouterr().out
        main(["pack", "show", "parity"])
        show_out = capsys.readouterr().out
        assert export_out == show_out

    def test_pack_show_nonexistent(self, capsys):
        code = main(["pack", "show", "nope"])
        assert code == 1

    def test_pack_list_json(self, capsys):
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("JSON Pack", [{"name": "X"}], pack_id="jp")
        code = main(["--output-format", "json", "pack", "list"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        user_packs = [p for p in data["packs"] if not p.get("builtin")]
        assert len(user_packs) == 1
        assert user_packs[0]["id"] == "jp"

    def test_pack_import_name_from_stem(self, capsys):
        pfile = self._tmp / "my-team.yaml"
        pfile.write_text("- name: Alice\n")
        code = main(["pack", "import", str(pfile)])
        assert code == 0
        out = capsys.readouterr().out
        assert "my-team" in out

    def test_pack_generate_success(self, capsys, monkeypatch):
        """pack generate calls LLM and saves a valid persona pack."""
        from unittest.mock import MagicMock

        from synth_panel.llm.models import CompletionResponse, TokenUsage, ToolInvocationBlock

        mock_personas = [
            {
                "name": "Alice Chen",
                "age": 34,
                "occupation": "Product Manager",
                "background": "Ten years in SaaS product management.",
                "personality_traits": ["analytical", "pragmatic"],
            },
            {
                "name": "Bob Rivera",
                "age": 28,
                "occupation": "Software Engineer",
                "background": "Full-stack developer at a startup.",
                "personality_traits": ["curious", "detail-oriented"],
            },
        ]

        mock_response = CompletionResponse(
            id="r1",
            model="test",
            content=[
                ToolInvocationBlock(
                    id="tc1",
                    name="generate_personas",
                    input={"personas": mock_personas},
                ),
            ],
            usage=TokenUsage(input_tokens=100, output_tokens=200),
        )

        mock_client = MagicMock()
        mock_client.send.return_value = mock_response
        monkeypatch.setattr("synth_panel.cli.commands.LLMClient", lambda: mock_client)

        code = main(
            [
                "pack",
                "generate",
                "--product",
                "project management tool",
                "--audience",
                "engineering teams",
                "--count",
                "2",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "2 personas" in out
        assert "project management tool personas" in out

    def test_pack_generate_custom_name_and_id(self, capsys, monkeypatch):
        """pack generate respects --name and --id flags."""
        from unittest.mock import MagicMock

        from synth_panel.llm.models import CompletionResponse, TokenUsage, ToolInvocationBlock

        mock_response = CompletionResponse(
            id="r1",
            model="test",
            content=[
                ToolInvocationBlock(
                    id="tc1",
                    name="generate_personas",
                    input={
                        "personas": [
                            {
                                "name": "Eve",
                                "age": 40,
                                "occupation": "CTO",
                                "background": "Led teams.",
                                "personality_traits": ["bold"],
                            }
                        ]
                    },
                ),
            ],
            usage=TokenUsage(input_tokens=10, output_tokens=20),
        )

        mock_client = MagicMock()
        mock_client.send.return_value = mock_response
        monkeypatch.setattr("synth_panel.cli.commands.LLMClient", lambda: mock_client)

        code = main(
            [
                "pack",
                "generate",
                "--product",
                "CRM",
                "--audience",
                "sales teams",
                "--count",
                "1",
                "--name",
                "Sales Personas",
                "--id",
                "sales-gen",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Sales Personas" in out
        assert "sales-gen" in out

    def test_pack_generate_llm_fallback_error(self, capsys, monkeypatch):
        """pack generate returns 1 when LLM fails to produce structured output."""
        from unittest.mock import MagicMock

        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="r1",
            model="test",
            content=[TextBlock(text="Sorry, I cannot do that.")],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )

        mock_client = MagicMock()
        mock_client.send.return_value = mock_response
        monkeypatch.setattr("synth_panel.cli.commands.LLMClient", lambda: mock_client)

        code = main(
            [
                "pack",
                "generate",
                "--product",
                "widget",
                "--audience",
                "everyone",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "structured output" in err.lower() or "tool call" in err.lower()

    def test_pack_generate_invalid_count(self, capsys):
        """pack generate rejects count outside 1-50."""
        code = main(
            [
                "pack",
                "generate",
                "--product",
                "x",
                "--audience",
                "y",
                "--count",
                "0",
            ]
        )
        assert code == 1
        assert "count" in capsys.readouterr().err.lower()

    def test_pack_generate_json_output(self, capsys, monkeypatch):
        """pack generate emits JSON when --output-format json is used."""
        from unittest.mock import MagicMock

        from synth_panel.llm.models import CompletionResponse, TokenUsage, ToolInvocationBlock

        mock_response = CompletionResponse(
            id="r1",
            model="test",
            content=[
                ToolInvocationBlock(
                    id="tc1",
                    name="generate_personas",
                    input={
                        "personas": [
                            {
                                "name": "Zara",
                                "age": 25,
                                "occupation": "Designer",
                                "background": "UX designer.",
                                "personality_traits": ["creative"],
                            }
                        ]
                    },
                ),
            ],
            usage=TokenUsage(input_tokens=10, output_tokens=20),
        )

        mock_client = MagicMock()
        mock_client.send.return_value = mock_response
        monkeypatch.setattr("synth_panel.cli.commands.LLMClient", lambda: mock_client)

        code = main(
            [
                "--output-format",
                "json",
                "pack",
                "generate",
                "--product",
                "design tool",
                "--audience",
                "designers",
                "--count",
                "1",
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["persona_count"] == 1

    def test_parser_pack_subcommands(self):
        parser = build_parser()
        args = parser.parse_args(["pack", "list"])
        assert args.command == "pack"
        assert args.pack_command == "list"

        args = parser.parse_args(["pack", "import", "file.yaml"])
        assert args.pack_command == "import"
        assert args.file == "file.yaml"

        args = parser.parse_args(["pack", "export", "my-pack"])
        assert args.pack_command == "export"
        assert args.pack_id == "my-pack"

        args = parser.parse_args(
            [
                "pack",
                "generate",
                "--product",
                "test product",
                "--audience",
                "test audience",
                "--count",
                "3",
            ]
        )
        assert args.pack_command == "generate"
        assert args.product == "test product"
        assert args.audience == "test audience"
        assert args.count == 3


# --- Schema loading tests ---


class TestSchemaLoading:
    def test_load_schema_from_file(self, tmp_path):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        p = tmp_path / "schema.json"
        p.write_text(json.dumps(schema))
        loaded = _load_schema(str(p))
        assert loaded == schema

    def test_load_schema_from_inline_json(self):
        schema_str = '{"type": "object", "properties": {"x": {"type": "number"}}}'
        loaded = _load_schema(schema_str)
        assert loaded["type"] == "object"
        assert "x" in loaded["properties"]

    def test_load_schema_invalid_json(self):
        with pytest.raises(ValueError, match="not a valid file path or JSON"):
            _load_schema("not json at all {{{")

    def test_load_schema_non_object(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _load_schema("[1, 2, 3]")


class TestParserSchemaFlag:
    def test_schema_flag_accepted(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
                "--schema",
                '{"type": "object"}',
            ]
        )
        assert args.schema == '{"type": "object"}'

    def test_schema_flag_default_none(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.schema is None


class TestPanelRunWithSchema:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_invalid_schema(self, mock_client_cls, mock_runtime_cls, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: X\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--schema",
                "not valid json {{{",
            ]
        )
        assert code == 1

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_passes_schema_to_orchestrator(self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path):
        """Verify --schema is loaded and passed to run_panel_parallel."""
        from synth_panel.cost import ZERO_USAGE
        from synth_panel.orchestrator import PanelistResult

        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="X",
                    responses=[{"question": "Q", "response": {"a": 1}, "structured": True}],
                    usage=ZERO_USAGE,
                )
            ],
            MagicMock(),
            {},
        )
        mock_synth.return_value = _mock_synthesis_result()

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: X\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")
        schema_file = tmp_path / "schema.json"
        schema_file.write_text('{"type": "object", "properties": {"a": {"type": "integer"}}}')

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--schema",
                str(schema_file),
            ]
        )
        assert code == 0
        # Verify run_panel_parallel was called with response_schema
        call_kwargs = mock_run.call_args
        assert call_kwargs[1]["response_schema"] == {"type": "object", "properties": {"a": {"type": "integer"}}}


# --- __main__.py test ---


class TestMainModule:
    def test_main_module_importable(self):
        """Verify __main__.py exists and references main()."""
        from pathlib import Path

        main_path = Path(__file__).parent.parent / "src" / "synth_panel" / "__main__.py"
        assert main_path.exists()
        content = main_path.read_text()
        assert "from synth_panel.main import main" in content


# ---------------------------------------------------------------------------
# instruments graph (sp-irf F3-D)
# ---------------------------------------------------------------------------


class TestInstrumentsGraph:
    """Render the round DAG of v1, v2, and v3 instruments."""

    def _write(self, tmp_path, body: str):
        p = tmp_path / "inst.yaml"
        p.write_text(body)
        return str(p)

    def test_v1_text(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph

        src = self._write(
            tmp_path,
            """
instrument:
  questions:
    - text: What frustrates you?
""",
        )
        args = MagicMock(source=src, format="text")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "instrument v" in out
        assert "[default]" in out or "default" in out

    def test_v2_mermaid_linear_chain(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph

        src = self._write(
            tmp_path,
            """
instrument:
  version: 2
  rounds:
    - name: explore
      questions:
        - text: What do you use today?
    - name: probe
      depends_on: explore
      questions:
        - text: Why?
""",
        )
        args = MagicMock(source=src, format="mermaid")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "flowchart TD" in out
        assert "explore([explore])" in out
        assert "probe([probe])" in out
        assert "explore --> probe" in out

    def test_v3_branching_mermaid_with_end(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph

        src = self._write(
            tmp_path,
            """
instrument:
  version: 3
  rounds:
    - name: explore
      questions:
        - text: What concerns you?
      route_when:
        - if:
            field: themes
            op: contains
            value: pricing
          goto: probe_pricing
        - else: __end__
    - name: probe_pricing
      questions:
        - text: How much?
""",
        )
        args = MagicMock(source=src, format="mermaid")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "flowchart TD" in out
        assert "explore -->|themes contains pricing| probe_pricing" in out
        assert "explore -->|else| __end__" in out
        # Terminal sentinel must be declared as a node when referenced
        assert "__end__(((end)))" in out

    def test_v3_branching_text_format(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph

        src = self._write(
            tmp_path,
            """
instrument:
  version: 3
  rounds:
    - name: a
      questions:
        - text: Q1?
      route_when:
        - if: {field: themes, op: equals, value: x}
          goto: b
        - else: __end__
    - name: b
      questions:
        - text: Q2?
""",
        )
        args = MagicMock(source=src, format="text")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "[a]" in out and "[b]" in out
        assert "if themes equals" in out
        assert "-> b" in out
        assert "else -> __end__" in out

    def test_missing_source(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph

        args = MagicMock(source=str(tmp_path / "nope.yaml"), format="text")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        err = capsys.readouterr().err
        assert rc == 1
        assert "Error" in err


# --- Variants flag tests (sp-5on.15) -----------------------------------------


class TestVariantsFlag:
    def test_parser_accepts_variants(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
                "--variants",
                "3",
            ]
        )
        assert args.variants == 3

    def test_parser_variants_default_is_none(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.variants is None

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    @patch("synth_panel.cli.commands.generate_panel_variants")
    def test_variants_expands_personas(
        self,
        mock_gen_variants,
        mock_client_cls,
        mock_runtime_cls,
        mock_synth,
        capsys,
        tmp_path,
    ):
        from synth_panel.perturbation import PersonaVariant, PerturbationAxis, PerturbationRecord, VariantSet

        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("variant answer")
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()

        # Build 2 variants for 1 persona
        base = {"name": "Alice", "age": 30}
        record = PerturbationRecord(
            axis=PerturbationAxis.TRAIT_SWAP,
            original_field="personality_traits",
            original_value="analytical",
            perturbed_value="intuitive",
            change_description="swapped trait",
        )
        v0 = PersonaVariant(
            persona=dict(base, name="Alice (v0)"),
            source_persona_name="Alice",
            variant_index=0,
            perturbation=record,
        )
        v1 = PersonaVariant(
            persona=dict(base, name="Alice (v1)"),
            source_persona_name="Alice",
            variant_index=1,
            perturbation=record,
        )
        mock_gen_variants.return_value = [VariantSet(source_persona=base, variants=[v0, v1])]

        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: What?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--variants",
                "2",
            ]
        )
        assert code == 0
        mock_gen_variants.assert_called_once()
        call_args = mock_gen_variants.call_args
        assert call_args.kwargs["k"] == 2

        err = capsys.readouterr().err
        assert "Generating 2 variants for 1 personas" in err
        assert "Variant expansion complete" in err

    @patch("synth_panel.cli.commands.LLMClient")
    def test_variants_out_of_range_returns_error(self, mock_client_cls, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: What?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--variants",
                "25",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "--variants must be between 1 and 20" in err


# --- sp-x8g: --dry-run preview -------------------------------------------


class TestPanelRunDryRun:
    """--dry-run previews the substituted panel without any LLM call."""

    def test_parser_accepts_dry_run_flag(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
                "--dry-run",
            ]
        )
        assert args.dry_run is True

    def test_parser_dry_run_default_false(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
            ]
        )
        assert args.dry_run is False

    @patch("synth_panel.cli.commands.generate_panel_variants")
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_dry_run_makes_no_llm_calls(
        self,
        mock_client_cls,
        mock_runtime_cls,
        mock_run_panel,
        mock_synth,
        mock_variants,
        capsys,
        tmp_path,
    ):
        """--dry-run must not trigger the orchestrator, variants, synthesis, or runtime."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Question 1?\n    - text: Question 2?\n")

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--dry-run",
            ]
        )
        assert code == 0
        mock_run_panel.assert_not_called()
        mock_synth.assert_not_called()
        mock_variants.assert_not_called()
        mock_runtime_cls.assert_not_called()

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_dry_run_text_output_shows_substituted_questions(self, mock_client_cls, mock_run_panel, capsys, tmp_path):
        """Question text appears after {var} substitution."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n  - name: Carol\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: How do you feel about {product}?\n"
            "    - text: What would {product} need to cost?\n"
        )

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--dry-run",
                "--var",
                "product=Acme Widget",
            ]
        )
        assert code == 0
        err = capsys.readouterr().err
        assert "DRY RUN" in err
        assert "Personas: 3" in err
        assert "Questions: 2" in err
        assert "Acme Widget" in err
        assert "{product}" not in err
        assert "Estimated input tokens" in err
        mock_run_panel.assert_not_called()

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_dry_run_multi_round_lists_rounds(self, mock_client_cls, mock_run_panel, capsys, tmp_path):
        """Multi-round instruments show each round's questions."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text(
            "instrument:\n"
            "  version: 2\n"
            "  rounds:\n"
            "    - name: discovery\n"
            "      questions:\n"
            "        - text: What frustrates you?\n"
            "    - name: deep_dive\n"
            "      depends_on: discovery\n"
            "      questions:\n"
            "        - text: Tell me more.\n"
        )

        code = main(
            [
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--dry-run",
            ]
        )
        assert code == 0
        err = capsys.readouterr().err
        assert "Round: discovery" in err
        assert "Round: deep_dive" in err
        assert "What frustrates you?" in err
        assert "Tell me more." in err
        assert "across 2 rounds" in err
        mock_run_panel.assert_not_called()

    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_dry_run_json_output(self, mock_client_cls, mock_run_panel, capsys, tmp_path):
        """JSON output returns a structured dry-run payload on stdout."""
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: Alice\n  - name: Bob\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Why {x}?\n")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
                "--dry-run",
                "--var",
                "x=now",
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["dry_run"] is True
        assert data["persona_count"] == 2
        assert data["question_count"] == 1
        assert data["estimated_input_tokens"] >= 1
        assert data["rounds"][0]["questions"] == ["Why now?"]
        mock_run_panel.assert_not_called()


# --- sp-nn8k: _build_rounds_shape propagates cost_fallback_warnings ------


class _StubInstrument:
    def __init__(self):
        self.rounds = [type("R", (), {"name": "default"})()]
        self.warnings = []


class TestBuildRoundsShapeCostFallback:
    """Cost fallback warnings must merge into ``warnings`` and flip the flag."""

    def _zero_cost(self):
        from synth_panel.cost import CostEstimate

        return CostEstimate()

    def _call(self, **overrides):
        defaults = dict(
            instrument=_StubInstrument(),
            results=[],
            synthesis_dict=None,
            panelist_cost=self._zero_cost(),
            total_usage=TokenUsage(),
            total_cost=self._zero_cost(),
            model="sonnet",
            persona_count=0,
            question_count=0,
        )
        defaults.update(overrides)
        return _build_rounds_shape(**defaults)

    def test_no_warnings_when_absent(self):
        out = self._call()
        assert out["warnings"] == []
        assert out["cost_is_estimated"] is False

    def test_cost_fallback_warnings_merge(self):
        warnings_in = ["Cost for model 'novel-x' computed using DEFAULT_PRICING fallback — real charges may differ."]
        out = self._call(cost_fallback_warnings=warnings_in)
        assert warnings_in[0] in out["warnings"]
        assert out["cost_is_estimated"] is True

    def test_instrument_warnings_preserved_alongside_cost_warnings(self):
        inst = _StubInstrument()
        inst.warnings = ["unreachable round: dead_end"]
        cost_warnings = [
            "Cost for model 'mystery-v9' computed using DEFAULT_PRICING fallback — real charges may differ."
        ]
        out = self._call(instrument=inst, cost_fallback_warnings=cost_warnings)
        assert "unreachable round: dead_end" in out["warnings"]
        assert cost_warnings[0] in out["warnings"]
        assert out["cost_is_estimated"] is True

    def test_empty_list_equals_no_warnings(self):
        out = self._call(cost_fallback_warnings=[])
        assert out["warnings"] == []
        assert out["cost_is_estimated"] is False
