"""Tests for the synth-panel CLI framework."""

from __future__ import annotations

import io
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.main import main
from synth_panel.cli.parser import build_parser
from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.slash import dispatch_slash, SLASH_COMMANDS
from synth_panel.cli.repl import SessionState
from synth_panel.cli.commands import (
    _load_personas,
    _load_instrument,
    _load_schema,
    handle_pack_list,
    handle_pack_import,
    handle_pack_export,
)
from synth_panel.prompts import persona_system_prompt
from synth_panel.llm.models import (
    CompletionResponse,
    TextBlock,
    TokenUsage as LLMTokenUsage,
    StopReason,
)
from synth_panel.runtime import TurnSummary
from synth_panel.cost import TokenUsage, ZERO_USAGE


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
        args = parser.parse_args([
            "--model", "opus",
            "--permission-mode", "read-only",
            "--output-format", "json",
            "--config", "/tmp/cfg.toml",
        ])
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
        args = parser.parse_args([
            "panel", "run",
            "--personas", "p.yaml",
            "--instrument", "i.yaml",
        ])
        assert args.command == "panel"
        assert args.panel_command == "run"
        assert args.personas == "p.yaml"
        assert args.instrument == "i.yaml"

    def test_panel_run_synthesis_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "panel", "run",
            "--personas", "p.yaml",
            "--instrument", "i.yaml",
            "--no-synthesis",
            "--synthesis-model", "opus",
            "--synthesis-prompt", "Summarize briefly.",
        ])
        assert args.no_synthesis is True
        assert args.synthesis_model == "opus"
        assert args.synthesis_prompt == "Summarize briefly."

    def test_panel_run_synthesis_defaults(self):
        parser = build_parser()
        args = parser.parse_args([
            "panel", "run",
            "--personas", "p.yaml",
            "--instrument", "i.yaml",
        ])
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
        p.write_text(
            "personas:\n"
            "  - name: Alice\n"
            "    age: 30\n"
            "    occupation: Engineer\n"
        )
        personas = _load_personas(str(p))
        assert len(personas) == 1
        assert personas[0]["name"] == "Alice"

    def test_load_personas_as_list(self, tmp_path):
        p = tmp_path / "personas.yaml"
        p.write_text(
            "- name: Bob\n"
            "  age: 25\n"
        )
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

    def test_load_instrument_with_key(self, tmp_path):
        p = tmp_path / "survey.yaml"
        p.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: What?\n"
        )
        instr = _load_instrument(str(p))
        assert instr.questions == [{"text": "What?"}]
        assert len(instr.rounds) == 1

    def test_load_instrument_questions_key(self, tmp_path):
        p = tmp_path / "survey.yaml"
        p.write_text(
            "questions:\n"
            "  - text: What?\n"
        )
        instr = _load_instrument(str(p))
        assert instr.questions == [{"text": "What?"}]
        assert len(instr.rounds) == 1

    def test_load_instrument_invalid(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("just a string")
        with pytest.raises(ValueError, match="Invalid instrument file"):
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
        personas_file.write_text(
            "personas:\n"
            "  - name: Alice\n"
            "    age: 30\n"
        )
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: What do you think?\n"
        )

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
        ])
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
        personas_file.write_text(
            "personas:\n"
            "  - name: Bob\n"
        )
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: Question?\n"
        )

        code = main([
            "--output-format", "json",
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["persona_count"] == 1
        assert data["question_count"] == 1
        # New rounds-shaped output (sp-zg4): single-round wraps per-persona
        # results inside a single round entry. Legacy flat shape requires
        # --legacy-output and will be removed in 0.6.0.
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

        code = main([
            "--output-format", "json",
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
            "--no-synthesis",
        ])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["synthesis"] is None
        assert data["total_cost"] == data["panelist_cost"]

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

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
            "--synthesis-model", "opus",
        ])
        assert code == 0
        mock_synth.assert_called_once()
        _, kwargs = mock_synth.call_args
        assert kwargs["model"] == "opus"

    def test_panel_run_missing_personas(self, capsys, tmp_path):
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main([
            "panel", "run",
            "--personas", "/nonexistent.yaml",
            "--instrument", str(survey_file),
        ])
        assert code == 1

    def test_panel_run_missing_instrument(self, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: X\n")

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", "/nonexistent.yaml",
        ])
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
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: Q1\n"
            "    - text: Q2\n"
        )

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
        ])
        assert code == 2
        err = capsys.readouterr().err
        assert "PANEL RUN INVALID" in err
        assert "4/4" in err
        # Synthesis MUST not run on a fully-errored panel
        mock_synth.assert_not_called()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_strict_flag_bails_on_any_error(
        self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path
    ):
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
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: Q1\n"
            "    - text: Q2\n"
        )

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
            "--strict",
        ])
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
        survey_file.write_text(
            "instrument:\n"
            "  questions:\n"
            "    - text: Q1\n"
            "    - text: Q2\n"
        )

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
        ])
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

        code = main([
            "--output-format", "json",
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
        ])
        assert code == 2
        data = json.loads(capsys.readouterr().out)
        assert data["run_invalid"] is True
        assert data["failure_stats"]["errored_pairs"] == 1
        assert data["failure_stats"]["total_pairs"] == 1
        assert data["failure_stats"]["failure_rate"] == 1.0
        mock_synth.assert_not_called()


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
        pfile.write_text(
            "name: My Pack\n"
            "personas:\n"
            "  - name: Alice\n"
            "    age: 30\n"
        )
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
        args = parser.parse_args([
            "panel", "run",
            "--personas", "p.yaml",
            "--instrument", "i.yaml",
            "--schema", '{"type": "object"}',
        ])
        assert args.schema == '{"type": "object"}'

    def test_schema_flag_default_none(self):
        parser = build_parser()
        args = parser.parse_args([
            "panel", "run",
            "--personas", "p.yaml",
            "--instrument", "i.yaml",
        ])
        assert args.schema is None


class TestPanelRunWithSchema:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_invalid_schema(self, mock_client_cls, mock_runtime_cls, capsys, tmp_path):
        personas_file = tmp_path / "personas.yaml"
        personas_file.write_text("personas:\n  - name: X\n")
        survey_file = tmp_path / "survey.yaml"
        survey_file.write_text("instrument:\n  questions:\n    - text: Q?\n")

        code = main([
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
            "--schema", "not valid json {{{",
        ])
        assert code == 1

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_passes_schema_to_orchestrator(self, mock_client_cls, mock_run, mock_synth, capsys, tmp_path):
        """Verify --schema is loaded and passed to run_panel_parallel."""
        from synth_panel.cost import ZERO_USAGE
        from synth_panel.orchestrator import PanelistResult

        mock_run.return_value = (
            [PanelistResult(persona_name="X", responses=[{"question": "Q", "response": {"a": 1}, "structured": True}], usage=ZERO_USAGE)],
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

        code = main([
            "--output-format", "json",
            "panel", "run",
            "--personas", str(personas_file),
            "--instrument", str(survey_file),
            "--schema", str(schema_file),
        ])
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
        src = self._write(tmp_path, """
instrument:
  questions:
    - text: What frustrates you?
""")
        args = MagicMock(source=src, format="text")
        rc = handle_instruments_graph(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "instrument v" in out
        assert "[default]" in out or "default" in out

    def test_v2_mermaid_linear_chain(self, tmp_path, capsys):
        from synth_panel.cli.commands import handle_instruments_graph
        src = self._write(tmp_path, """
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
""")
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
        src = self._write(tmp_path, """
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
""")
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
        src = self._write(tmp_path, """
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
""")
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
