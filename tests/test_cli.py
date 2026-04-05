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
        assert "questions" in instr

    def test_load_instrument_questions_key(self, tmp_path):
        p = tmp_path / "survey.yaml"
        p.write_text(
            "questions:\n"
            "  - text: What?\n"
        )
        instr = _load_instrument(str(p))
        assert "questions" in instr

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


class TestPanelRun:
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_basic(self, mock_client_cls, mock_runtime_cls, capsys, tmp_path):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("I think...")
        mock_runtime_cls.return_value = mock_runtime

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

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_panel_run_json(self, mock_client_cls, mock_runtime_cls, capsys, tmp_path):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary("answer")
        mock_runtime_cls.return_value = mock_runtime

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
        assert len(data["results"]) == 1

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


# --- __main__.py test ---


class TestMainModule:
    def test_main_module_importable(self):
        """Verify __main__.py exists and references main()."""
        from pathlib import Path
        main_path = Path(__file__).parent.parent / "src" / "synth_panel" / "__main__.py"
        assert main_path.exists()
        content = main_path.read_text()
        assert "from synth_panel.main import main" in content
