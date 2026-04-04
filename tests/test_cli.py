"""Tests for the synth-panel CLI framework."""

from __future__ import annotations

import io
import json

import pytest

from synth_panel.main import main
from synth_panel.cli.parser import build_parser
from synth_panel.cli.output import OutputFormat, emit
from synth_panel.cli.slash import dispatch_slash, SLASH_COMMANDS
from synth_panel.cli.repl import SessionState


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

    def test_login_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["login"])
        assert args.command == "login"

    def test_logout_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["logout"])
        assert args.command == "logout"

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


# --- main() integration tests ---


class TestMain:
    def test_prompt_command(self, capsys):
        code = main(["prompt", "hello", "world"])
        assert code == 0
        out = capsys.readouterr().out
        assert "hello world" in out

    def test_prompt_json_output(self, capsys):
        code = main(["--output-format", "json", "prompt", "test"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert "test" in data["message"]

    def test_login_command(self, capsys):
        code = main(["login"])
        assert code == 0

    def test_logout_command(self, capsys):
        code = main(["logout"])
        assert code == 0
