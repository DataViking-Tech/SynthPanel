"""Tests for the login / logout / whoami CLI (sp-lve)."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from synth_panel.cli.parser import build_parser
from synth_panel.credentials import load_credentials, save_credential
from synth_panel.main import main


class TestParser:
    def test_login_defaults_to_anthropic(self):
        parser = build_parser()
        args = parser.parse_args(["login"])
        assert args.command == "login"
        assert args.provider == "anthropic"
        assert args.api_key is None

    def test_login_accepts_provider_and_api_key(self):
        parser = build_parser()
        args = parser.parse_args(["login", "--provider", "openai", "--api-key", "sk-123"])
        assert args.provider == "openai"
        assert args.api_key == "sk-123"

    def test_logout_has_all_option(self):
        parser = build_parser()
        args = parser.parse_args(["logout", "--provider", "all"])
        assert args.command == "logout"
        assert args.provider == "all"

    def test_whoami_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["whoami"])
        assert args.command == "whoami"


class TestLogin:
    def test_login_stores_key_from_flag(self, capsys):
        rc = main(["login", "--provider", "anthropic", "--api-key", "sk-cli"])
        assert rc == 0
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-cli"}
        assert "Stored" in capsys.readouterr().out

    def test_login_reads_key_from_piped_stdin(self, capsys):
        with patch("sys.stdin", new=io.StringIO("sk-piped\n")):
            rc = main(["login", "--provider", "openai"])
        assert rc == 0
        assert load_credentials() == {"OPENAI_API_KEY": "sk-piped"}

    def test_login_rejects_empty_key(self, capsys):
        with patch("sys.stdin", new=io.StringIO("\n")):
            rc = main(["login", "--provider", "anthropic"])
        assert rc == 2
        assert load_credentials() == {}

    def test_login_emits_json_when_requested(self, capsys):
        rc = main(["--output-format", "json", "login", "--provider", "xai", "--api-key", "xk"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["message"] == "credential_stored"
        assert payload["env_var"] == "XAI_API_KEY"


class TestLogout:
    def test_logout_removes_single_provider(self, capsys):
        save_credential("ANTHROPIC_API_KEY", "sk-a")
        save_credential("OPENAI_API_KEY", "sk-b")
        rc = main(["logout", "--provider", "anthropic"])
        assert rc == 0
        assert load_credentials() == {"OPENAI_API_KEY": "sk-b"}

    def test_logout_all_clears_store(self, capsys):
        save_credential("ANTHROPIC_API_KEY", "sk-a")
        save_credential("OPENAI_API_KEY", "sk-b")
        rc = main(["logout", "--provider", "all"])
        assert rc == 0
        assert load_credentials() == {}

    def test_logout_missing_provider_still_exits_zero(self, capsys):
        rc = main(["logout", "--provider", "anthropic"])
        assert rc == 0
        assert "No stored credential" in capsys.readouterr().out


class TestWhoami:
    def test_whoami_reports_no_credentials(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        rc = main(["whoami"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No provider credentials found" in out

    def test_whoami_reports_stored_credential(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        save_credential("ANTHROPIC_API_KEY", "sk-disk")
        rc = main(["whoami"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ANTHROPIC_API_KEY" in out
        assert "stored" in out

    def test_whoami_reports_env_source_when_set(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        rc = main(["whoami"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "env" in out

    def test_whoami_json_output(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        save_credential("XAI_API_KEY", "sk-x")
        rc = main(["--output-format", "json", "whoami"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["message"] == "credentials_status"
        xai = next(p for p in payload["providers"] if p["env_var"] == "XAI_API_KEY")
        assert xai["available"] is True
        assert xai["source"] == "stored"
