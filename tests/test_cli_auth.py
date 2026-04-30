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
        rc = main(["login", "--provider", "anthropic", "--api-key", "sk-ant-cli"])
        assert rc == 0
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-ant-cli"}
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
        rc = main(["--output-format", "json", "login", "--provider", "xai", "--api-key", "xai-k"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["message"] == "credential_stored"
        assert payload["env_var"] == "XAI_API_KEY"


class TestLoginPrefixValidation:
    """Reject visibly-broken keys at login time (sy-bybx)."""

    def test_strips_surrounding_whitespace(self, capsys):
        rc = main(["login", "--provider", "anthropic", "--api-key", "  sk-ant-foo  "])
        assert rc == 0
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-ant-foo"}

    def test_rejects_anthropic_key_without_sk_ant_prefix(self, capsys):
        rc = main(["login", "--provider", "anthropic", "--api-key", "sk-proj-mismatch"])
        assert rc == 2
        assert load_credentials() == {}
        err = capsys.readouterr().err
        assert "sk-ant-" in err
        assert "Anthropic" in err
        # Bug example: the hint should suggest --provider openai
        assert "--provider openai" in err

    def test_rejects_openrouter_key_for_openai_provider(self, capsys):
        rc = main(["login", "--provider", "openai", "--api-key", "sk-or-mistake"])
        assert rc == 2
        assert load_credentials() == {}
        err = capsys.readouterr().err
        assert "--provider openrouter" in err

    def test_rejects_anthropic_key_for_openai_provider(self, capsys):
        rc = main(["login", "--provider", "openai", "--api-key", "sk-ant-mistake"])
        assert rc == 2
        assert load_credentials() == {}
        err = capsys.readouterr().err
        assert "--provider anthropic" in err

    def test_rejects_xai_key_for_anthropic_provider(self, capsys):
        rc = main(["login", "--provider", "anthropic", "--api-key", "xai-mistake"])
        assert rc == 2
        assert load_credentials() == {}
        err = capsys.readouterr().err
        assert "sk-ant-" in err
        assert "--provider xai" in err

    def test_rejects_garbage_with_no_recognised_prefix(self, capsys):
        rc = main(["login", "--provider", "anthropic", "--api-key", "totally-bogus"])
        assert rc == 2
        assert load_credentials() == {}
        err = capsys.readouterr().err
        assert "sk-ant-" in err

    def test_accepts_sk_proj_for_openai(self, capsys):
        rc = main(["login", "--provider", "openai", "--api-key", "sk-proj-fine"])
        assert rc == 0
        assert load_credentials() == {"OPENAI_API_KEY": "sk-proj-fine"}

    def test_accepts_loose_format_for_gemini(self, capsys):
        rc = main(["login", "--provider", "gemini", "--api-key", "AIzaWhatever"])
        assert rc == 0
        assert load_credentials() == {"GEMINI_API_KEY": "AIzaWhatever"}

    def test_accepts_loose_format_for_google(self, capsys):
        rc = main(["login", "--provider", "google", "--api-key", "AIzaWhatever"])
        assert rc == 0
        assert load_credentials() == {"GOOGLE_API_KEY": "AIzaWhatever"}

    def test_invalid_prefix_emits_structured_json(self, capsys):
        rc = main(
            [
                "--output-format",
                "json",
                "login",
                "--provider",
                "anthropic",
                "--api-key",
                "sk-proj-mismatch",
            ]
        )
        assert rc == 2
        payload = json.loads(capsys.readouterr().out)
        assert payload["message"] == "invalid_key_prefix"
        assert payload["env_var"] == "ANTHROPIC_API_KEY"
        assert payload["expected_prefixes"] == ["sk-ant-"]
        assert payload["detected_env_var"] == "OPENAI_API_KEY"


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


class TestDoctor:
    """synthpanel doctor preflight — must never leak credentials (GH #310)."""

    def test_doctor_is_registered(self):
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"

    def test_doctor_fails_when_no_credentials(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.delenv("SYNTHPANEL_CREDENTIALS_PATH", raising=False)
        rc = main(["doctor"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "No LLM credentials found" in err

    def test_doctor_passes_when_env_credentials_set(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.delenv("SYNTHPANEL_CREDENTIALS_PATH", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-do-not-leak-this-value")
        rc = main(["doctor"])
        assert rc == 0
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "sk-ant-do-not-leak-this-value" not in combined
        assert "do-not-leak" not in combined

    def test_doctor_never_echoes_stored_credentials(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        save_credential("OPENAI_API_KEY", "sk-stored-ultra-secret-999")
        rc = main(["doctor"])
        assert rc == 0
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "sk-stored-ultra-secret-999" not in combined
        assert "ultra-secret" not in combined

    def test_doctor_json_has_no_secret_values(self, capsys, monkeypatch):
        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("XAI_API_KEY", "xai-json-secret-banned")
        rc = main(["--output-format", "json", "doctor"])
        assert rc == 0
        raw = capsys.readouterr().out
        assert "xai-json-secret-banned" not in raw
        payload = json.loads(raw)
        assert payload["message"] == "doctor_report"
        assert payload["credential_configured"] is True
        assert payload["checks_ok"] is True
        dumped = json.dumps(payload)
        assert "xai-json-secret-banned" not in dumped


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
