"""Tests for the credential store (sp-lve)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from synth_panel.credentials import (
    CredentialIntegrityError,
    PROVIDER_KEY_PREFIXES,
    credentials_path,
    delete_credential,
    detect_provider_from_key,
    get_credential,
    has_credential,
    load_credentials,
    save_credential,
)


class TestCredentialsPath:
    def test_honors_override_env_var(self, monkeypatch, tmp_path):
        override = tmp_path / "custom.json"
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(override))
        assert credentials_path() == override

    def test_respects_xdg_config_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SYNTHPANEL_CREDENTIALS_PATH", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert credentials_path() == tmp_path / "synthpanel" / "credentials.json"

    def test_default_is_under_home_dot_config(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SYNTHPANEL_CREDENTIALS_PATH", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        assert credentials_path() == tmp_path / ".config" / "synthpanel" / "credentials.json"


class TestSaveAndLoad:
    def test_save_then_load_roundtrip(self):
        save_credential("ANTHROPIC_API_KEY", "sk-test-1")
        save_credential("OPENAI_API_KEY", "sk-test-2")
        loaded = load_credentials()
        assert loaded == {"ANTHROPIC_API_KEY": "sk-test-1", "OPENAI_API_KEY": "sk-test-2"}

    def test_save_overwrites_existing_value(self):
        save_credential("ANTHROPIC_API_KEY", "sk-old")
        save_credential("ANTHROPIC_API_KEY", "sk-new")
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-new"}

    def test_save_normalises_whitespace_and_case(self):
        path = save_credential("  anthropic_api_key  ", "  sk-val  ")
        data = json.loads(path.read_text())
        assert data == {"ANTHROPIC_API_KEY": "sk-val"}

    def test_save_rejects_empty_value(self):
        with pytest.raises(ValueError):
            save_credential("ANTHROPIC_API_KEY", "   ")

    def test_save_rejects_empty_env_var(self):
        with pytest.raises(ValueError):
            save_credential("", "sk-test")

    def test_load_returns_empty_when_missing(self):
        assert load_credentials() == {}

    def test_load_tolerates_malformed_json(self, monkeypatch, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(path))
        assert load_credentials() == {}

    def test_load_tolerates_wrong_top_level_type(self, monkeypatch, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(path))
        assert load_credentials() == {}

    def test_load_drops_non_string_entries(self, monkeypatch, tmp_path):
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps({"ANTHROPIC_API_KEY": "sk-ok", "N": 42, "X": ""}))
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(path))
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-ok"}


class TestFilePermissions:
    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_credential_file_is_mode_0600(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-test")
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == 0o600

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_parent_directory_is_mode_0700(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-test")
        parent_mode = stat.S_IMODE(path.parent.stat().st_mode)
        assert parent_mode == 0o700


class TestDelete:
    def test_delete_removes_entry(self):
        save_credential("ANTHROPIC_API_KEY", "sk-a")
        save_credential("OPENAI_API_KEY", "sk-b")
        assert delete_credential("ANTHROPIC_API_KEY") is True
        assert load_credentials() == {"OPENAI_API_KEY": "sk-b"}

    def test_delete_missing_returns_false(self):
        assert delete_credential("ANTHROPIC_API_KEY") is False

    def test_delete_last_entry_removes_file(self):
        save_credential("ANTHROPIC_API_KEY", "sk-only")
        path = credentials_path()
        assert path.exists()
        assert delete_credential("ANTHROPIC_API_KEY") is True
        assert not path.exists()


class TestGetCredential:
    def test_env_var_wins_over_stored(self, monkeypatch):
        save_credential("ANTHROPIC_API_KEY", "sk-stored")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        assert get_credential("ANTHROPIC_API_KEY") == "sk-env"

    def test_falls_back_to_stored_when_env_empty(self, monkeypatch):
        save_credential("ANTHROPIC_API_KEY", "sk-stored")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert get_credential("ANTHROPIC_API_KEY") == "sk-stored"

    def test_whitespace_only_env_is_treated_as_unset(self, monkeypatch):
        save_credential("ANTHROPIC_API_KEY", "sk-stored")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
        assert get_credential("ANTHROPIC_API_KEY") == "sk-stored"

    def test_returns_none_when_nowhere(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert get_credential("ANTHROPIC_API_KEY") is None

    def test_has_credential_mirrors_get_credential(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert has_credential("ANTHROPIC_API_KEY") is False
        save_credential("ANTHROPIC_API_KEY", "sk-x")
        assert has_credential("ANTHROPIC_API_KEY") is True


class TestSha256Sidecar:
    def test_save_writes_sidecar(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-x")
        sidecar = path.parent / (path.name + ".sha256")
        assert sidecar.exists()
        assert len(sidecar.read_text().strip()) == 64  # hex sha256

    def test_load_passes_when_sidecar_matches(self):
        save_credential("ANTHROPIC_API_KEY", "sk-x")
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-x"}

    def test_load_raises_on_sidecar_mismatch(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-x")
        path.write_text('{"ANTHROPIC_API_KEY": "tampered"}\n', encoding="utf-8")
        with pytest.raises(CredentialIntegrityError, match="synthpanel login"):
            load_credentials()

    def test_migration_generates_sidecar_on_first_read(self, monkeypatch, tmp_path):
        creds = tmp_path / "credentials.json"
        creds.write_text('{"ANTHROPIC_API_KEY": "sk-legacy"}\n', encoding="utf-8")
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(creds))
        result = load_credentials()
        assert result == {"ANTHROPIC_API_KEY": "sk-legacy"}
        sidecar = tmp_path / "credentials.json.sha256"
        assert sidecar.exists()

    def test_migration_sidecar_matches_content(self, monkeypatch, tmp_path):
        import hashlib

        content = '{"ANTHROPIC_API_KEY": "sk-legacy"}\n'
        creds = tmp_path / "credentials.json"
        creds.write_text(content, encoding="utf-8")
        monkeypatch.setenv("SYNTHPANEL_CREDENTIALS_PATH", str(creds))
        load_credentials()
        sidecar = tmp_path / "credentials.json.sha256"
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert sidecar.read_text().strip() == expected

    def test_save_login_recovers_from_tampered_file(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-original")
        path.write_text('{"ANTHROPIC_API_KEY": "tampered"}\n', encoding="utf-8")
        # login should succeed despite mismatch (starts fresh)
        new_path = save_credential("ANTHROPIC_API_KEY", "sk-new")
        assert load_credentials() == {"ANTHROPIC_API_KEY": "sk-new"}
        assert new_path == path

    def test_delete_updates_sidecar_when_entries_remain(self):
        import hashlib

        save_credential("ANTHROPIC_API_KEY", "sk-a")
        save_credential("OPENAI_API_KEY", "sk-b")
        delete_credential("ANTHROPIC_API_KEY")
        path = credentials_path()
        sidecar = path.parent / (path.name + ".sha256")
        content = path.read_text(encoding="utf-8")
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert sidecar.read_text().strip() == expected

    def test_delete_removes_sidecar_when_file_deleted(self):
        save_credential("ANTHROPIC_API_KEY", "sk-only")
        path = credentials_path()
        sidecar = path.parent / (path.name + ".sha256")
        delete_credential("ANTHROPIC_API_KEY")
        assert not path.exists()
        assert not sidecar.exists()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_sidecar_file_is_mode_0600(self):
        path = save_credential("ANTHROPIC_API_KEY", "sk-x")
        sidecar = path.parent / (path.name + ".sha256")
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


class TestProviderIntegration:
    """Provider config reads credentials through ``get_credential``."""

    def test_anthropic_provider_uses_stored_credential(self, monkeypatch):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        save_credential("ANTHROPIC_API_KEY", "sk-disk")
        provider = AnthropicProvider()
        assert provider._api_key == "sk-disk"

    def test_anthropic_provider_env_wins_over_stored(self, monkeypatch):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        save_credential("ANTHROPIC_API_KEY", "sk-disk")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        provider = AnthropicProvider()
        assert provider._api_key == "sk-env"

    def test_gemini_provider_uses_stored_credential(self, monkeypatch):
        from synth_panel.llm.providers.gemini import GeminiProvider

        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        save_credential("GOOGLE_API_KEY", "sk-gem")
        provider = GeminiProvider()
        assert provider._api_key == "sk-gem"

    def test_missing_credentials_error_mentions_login(self, monkeypatch):
        from synth_panel.llm.errors import LLMErrorCategory
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # No stored credential thanks to the autouse conftest fixture.
        with pytest.raises(Exception) as exc_info:
            AnthropicProvider()
        err = exc_info.value
        assert getattr(err, "category", None) == LLMErrorCategory.MISSING_CREDENTIALS
        assert "synthpanel login" in str(err)


class TestProviderKeyPrefixes:
    """Prefix table + key detection for cross-provider mistake catch (sy-bybx)."""

    def test_table_covers_every_known_env_var(self):
        from synth_panel.credentials import KNOWN_CREDENTIAL_ENV_VARS

        # Each known credential has an entry — even if the entry is `()`
        # to mean "no prefix convention". Missing entries would silently
        # disable validation for that provider.
        for env_var in KNOWN_CREDENTIAL_ENV_VARS:
            assert env_var in PROVIDER_KEY_PREFIXES

    def test_detects_anthropic_key(self):
        assert detect_provider_from_key("sk-ant-foo") == "ANTHROPIC_API_KEY"

    def test_detects_openrouter_key(self):
        assert detect_provider_from_key("sk-or-bar") == "OPENROUTER_API_KEY"

    def test_detects_openai_project_key(self):
        assert detect_provider_from_key("sk-proj-baz") == "OPENAI_API_KEY"

    def test_detects_xai_key(self):
        assert detect_provider_from_key("xai-key") == "XAI_API_KEY"

    def test_falls_back_to_openai_for_bare_sk_prefix(self):
        # ``sk-`` is OpenAI's generic prefix, but it's also a substring of
        # other providers' distinctive prefixes — make sure we resolve it
        # to OpenAI only when nothing more specific matches.
        assert detect_provider_from_key("sk-something") == "OPENAI_API_KEY"

    def test_returns_none_for_unrecognised_format(self):
        assert detect_provider_from_key("AIzaSyGeminiStyle") is None
        assert detect_provider_from_key("totally-bogus") is None
        assert detect_provider_from_key("") is None


class TestClientErrorMessage:
    def test_no_credentials_error_mentions_login(self, monkeypatch):
        from synth_panel.llm.client import LLMClient
        from synth_panel.llm.errors import LLMError, LLMErrorCategory
        from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock

        for env in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(env, raising=False)

        client = LLMClient()
        req = CompletionRequest(
            model="mystery-unknown-model",
            max_tokens=16,
            messages=[InputMessage(role="user", content=[TextBlock(text="hi")])],
        )
        with pytest.raises(LLMError) as exc_info:
            client.send(req)
        assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS
        assert "synthpanel login" in str(exc_info.value)
