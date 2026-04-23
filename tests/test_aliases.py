"""Tests for model alias resolution."""

from __future__ import annotations

import json
import logging
import textwrap

import pytest

from synth_panel.llm.aliases import (
    _reset_cache,
    get_base_url_override,
    resolve_alias,
)


@pytest.fixture(autouse=True)
def _clean_alias_cache():
    """Reset the cached alias map before and after every test."""
    _reset_cache()
    yield
    _reset_cache()


# --- hardcoded (tier 3) ---


def test_known_aliases():
    assert resolve_alias("opus").startswith("claude-opus")
    assert resolve_alias("sonnet").startswith("claude-sonnet")
    assert resolve_alias("haiku").startswith("claude-haiku")
    assert resolve_alias("grok").startswith("grok-")


def test_passthrough():
    assert resolve_alias("claude-sonnet-4-6-20250414") == "claude-sonnet-4-6-20250414"
    assert resolve_alias("my-custom-model") == "my-custom-model"


# --- local prefixes ---


def test_ollama_prefix_stripped():
    assert resolve_alias("ollama:llama3") == "llama3"
    assert resolve_alias("ollama:mistral:7b") == "mistral:7b"


def test_local_prefix_stripped():
    assert resolve_alias("local:phi3") == "phi3"
    assert resolve_alias("local:codellama") == "codellama"


def test_get_base_url_override_ollama():
    assert get_base_url_override("ollama:llama3") == "http://localhost:11434"


def test_get_base_url_override_local():
    assert get_base_url_override("local:phi3") == "http://localhost:1234"


def test_get_base_url_override_none():
    assert get_base_url_override("sonnet") is None
    assert get_base_url_override("gpt-4o") is None


# --- env var override (tier 1) ---


def test_env_var_overrides_hardcoded(monkeypatch):
    monkeypatch.setenv("SYNTHPANEL_MODEL_ALIASES", json.dumps({"sonnet": "my-sonnet"}))
    assert resolve_alias("sonnet") == "my-sonnet"


def test_env_var_adds_new_alias(monkeypatch):
    monkeypatch.setenv("SYNTHPANEL_MODEL_ALIASES", json.dumps({"fast": "claude-haiku-4-5-20251001"}))
    assert resolve_alias("fast") == "claude-haiku-4-5-20251001"
    # hardcoded still works
    assert resolve_alias("opus").startswith("claude-opus")


def test_env_var_invalid_json_ignored(monkeypatch, caplog):
    monkeypatch.setenv("SYNTHPANEL_MODEL_ALIASES", "not json{")
    with caplog.at_level(logging.WARNING, logger="synth_panel.llm.aliases"):
        # falls back to hardcoded
        assert resolve_alias("sonnet").startswith("claude-sonnet")
    assert any("SYNTHPANEL_MODEL_ALIASES" in rec.getMessage() and "JSON" in rec.getMessage() for rec in caplog.records)


def test_env_var_non_dict_ignored(monkeypatch):
    monkeypatch.setenv("SYNTHPANEL_MODEL_ALIASES", json.dumps(["a", "b"]))
    assert resolve_alias("sonnet").startswith("claude-sonnet")


# --- file override (tier 2) ---


def test_file_overrides_hardcoded(monkeypatch, tmp_path):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        aliases:
          sonnet: custom-sonnet-model
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    assert resolve_alias("sonnet") == "custom-sonnet-model"


def test_file_adds_new_alias(monkeypatch, tmp_path):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        aliases:
          smart: claude-opus-4-6
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    assert resolve_alias("smart") == "claude-opus-4-6"
    assert resolve_alias("sonnet").startswith("claude-sonnet")


def test_file_missing_is_fine(monkeypatch, tmp_path):
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", tmp_path / "nope.yaml")
    assert resolve_alias("sonnet").startswith("claude-sonnet")


def test_file_invalid_yaml_ignored(monkeypatch, tmp_path, caplog):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(": : : not valid yaml [")
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    with caplog.at_level(logging.WARNING, logger="synth_panel.llm.aliases"):
        assert resolve_alias("sonnet").startswith("claude-sonnet")
    assert any(
        str(aliases_file) in rec.getMessage() and "failed to parse" in rec.getMessage() for rec in caplog.records
    )


def test_file_flat_format(monkeypatch, tmp_path):
    """Accept a flat dict (no 'aliases' wrapper) for convenience."""
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        fast: claude-haiku-4-5-20251001
        smart: claude-opus-4-6
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    assert resolve_alias("fast") == "claude-haiku-4-5-20251001"
    assert resolve_alias("smart") == "claude-opus-4-6"


# --- merge precedence ---


def test_env_beats_file(monkeypatch, tmp_path):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        aliases:
          sonnet: from-file
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    monkeypatch.setenv("SYNTHPANEL_MODEL_ALIASES", json.dumps({"sonnet": "from-env"}))
    assert resolve_alias("sonnet") == "from-env"


def test_file_beats_hardcoded(monkeypatch, tmp_path):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        aliases:
          sonnet: from-file
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    assert resolve_alias("sonnet") == "from-file"


def test_all_three_tiers_merge(monkeypatch, tmp_path):
    """Env > file > hardcoded, each can add unique keys."""
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text(
        textwrap.dedent("""\
        aliases:
          file-only: from-file
          shared: from-file
        """)
    )
    monkeypatch.setattr("synth_panel.llm.aliases._ALIASES_FILE", aliases_file)
    monkeypatch.setenv(
        "SYNTHPANEL_MODEL_ALIASES",
        json.dumps({"env-only": "from-env", "shared": "from-env"}),
    )
    assert resolve_alias("opus").startswith("claude-opus")  # hardcoded
    assert resolve_alias("file-only") == "from-file"  # file
    assert resolve_alias("env-only") == "from-env"  # env
    assert resolve_alias("shared") == "from-env"  # env wins
