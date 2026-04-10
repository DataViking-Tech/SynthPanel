"""Tests for model alias resolution."""

from __future__ import annotations

from synth_panel.llm.aliases import get_base_url_override, resolve_alias


def test_known_aliases():
    assert resolve_alias("opus").startswith("claude-opus")
    assert resolve_alias("sonnet").startswith("claude-sonnet")
    assert resolve_alias("haiku").startswith("claude-haiku")
    assert resolve_alias("grok").startswith("grok-")


def test_passthrough():
    assert resolve_alias("claude-sonnet-4-6-20250414") == "claude-sonnet-4-6-20250414"
    assert resolve_alias("my-custom-model") == "my-custom-model"


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
