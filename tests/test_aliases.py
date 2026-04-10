"""Tests for model alias resolution."""

from __future__ import annotations

from synth_panel.llm.aliases import resolve_alias


def test_known_aliases():
    assert resolve_alias("opus").startswith("claude-opus")
    assert resolve_alias("sonnet").startswith("claude-sonnet")
    assert resolve_alias("haiku").startswith("claude-haiku")
    assert resolve_alias("grok").startswith("grok-")


def test_passthrough():
    assert resolve_alias("claude-sonnet-4-6-20250414") == "claude-sonnet-4-6-20250414"
    assert resolve_alias("my-custom-model") == "my-custom-model"
