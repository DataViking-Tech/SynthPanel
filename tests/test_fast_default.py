"""Tests for the shared large-panel fast-default swap (synthbench#261).

The policy lives in :mod:`althing.llm.fast_default` and is consumed
by all three entry points: MCP (``_resolve_mcp_default_model_for_panel``,
covered in tests/test_mcp_server.py), SDK (``sdk._default_model_for_panel``),
and CLI (``panel run`` when ``--model`` is omitted).
"""

from __future__ import annotations

import json

import pytest

from althing.llm import fast_default
from althing.main import main

_ALL_KEY_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_keys(monkeypatch):
    for var in _ALL_KEY_VARS:
        monkeypatch.delenv(var, raising=False)


def _write_panel_files(tmp_path, persona_count: int):
    personas_file = tmp_path / "personas.yaml"
    lines = ["personas:"] + [f"  - name: Persona {i}" for i in range(persona_count)]
    personas_file.write_text("\n".join(lines) + "\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  questions:\n    - text: Question 1?\n")
    return personas_file, survey_file


class TestSharedPolicy:
    def test_swaps_openrouter_auto_at_threshold(self):
        model, swapped_from = fast_default.fast_default_for_panel(
            "openrouter/auto", fast_default.LARGE_PANEL_PERSONA_THRESHOLD
        )
        assert model == "openrouter/anthropic/claude-haiku-4.5"
        assert swapped_from == "openrouter/auto"

    def test_no_swap_below_threshold(self):
        model, swapped_from = fast_default.fast_default_for_panel(
            "openrouter/auto", fast_default.LARGE_PANEL_PERSONA_THRESHOLD - 1
        )
        assert model == "openrouter/auto"
        assert swapped_from is None

    def test_fast_aliases_untouched(self):
        for alias in ("haiku", "sonnet", "gpt-4o-mini", "gemini-2.5-flash", "grok-3"):
            model, swapped_from = fast_default.fast_default_for_panel(alias, 50)
            assert model == alias
            assert swapped_from is None

    def test_note_mentions_model_count_and_override(self):
        note = fast_default.format_fast_default_note(
            "openrouter/anthropic/claude-haiku-4.5", "openrouter/auto", 20, override_hint="--model"
        )
        assert "openrouter/anthropic/claude-haiku-4.5" in note
        assert "20-persona" in note
        assert "--model" in note

    def test_mcp_reexports_shared_values(self):
        """server.py must consume the shared policy, not a private copy."""
        pytest.importorskip("mcp")
        from althing.mcp import server

        assert server.LARGE_PANEL_PERSONA_THRESHOLD is fast_default.LARGE_PANEL_PERSONA_THRESHOLD
        assert server._LARGE_PANEL_FAST_MODEL_SWAP is fast_default.FAST_MODEL_SWAP


class TestSdkDefault:
    def test_openrouter_only_env_swaps_for_large_panel(self, monkeypatch, caplog):
        from althing import sdk

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        with caplog.at_level("WARNING", logger="althing.sdk"):
            model = sdk._default_model_for_panel(20)
        assert model == "openrouter/anthropic/claude-haiku-4.5"
        assert any("auto-selected" in r.getMessage() for r in caplog.records)

    def test_openrouter_only_env_keeps_auto_for_small_panel(self, monkeypatch):
        from althing import sdk

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        assert sdk._default_model_for_panel(3) == "openrouter/auto"

    def test_anthropic_default_untouched(self, monkeypatch):
        from althing import sdk

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
        assert sdk._default_model_for_panel(50) == "sonnet"


class TestCliPanelRun:
    def test_dry_run_swaps_default_and_notes_on_stderr(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        personas_file, survey_file = _write_panel_files(tmp_path, 12)

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
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        payload = json.loads(captured.out)
        assert payload["model"] == "openrouter/anthropic/claude-haiku-4.5"
        assert "auto-selected openrouter/anthropic/claude-haiku-4.5" in captured.err
        assert "--model" in captured.err

    def test_dry_run_small_panel_keeps_openrouter_auto(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        personas_file, survey_file = _write_panel_files(tmp_path, 3)

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
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        assert json.loads(captured.out)["model"] == "openrouter/auto"
        assert "auto-selected" not in captured.err

    def test_explicit_model_never_swapped(self, monkeypatch, capsys, tmp_path):
        """A deliberate openrouter/auto is honored even for large panels."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        personas_file, survey_file = _write_panel_files(tmp_path, 12)

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
                "openrouter/auto",
                "--dry-run",
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        assert json.loads(captured.out)["model"] == "openrouter/auto"
        assert "auto-selected" not in captured.err

    def test_anthropic_default_untouched_for_large_panel(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
        personas_file, survey_file = _write_panel_files(tmp_path, 12)

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
            ]
        )
        captured = capsys.readouterr()
        assert code == 0
        assert json.loads(captured.out)["model"] == "sonnet"
        assert "auto-selected" not in captured.err
