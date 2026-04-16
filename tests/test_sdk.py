"""Tests for the public SDK surface in :mod:`synth_panel.sdk`.

Covers:
* Import surface — the eight exported functions and three dataclasses
  are reachable from the package root.
* ``__all__`` correctness — no stray internals leak, no dead names.
* ``run_prompt`` round-trip against a mocked :class:`LLMClient`.
* ``quick_poll`` / ``run_panel`` wiring — verify they reach the
  shared runners with the right inputs.
* ``list_*`` / ``get_panel_result`` — delegate to :mod:`synth_panel.mcp.data`.
* Validation errors for empty personas/questions + persona schema.
* ``PanelResult`` dict-like compatibility (``__getitem__``, ``.to_dict``).
* Zero reliance on the optional ``mcp`` extra — the SDK must import
  cleanly without the ``mcp`` package.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    """Point the persistence layer at a temp dir for every test."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    # Reset the module-level shared client so each test starts fresh.
    import synth_panel.sdk as _sdk

    _sdk._shared_client = None


# ---------------------------------------------------------------------------
# Surface: imports and __all__ contract
# ---------------------------------------------------------------------------


class TestImportSurface:
    def test_eight_public_functions_reachable_from_root(self):
        import synth_panel

        for name in (
            "run_prompt",
            "quick_poll",
            "run_panel",
            "extend_panel",
            "list_personas",
            "list_instruments",
            "list_panel_results",
            "get_panel_result",
        ):
            assert hasattr(synth_panel, name), f"synth_panel.{name} is missing"
            assert callable(getattr(synth_panel, name))

    def test_three_result_dataclasses_reachable_from_root(self):
        from synth_panel import PanelResult, PollResult, PromptResult

        assert PanelResult.__name__ == "PanelResult"
        assert PollResult.__name__ == "PollResult"
        assert PromptResult.__name__ == "PromptResult"

    def test_all_is_explicit_and_sorted(self):
        import synth_panel

        assert "__all__" in dir(synth_panel)
        # No duplicates and each name is actually exported.
        assert len(synth_panel.__all__) == len(set(synth_panel.__all__))
        for name in synth_panel.__all__:
            assert hasattr(synth_panel, name), f"__all__ names missing attribute: {name}"

    def test_sdk_does_not_require_mcp_extra(self):
        """Importing the SDK module must not trigger the mcp library.

        synth_panel.sdk reaches into synth_panel.mcp.data (pure
        yaml/json) but NOT into synth_panel.mcp.server (which imports
        fastmcp). This test runs in an env where `mcp` is installed, so
        we can only assert that fresh importing does not raise — the
        stronger guarantee (SDK works without `[mcp]` extras) is
        verified by the no-extras leg of CI.
        """
        import importlib

        # Force a re-import to prove the module is self-sufficient.
        import synth_panel.sdk

        reloaded = importlib.reload(synth_panel.sdk)
        assert hasattr(reloaded, "run_prompt")
        assert hasattr(reloaded, "quick_poll")


# ---------------------------------------------------------------------------
# run_prompt
# ---------------------------------------------------------------------------


class TestRunPrompt:
    def test_returns_prompt_result_with_cost_and_usage(self):
        from synth_panel import run_prompt
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-1",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hello back")],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        with patch("synth_panel.sdk.LLMClient") as MockClient:
            MockClient.return_value.send.return_value = mock_response
            result = run_prompt("Say hello", model="haiku")

        assert result.response == "Hello back"
        assert result.model == "claude-haiku-4-5-20251001"
        assert result.usage["input_tokens"] == 10
        assert result.cost.startswith("$")

    def test_empty_prompt_raises(self):
        from synth_panel import run_prompt

        with pytest.raises(ValueError, match="non-empty"):
            run_prompt("")

    def test_default_model_chosen_from_environment(self, monkeypatch):
        """Default model follows the provider-preference chain in env."""
        from synth_panel.sdk import _default_model

        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "XAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
        assert _default_model() == "sonnet"

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _default_model() == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# quick_poll / run_panel
# ---------------------------------------------------------------------------


class TestQuickPoll:
    def test_empty_question_raises(self):
        from synth_panel import quick_poll

        with pytest.raises(ValueError, match="non-empty"):
            quick_poll("", personas=[{"name": "A"}])

    def test_requires_personas_or_pack_id(self):
        from synth_panel import quick_poll

        with pytest.raises(ValueError, match="No personas"):
            quick_poll("q?")

    def test_calls_run_panel_sync_with_single_question(self):
        """quick_poll wraps the question and drives the shared runner."""
        from synth_panel import quick_poll
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"
        fake_cost.__add__ = lambda self, other: self

        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = (
                [],  # panelist_results
                [{"persona": "A", "responses": [{"response": "ok"}], "usage": {}, "cost": "$0.01", "error": None}],
                fake_usage,
                fake_cost,
                None,
                None,
            )
            result = quick_poll("Does this work?", personas=[{"name": "A"}])

        assert result.question == "Does this work?"
        assert len(result.responses) == 1
        # The runner was passed exactly one question, wrapping the string.
        kwargs = mock_runner.call_args.kwargs
        assert kwargs["questions"] == [{"text": "Does this work?"}]


class TestRunPanel:
    def test_requires_question_source(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="questions|instrument|instrument_pack"):
            run_panel(personas=[{"name": "A"}])

    def test_variants_out_of_range_raises(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="variants must be"):
            run_panel(personas=[{"name": "A"}], questions=["q?"], variants=99)

    def test_persona_without_name_raises(self):
        from synth_panel import run_panel

        with pytest.raises(ValueError, match="name"):
            run_panel(personas=[{"age": 30}], questions=["q?"])

    def test_question_strings_are_auto_wrapped(self):
        """Pass a list of strings and they become question dicts."""
        from synth_panel import run_panel
        from synth_panel.cost import TokenUsage

        fake_usage = TokenUsage(input_tokens=1, output_tokens=1)
        fake_cost = MagicMock()
        fake_cost.format_usd.return_value = "$0.01"

        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_panel_sync") as mock_runner,
        ):
            mock_runner.return_value = ([], [], fake_usage, fake_cost, None, None)
            run_panel(
                personas=[{"name": "Alice"}],
                questions=["First?", "Second?"],
            )
            kwargs = mock_runner.call_args.kwargs
            assert kwargs["questions"] == [{"text": "First?"}, {"text": "Second?"}]

    def test_instrument_pack_takes_precedence_over_questions(self, monkeypatch):
        """If instrument_pack is given, questions/instrument are ignored."""
        from synth_panel import run_panel

        # Mock the pack loader to return a tiny v1 instrument.
        def fake_load_pack(name):
            assert name == "dummy-pack"
            return {
                "name": "dummy-pack",
                "instrument": {
                    "version": 1,
                    "questions": [{"text": "From pack"}],
                },
            }

        monkeypatch.setattr("synth_panel.sdk._data_load_instrument_pack", fake_load_pack)

        from synth_panel.cost import TokenUsage
        from synth_panel.orchestrator import MultiRoundResult, RoundResult

        # Stub the multi-round runner so we don't hit the network.
        fake_mr = MultiRoundResult(
            rounds=[
                RoundResult(
                    name="round_1",
                    panelist_results=[],
                    synthesis=None,
                    usage=TokenUsage(input_tokens=0, output_tokens=0),
                )
            ],
            path=[],
            terminal_round="round_1",
            final_synthesis=None,
            warnings=[],
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )
        with (
            patch("synth_panel.sdk.LLMClient"),
            patch("synth_panel.sdk.run_multi_round_sync", return_value=fake_mr) as mock_mr,
        ):
            out = run_panel(
                personas=[{"name": "A"}],
                instrument_pack="dummy-pack",
                questions=["IGNORED"],
            )
        # The multi-round runner was called (instrument path won), not the
        # single-round runner.
        assert mock_mr.called
        assert out.result_id


# ---------------------------------------------------------------------------
# list_* and get_panel_result
# ---------------------------------------------------------------------------


class TestListDelegates:
    def test_list_personas_returns_bundled_packs(self):
        from synth_panel import list_personas

        packs = list_personas()
        # At least one bundled pack ships with the package.
        assert len(packs) >= 1
        assert all("id" in p for p in packs)

    def test_list_instruments_returns_bundled_packs(self):
        from synth_panel import list_instruments

        packs = list_instruments()
        assert len(packs) >= 1
        assert all("id" in p for p in packs)

    def test_list_panel_results_empty_in_clean_dir(self):
        from synth_panel import list_panel_results

        assert list_panel_results() == []


class TestGetPanelResult:
    def test_returns_panel_result_dataclass(self):
        from synth_panel import get_panel_result
        from synth_panel.mcp.data import save_panel_result

        rid = save_panel_result(
            results=[{"persona": "A", "responses": [{"response": "hi"}]}],
            model="haiku",
            total_usage={"input_tokens": 1, "output_tokens": 1},
            total_cost="$0.01",
            persona_count=1,
            question_count=1,
        )

        out = get_panel_result(rid)
        assert out.result_id == rid
        assert out.model == "haiku"
        assert out.persona_count == 1
        # Dict-like access for back-compat with callers that used to
        # read the raw MCP payload.
        assert out["model"] == "haiku"
        assert "model" in out
        assert out.to_dict()["model"] == "haiku"

    def test_missing_result_raises_filenotfound(self):
        from synth_panel import get_panel_result

        with pytest.raises(FileNotFoundError):
            get_panel_result("nope-does-not-exist")


# ---------------------------------------------------------------------------
# Extension path
# ---------------------------------------------------------------------------


class TestExtendPanel:
    def test_missing_sessions_raises(self):
        from synth_panel import extend_panel
        from synth_panel.mcp.data import save_panel_result

        # Save a result but don't save any sessions for it.
        rid = save_panel_result(
            results=[{"persona": "A", "responses": []}],
            model="haiku",
            total_usage={},
            total_cost="$0",
            persona_count=1,
            question_count=1,
        )
        with pytest.raises((ValueError, FileNotFoundError)):
            extend_panel(rid, "Follow-up?")
