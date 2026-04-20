"""Tests for MCP sampling bridge (``synth_panel.mcp.sampling``).

Covers the four routing branches required by sp-6at:

1. sampling supported + no creds  → sampling path
2. sampling supported + creds     → BYOK path (default)
3. sampling supported + creds + ``use_sampling=True`` → sampling path
4. no sampling + no creds         → friendly error
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp")


# ---------------------------------------------------------------------------
# Env fixture — sampling tests own their env explicitly, overriding the
# BYOK-simulating default from tests/test_mcp_server.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    """Scrub provider env vars — sampling tests opt into them per-case."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# Unit tests for the pure helpers in synth_panel.mcp.sampling
# ---------------------------------------------------------------------------


class TestHasByokCredentials:
    def test_no_creds_returns_false(self):
        from synth_panel.mcp.sampling import has_byok_credentials

        assert has_byok_credentials({}) is False

    def test_any_known_var_returns_true(self, monkeypatch):
        from synth_panel.mcp.sampling import has_byok_credentials

        # Must cover every provider the CLI auto-detects, otherwise a user
        # whose only key is (e.g.) OPENROUTER_API_KEY gets misrouted into
        # sampling or a "missing credentials" error despite the CLI
        # recognising the same key (see sp-t6r).
        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            assert has_byok_credentials({var: "x"}) is True, var

    def test_openrouter_key_is_recognised_in_decide_mode(self):
        from synth_panel.mcp.sampling import decide_mode

        # sp-t6r regression: OPENROUTER_API_KEY must route to BYOK, not
        # sampling, even when the client supports sampling — otherwise we
        # silently downgrade users with a valid provider key.
        ctx = MagicMock()
        ctx.session.check_client_capability.return_value = True
        d = decide_mode(ctx, env={"OPENROUTER_API_KEY": "sk-or-x"})
        assert d.mode == "byok"

    def test_empty_string_is_not_creds(self):
        from synth_panel.mcp.sampling import has_byok_credentials

        assert has_byok_credentials({"ANTHROPIC_API_KEY": "   "}) is False


class TestClientSupportsSampling:
    def test_none_ctx_returns_false(self):
        from synth_panel.mcp.sampling import client_supports_sampling

        assert client_supports_sampling(None) is False

    def test_session_raising_property_returns_false(self):
        """``ctx.session`` outside a request raises — must not propagate."""
        from synth_panel.mcp.sampling import client_supports_sampling

        ctx = MagicMock()
        # Simulate FastMCP Context.session property raising ValueError
        type(ctx).session = property(lambda self: (_ for _ in ()).throw(ValueError("outside request")))
        assert client_supports_sampling(ctx) is False

    def test_session_check_returns_true(self):
        from synth_panel.mcp.sampling import client_supports_sampling

        ctx = MagicMock()
        ctx.session.check_client_capability.return_value = True
        assert client_supports_sampling(ctx) is True

    def test_session_check_returns_false(self):
        from synth_panel.mcp.sampling import client_supports_sampling

        ctx = MagicMock()
        ctx.session.check_client_capability.return_value = False
        assert client_supports_sampling(ctx) is False


class TestDecideMode:
    """The four-branch routing matrix — the heart of the acceptance tests."""

    def _ctx(self, *, supports: bool):
        ctx = MagicMock()
        ctx.session.check_client_capability.return_value = supports
        return ctx

    def test_auto_sampling_supported_no_creds_picks_sampling(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(self._ctx(supports=True), env={})
        assert d.mode == "sampling"
        assert d.hint is not None  # first-run hint

    def test_auto_sampling_supported_with_creds_picks_byok(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(
            self._ctx(supports=True),
            env={"ANTHROPIC_API_KEY": "sk-x"},
        )
        assert d.mode == "byok"
        assert d.hint is None

    def test_auto_no_sampling_no_creds_returns_friendly_error(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(self._ctx(supports=False), env={})
        assert d.mode == "error"
        assert d.error is not None
        assert "ANTHROPIC_API_KEY" in d.error
        assert "sampling-capable" in d.error

    def test_explicit_use_sampling_true_with_creds_picks_sampling(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(
            self._ctx(supports=True),
            use_sampling=True,
            env={"ANTHROPIC_API_KEY": "sk-x"},
        )
        assert d.mode == "sampling"

    def test_explicit_use_sampling_true_but_unsupported_errors(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(self._ctx(supports=False), use_sampling=True, env={})
        assert d.mode == "error"
        assert "use_sampling=True" in d.error

    def test_explicit_use_sampling_false_forces_byok(self):
        from synth_panel.mcp.sampling import decide_mode

        d = decide_mode(self._ctx(supports=True), use_sampling=False, env={})
        assert d.mode == "byok"


# ---------------------------------------------------------------------------
# Integration — run_prompt across the four branches
# ---------------------------------------------------------------------------


def _make_sampling_ctx(supports: bool, *, sample_text: str = "sampled!"):
    """Build a MagicMock Context that the tool can invoke without
    hitting the FastMCP test harness's Context auto-injection path."""
    ctx = MagicMock()
    ctx.session.check_client_capability.return_value = supports

    async def _create_message(**kwargs: Any):
        # Mimic CreateMessageResult — only the fields _sample_text reads.
        msg = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = sample_text
        msg.content = text_block
        msg.model = "host-agent-model"
        msg.role = "assistant"
        msg.stopReason = "endTurn"
        return msg

    ctx.session.create_message = AsyncMock(side_effect=_create_message)
    ctx.report_progress = AsyncMock()
    return ctx


class TestRunPromptSamplingBranches:
    @pytest.mark.asyncio
    async def test_sampling_supported_no_creds_uses_sampling(self):
        from synth_panel.mcp.server import run_prompt

        ctx = _make_sampling_ctx(supports=True, sample_text="hi from host")
        raw = await run_prompt(prompt="Say hi", ctx=ctx)
        data = json.loads(raw)

        assert data["mode"] == "sampling"
        assert data["response"] == "hi from host"
        assert data["model"] == "host-agent-model"
        assert data["hint"] is not None
        ctx.session.create_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_sampling_supported_with_creds_uses_byok(self, monkeypatch):
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage
        from synth_panel.mcp.server import run_prompt

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        ctx = _make_sampling_ctx(supports=True)

        mock_response = CompletionResponse(
            id="r",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="byok-output")],
            usage=TokenUsage(input_tokens=3, output_tokens=2),
        )
        with (
            patch("synth_panel.mcp.server._shared_client", None),
            patch("synth_panel.mcp.server.LLMClient") as MockClient,
        ):
            MockClient.return_value.send.return_value = mock_response
            raw = await run_prompt(prompt="Hi", ctx=ctx)

        data = json.loads(raw)
        assert data["mode"] == "byok"
        assert data["response"] == "byok-output"
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_use_sampling_overrides_creds(self, monkeypatch):
        from synth_panel.mcp.server import run_prompt

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        ctx = _make_sampling_ctx(supports=True, sample_text="sampled-anyway")

        raw = await run_prompt(prompt="Hi", use_sampling=True, ctx=ctx)
        data = json.loads(raw)

        assert data["mode"] == "sampling"
        assert data["response"] == "sampled-anyway"

    @pytest.mark.asyncio
    async def test_no_sampling_no_creds_returns_friendly_error(self):
        from synth_panel.mcp.server import run_prompt

        ctx = _make_sampling_ctx(supports=False)
        raw = await run_prompt(prompt="Hi", ctx=ctx)
        data = json.loads(raw)

        assert "error" in data
        assert "ANTHROPIC_API_KEY" in data["error"]
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_use_sampling_true_but_unsupported_errors(self):
        from synth_panel.mcp.server import run_prompt

        ctx = _make_sampling_ctx(supports=False)
        raw = await run_prompt(prompt="Hi", use_sampling=True, ctx=ctx)
        data = json.loads(raw)

        assert "error" in data
        assert "use_sampling=True" in data["error"]


# ---------------------------------------------------------------------------
# Integration — run_quick_poll sampling path
# ---------------------------------------------------------------------------


class TestRunQuickPollSampling:
    @pytest.mark.asyncio
    async def test_sampling_runs_per_persona_and_synthesis(self):
        from synth_panel.mcp.server import run_quick_poll

        ctx = _make_sampling_ctx(supports=True, sample_text="persona response")
        personas = [{"name": "Alice"}, {"name": "Bob"}]
        raw = await run_quick_poll(
            question="Is the sky blue?",
            personas=personas,
            ctx=ctx,
        )
        data = json.loads(raw)

        assert data["mode"] == "sampling"
        assert data["persona_count"] == 2
        assert data["question_count"] == 1
        assert len(data["results"]) == 2
        assert data["synthesis"] is not None
        # 2 persona calls + 1 synthesis call
        assert ctx.session.create_message.await_count == 3

    @pytest.mark.asyncio
    async def test_sampling_persona_cap_enforced(self):
        from synth_panel.mcp.server import SAMPLING_MAX_PERSONAS, run_quick_poll

        ctx = _make_sampling_ctx(supports=True)
        personas = [{"name": f"P{i}"} for i in range(SAMPLING_MAX_PERSONAS + 1)]
        raw = await run_quick_poll(
            question="test?",
            personas=personas,
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert f"{SAMPLING_MAX_PERSONAS} personas" in data["error"]
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_byok_path_unchanged_with_creds(self, monkeypatch):
        from synth_panel.mcp.server import run_quick_poll

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        ctx = _make_sampling_ctx(supports=True)

        mock_run = AsyncMock(return_value={"results": [], "model": "haiku"})
        with patch("synth_panel.mcp.server._run_panel_async", mock_run):
            raw = await run_quick_poll(
                question="q?",
                personas=[{"name": "Alice"}],
                ctx=ctx,
            )
        data = json.loads(raw)
        assert data["mode"] == "byok"
        mock_run.assert_awaited_once()
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sampling_no_creds_returns_error(self):
        from synth_panel.mcp.server import run_quick_poll

        ctx = _make_sampling_ctx(supports=False)
        raw = await run_quick_poll(
            question="q?",
            personas=[{"name": "Alice"}],
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert "ANTHROPIC_API_KEY" in data["error"]


# ---------------------------------------------------------------------------
# Integration — run_panel sampling fallback (sp-5no)
# ---------------------------------------------------------------------------


class TestRunPanelSampling:
    @pytest.mark.asyncio
    async def test_sampling_supported_no_creds_runs_panel(self):
        from synth_panel.mcp.server import run_panel

        ctx = _make_sampling_ctx(supports=True, sample_text="panel-answer")
        raw = await run_panel(
            questions=[{"text": "What do you think?"}],
            personas=[{"name": "Alice"}, {"name": "Bob"}],
            ctx=ctx,
        )
        data = json.loads(raw)

        assert data["mode"] == "sampling"
        assert data["persona_count"] == 2
        assert data["question_count"] == 1
        # 2 persona calls + 1 synthesis (default on)
        assert ctx.session.create_message.await_count == 3
        # sampling path never raises on the 'usage' field
        assert data["usage"] is None
        assert data["results"][0]["responses"][0]["answer"] == "panel-answer"

    @pytest.mark.asyncio
    async def test_sampling_persona_cap_enforced(self):
        from synth_panel.mcp.server import SAMPLING_MAX_PERSONAS, run_panel

        ctx = _make_sampling_ctx(supports=True)
        personas = [{"name": f"P{i}"} for i in range(SAMPLING_MAX_PERSONAS + 1)]
        raw = await run_panel(
            questions=[{"text": "q?"}],
            personas=personas,
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert f"{SAMPLING_MAX_PERSONAS} personas" in data["error"]
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_sampling_question_cap_enforced(self):
        from synth_panel.mcp.server import SAMPLING_MAX_QUESTIONS, run_panel

        ctx = _make_sampling_ctx(supports=True)
        questions = [{"text": f"Q{i}"} for i in range(SAMPLING_MAX_QUESTIONS + 1)]
        raw = await run_panel(
            questions=questions,
            personas=[{"name": "Alice"}],
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert f"{SAMPLING_MAX_QUESTIONS} questions" in data["error"]
        ctx.session.create_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_sampling_rejects_variants(self):
        from synth_panel.mcp.server import run_panel

        ctx = _make_sampling_ctx(supports=True)
        raw = await run_panel(
            questions=[{"text": "q?"}],
            personas=[{"name": "Alice"}],
            variants=3,
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert "variants" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_sampling_rejects_v3_branching_instrument(self):
        from synth_panel.mcp.server import run_panel

        ctx = _make_sampling_ctx(supports=True)
        instrument = {
            "version": 3,
            "rounds": [
                {
                    "name": "discovery",
                    "questions": [{"text": "What frustrates you?"}],
                    "route_when": [
                        {"if": {"field": "themes", "op": "contains", "value": "price"}, "goto": "probe"},
                        {"else": "__end__"},
                    ],
                },
                {
                    "name": "probe",
                    "questions": [{"text": "Dig deeper?"}],
                },
            ],
        }
        raw = await run_panel(
            instrument=instrument,
            personas=[{"name": "Alice"}],
            ctx=ctx,
        )
        data = json.loads(raw)
        assert "error" in data
        assert "branching" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_no_sampling_no_creds_returns_friendly_error(self):
        from synth_panel.mcp.server import run_panel

        ctx = _make_sampling_ctx(supports=False)
        raw = await run_panel(
            questions=[{"text": "q?"}],
            personas=[{"name": "Alice"}],
            ctx=ctx,
        )
        data = json.loads(raw)
        # The critical regression: no 'usage' KeyError, a structured
        # error payload instead.
        assert "error" in data
        assert "ANTHROPIC_API_KEY" in data["error"]
        ctx.session.create_message.assert_not_called()
