"""Tests for althing.mcp.server — tool/resource/prompt registration and data tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point data dir at temp for all tests.

    Also sets a fake ``ANTHROPIC_API_KEY`` so tests simulate the BYOK
    path by default — the sampling-mode tests clear these env vars
    explicitly to exercise the sampling branch.
    """
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")


from althing.mcp.server import MCP_DEFAULT_MODEL, mcp

# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------


class TestServerRegistration:
    """Verify that tools, resources, and prompts are registered."""

    def test_default_model_is_haiku(self):
        assert MCP_DEFAULT_MODEL == "haiku"

    def test_resolve_default_model_prefers_provider_with_credentials(self, monkeypatch):
        """sp-t6r: when the only key set is non-Anthropic, the MCP default
        must match that provider — otherwise we default to ``haiku``
        (Anthropic) and the LLM client rejects the run with a misleading
        missing-key error despite the user having valid credentials."""
        from althing.mcp.server import _resolve_mcp_default_model

        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

        # Preference chain — each provider should light up its own alias.
        cases = [
            ("ANTHROPIC_API_KEY", "haiku"),
            ("OPENAI_API_KEY", "gpt-4o-mini"),
            ("GEMINI_API_KEY", "gemini-2.5-flash"),
            ("GOOGLE_API_KEY", "gemini-2.5-flash"),
            ("XAI_API_KEY", "grok-3"),
            ("OPENROUTER_API_KEY", "openrouter/auto"),
        ]
        for env_var, expected_alias in cases:
            monkeypatch.setenv(env_var, "sk-x")
            try:
                assert _resolve_mcp_default_model() == expected_alias, env_var
            finally:
                monkeypatch.delenv(env_var, raising=False)

    def test_resolve_default_model_falls_back_to_haiku(self, monkeypatch):
        from althing.mcp.server import _resolve_mcp_default_model

        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_mcp_default_model() == MCP_DEFAULT_MODEL


class TestLargePanelFastModelSwap:
    """sy-2ag / GH#462: auto-pick a fast model when persona_count >= 10.

    Default ``openrouter/auto`` routes to slow workhorse models that
    stall 15+ min on 20-persona panels; pinning haiku-4-5 cuts the same
    run to 25-40s. The swap only fires for the auto-resolved default;
    explicit model arguments are honored verbatim.
    """

    @pytest.fixture(autouse=True)
    def _isolate_provider_creds(self, monkeypatch):
        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "XAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_openrouter_swapped_at_threshold(self, monkeypatch):
        from althing.mcp.server import (
            LARGE_PANEL_PERSONA_THRESHOLD,
            _resolve_mcp_default_model_for_panel,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        result = _resolve_mcp_default_model_for_panel(LARGE_PANEL_PERSONA_THRESHOLD)
        assert result == "openrouter/anthropic/claude-haiku-4.5"

    def test_openrouter_swapped_above_threshold(self, monkeypatch):
        from althing.mcp.server import _resolve_mcp_default_model_for_panel

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        assert _resolve_mcp_default_model_for_panel(20) == "openrouter/anthropic/claude-haiku-4.5"

    def test_openrouter_not_swapped_below_threshold(self, monkeypatch):
        from althing.mcp.server import (
            LARGE_PANEL_PERSONA_THRESHOLD,
            _resolve_mcp_default_model_for_panel,
        )

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        result = _resolve_mcp_default_model_for_panel(LARGE_PANEL_PERSONA_THRESHOLD - 1)
        assert result == "openrouter/auto"

    def test_anthropic_default_not_swapped(self, monkeypatch):
        """Haiku is already fast — no swap regardless of persona count."""
        from althing.mcp.server import _resolve_mcp_default_model_for_panel

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
        assert _resolve_mcp_default_model_for_panel(50) == "haiku"

    def test_openai_default_not_swapped(self, monkeypatch):
        from althing.mcp.server import _resolve_mcp_default_model_for_panel

        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        assert _resolve_mcp_default_model_for_panel(50) == "gpt-4o-mini"

    def test_gemini_default_not_swapped(self, monkeypatch):
        from althing.mcp.server import _resolve_mcp_default_model_for_panel

        monkeypatch.setenv("GEMINI_API_KEY", "sk-x")
        assert _resolve_mcp_default_model_for_panel(50) == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_run_panel_uses_fast_default_for_large_panel(self, monkeypatch):
        """End-to-end: 10 personas + openrouter env → fast model passed downstream."""
        from althing.mcp.server import mcp

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        # The outer fixture sets ANTHROPIC_API_KEY — clear so OPENROUTER wins.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        personas = [{"name": f"P{i}"} for i in range(10)]
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": personas,
                    "questions": [{"text": "Hello?"}],
                    "synthesis": False,
                    "decision_being_informed": "auto-fast-model smoke check",
                },
            )
        # _run_panel_async signature: (personas, questions, model, ctx, ...)
        assert mock_run.call_args[0][2] == "openrouter/anthropic/claude-haiku-4.5"

    @pytest.mark.asyncio
    async def test_run_panel_honors_explicit_openrouter_auto(self, monkeypatch):
        """An explicit ``openrouter/auto`` is honored — only defaults swap."""
        from althing.mcp.server import mcp

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        personas = [{"name": f"P{i}"} for i in range(15)]
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": personas,
                    "questions": [{"text": "Hello?"}],
                    "model": "openrouter/auto",
                    "synthesis": False,
                    "decision_being_informed": "auto-fast-model explicit check",
                },
            )
        assert mock_run.call_args[0][2] == "openrouter/auto"

    @pytest.mark.asyncio
    async def test_run_panel_no_swap_below_threshold(self, monkeypatch):
        """A 9-persona panel keeps the openrouter/auto default."""
        from althing.mcp.server import mcp

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-x")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        personas = [{"name": f"P{i}"} for i in range(9)]
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": personas,
                    "questions": [{"text": "Hello?"}],
                    "synthesis": False,
                    "decision_being_informed": "auto-fast-model below-threshold check",
                },
            )
        assert mock_run.call_args[0][2] == "openrouter/auto"

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "run_prompt",
            "run_panel",
            "run_quick_poll",
            "extend_panel",
            "list_persona_packs",
            "get_persona_pack",
            "save_persona_pack",
            "list_instrument_packs",
            "get_instrument_pack",
            "save_instrument_pack",
            "list_panel_results",
            "get_panel_result",
        }
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"

    @pytest.mark.asyncio
    async def test_prompts_registered(self):
        prompts = await mcp.list_prompts()
        prompt_names = {p.name for p in prompts}
        assert {"focus_group", "name_test", "concept_test"}.issubset(prompt_names)

    @pytest.mark.asyncio
    async def test_resources_registered(self):
        # Resource templates should be registered
        templates = await mcp.list_resource_templates()
        uris = {t.uriTemplate for t in templates}
        assert "persona-pack://{pack_id}" in uris
        assert "panel-result://{result_id}" in uris


# ---------------------------------------------------------------------------
# Data tools (no LLM calls)
# ---------------------------------------------------------------------------


class TestRunPrompt:
    """Test the run_prompt tool (mocks LLM)."""

    @pytest.mark.asyncio
    async def test_run_prompt_returns_response_and_cost(self):
        from althing.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-1",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hello back!")],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        with (
            patch("althing.mcp.server._shared_client", None),
            patch("althing.mcp.server.LLMClient") as MockClient,
        ):
            MockClient.return_value.send.return_value = mock_response
            result = await mcp.call_tool("run_prompt", {"prompt": "Say hello"})

        data = json.loads(result[0][0].text)
        assert data["response"] == "Hello back!"
        assert data["model"] == "claude-haiku-4-5-20251001"
        assert "cost" in data
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 5

    @pytest.mark.asyncio
    async def test_run_prompt_uses_default_model(self):
        from althing.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-2",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hi")],
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )
        with (
            patch("althing.mcp.server._shared_client", None),
            patch("althing.mcp.server.LLMClient") as MockClient,
        ):
            MockClient.return_value.send.return_value = mock_response
            await mcp.call_tool("run_prompt", {"prompt": "Hi"})
            # Verify the request used 'haiku' model (MCP default)
            call_args = MockClient.return_value.send.call_args
            assert call_args[0][0].model == "haiku"

    @pytest.mark.asyncio
    async def test_run_prompt_custom_model(self):
        from althing.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-3",
            model="claude-sonnet-4-6",
            content=[TextBlock(text="Hi")],
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )
        with (
            patch("althing.mcp.server._shared_client", None),
            patch("althing.mcp.server.LLMClient") as MockClient,
        ):
            MockClient.return_value.send.return_value = mock_response
            await mcp.call_tool(
                "run_prompt",
                {
                    "prompt": "Hi",
                    "model": "sonnet",
                },
            )
            call_args = MockClient.return_value.send.call_args
            assert call_args[0][0].model == "sonnet"


class TestDataTools:
    """Test tools that don't require LLM calls."""

    @pytest.mark.asyncio
    async def test_list_persona_packs_builtins_only(self):
        result = await mcp.call_tool("list_persona_packs", {})
        # call_tool returns a list of content blocks
        text = result[0][0].text
        data = json.loads(text)
        assert all(p["builtin"] for p in data)
        assert len(data) >= 1  # at least one bundled pack

    @pytest.mark.asyncio
    async def test_save_and_get_persona_pack(self):
        # Save
        save_result = await mcp.call_tool(
            "save_persona_pack",
            {
                "name": "Test Pack",
                "personas": [{"name": "Alice"}, {"name": "Bob"}],
                "pack_id": "test-1",
            },
        )
        saved = json.loads(save_result[0][0].text)
        assert saved["id"] == "test-1"
        assert saved["persona_count"] == 2

        # Get
        get_result = await mcp.call_tool("get_persona_pack", {"pack_id": "test-1"})
        pack = json.loads(get_result[0][0].text)
        assert pack["name"] == "Test Pack"
        assert len(pack["personas"]) == 2

        # List — saved pack should appear alongside builtins
        list_result = await mcp.call_tool("list_persona_packs", {})
        packs = json.loads(list_result[0][0].text)
        saved_ids = [p["id"] for p in packs if not p.get("builtin")]
        assert "test-1" in saved_ids

    @pytest.mark.asyncio
    async def test_list_panel_results_empty(self):
        result = await mcp.call_tool("list_panel_results", {})
        data = json.loads(result[0][0].text)
        assert data == []


# ---------------------------------------------------------------------------
# run_panel pack_id parameter
# ---------------------------------------------------------------------------


class TestRunPanelPackId:
    """Test run_panel's pack_id parameter for resolving saved persona packs."""

    def _save_pack(self, pack_id: str, personas: list[dict]) -> None:
        """Helper to save a persona pack directly."""
        from althing.mcp.data import save_persona_pack as _save

        _save("Test Pack", personas, pack_id)

    @pytest.mark.asyncio
    async def test_pack_id_only(self):
        """pack_id alone should resolve personas from storage."""
        self._save_pack("demo-pack", [{"name": "Alice"}, {"name": "Bob"}])
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "questions": [{"text": "Hello?"}],
                    "pack_id": "demo-pack",
                },
            )
            args = mock_run.call_args
            personas_used = args[0][0]
            assert len(personas_used) == 2
            assert personas_used[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_pack_id_merges_with_inline(self):
        """Inline personas come first, pack personas appended."""
        self._save_pack("merge-pack", [{"name": "Charlie"}])
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "pack_id": "merge-pack",
                },
            )
            personas_used = mock_run.call_args[0][0]
            assert len(personas_used) == 2
            assert personas_used[0]["name"] == "Alice"
            assert personas_used[1]["name"] == "Charlie"

    @pytest.mark.asyncio
    async def test_no_personas_no_pack_id_returns_error(self):
        """Neither personas nor pack_id should return an error."""
        result = await mcp.call_tool(
            "run_panel",
            {
                "questions": [{"text": "Hello?"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_pack_id_returns_typed_envelope(self):
        """Non-existent pack_id returns a typed INVALID_TOOL_ARG envelope
        (naming the field) rather than raising a raw FileNotFoundError."""
        result = await mcp.call_tool(
            "run_panel",
            {
                "questions": [{"text": "Hello?"}],
                "pack_id": "nonexistent",
                "decision_being_informed": "choosing which persona pack to reuse",
            },
        )
        data = json.loads(result[0][0].text)
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "pack_id"
        assert "nonexistent" in data["message"]
        assert data["retry_safe"] is False

    @pytest.mark.asyncio
    async def test_inline_personas_without_pack_id(self):
        """Traditional usage: inline personas only, no pack_id."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                },
            )
            personas_used = mock_run.call_args[0][0]
            assert len(personas_used) == 1
            assert personas_used[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# run_panel extract_schema parameter
# ---------------------------------------------------------------------------


class TestRunPanelExtractSchema:
    """Test run_panel's extract_schema parameter (string name and inline dict)."""

    @pytest.mark.asyncio
    async def test_inline_dict_schema_passed_through(self):
        """An inline dict extract_schema is forwarded as the resolved envelope.

        v1.0.3 P1: ``resolve_extract_schema`` now wraps every input into
        a ``{"schema": ..., "model": ...}`` envelope so downstream code
        can apply typed Pydantic validation when a model is available.
        For raw dicts the model is ``None``.
        """
        schema = {
            "type": "object",
            "properties": {"mood": {"type": "string"}},
            "required": ["mood"],
        }
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "How do you feel?"}],
                    "extract_schema": schema,
                },
            )
            kwargs = mock_run.call_args[1]
            assert kwargs["extract_schema"] == {"schema": schema, "model": None}

    @pytest.mark.asyncio
    async def test_string_name_resolves_to_registry_schema(self):
        """A string extract_schema resolves to the built-in registry entry."""
        from althing.mcp.server import EXTRACT_SCHEMA_REGISTRY

        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "How do you feel?"}],
                    "extract_schema": "sentiment",
                },
            )
            kwargs = mock_run.call_args[1]
            # Resolved envelope (v1.0.3 P1). ``sentiment`` has no Pydantic
            # mirror in MODEL_REGISTRY, so model is None.
            assert kwargs["extract_schema"] == {
                "schema": EXTRACT_SCHEMA_REGISTRY["sentiment"],
                "model": None,
            }

    @pytest.mark.asyncio
    async def test_unknown_name_returns_error(self):
        """An unrecognised schema name returns a JSON error (not an exception)."""
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "extract_schema": "nonexistent",
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "nonexistent" in data["error"]

    @pytest.mark.asyncio
    async def test_none_schema_passes_none(self):
        """Omitting extract_schema passes None to the async runner."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                },
            )
            kwargs = mock_run.call_args[1]
            assert kwargs["extract_schema"] is None


# ---------------------------------------------------------------------------
# run_panel models (ensemble) parameter
# ---------------------------------------------------------------------------


class TestRunPanelModels:
    """Test run_panel's models parameter for multi-model ensemble."""

    @pytest.mark.asyncio
    async def test_models_triggers_ensemble(self):
        """Providing models list triggers ensemble path via _run_ensemble_sync."""
        with patch("althing.mcp.server._run_ensemble_sync") as mock_ens:
            mock_ens.return_value = {
                "per_model_results": {},
                "cost_breakdown": {},
                "models": ["haiku", "sonnet"],
                "total_usage": {},
            }
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "models": ["haiku", "sonnet"],
                },
            )
            mock_ens.assert_called_once()
            data = json.loads(result[0][0].text)
            assert "per_model_results" in data
            assert data["models"] == ["haiku", "sonnet"]

    @pytest.mark.asyncio
    async def test_single_model_list_promotes_to_model(self):
        """hq-6j40: ``models=[X]`` is forgiving-promoted to ``model=X``.

        Without promotion the caller's model is silently dropped and the
        request runs against the MCP default — billing the wrong provider.
        """
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "models": ["openrouter/anthropic/claude-sonnet-4.5"],
                },
            )
            mock_run.assert_called_once()
            # Third positional arg of _run_panel_async is ``model``.
            args = mock_run.call_args.args
            assert args[2] == "openrouter/anthropic/claude-sonnet-4.5"

    @pytest.mark.asyncio
    async def test_empty_models_list_returns_typed_error(self):
        """hq-6j40: ``models=[]`` is a caller bug — surface it explicitly."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "models": [],
                },
            )
            mock_run.assert_not_called()
            data = json.loads(result[0][0].text)
            assert data["error_code"] == "INVALID_TOOL_ARG"
            assert data["field_path"] == "models"
            assert "at least one" in data["error"]

    @pytest.mark.asyncio
    async def test_model_and_models_mutually_exclusive(self):
        """hq-6j40: setting both ``model`` and ``models`` is ambiguous."""
        with (
            patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run,
            patch("althing.mcp.server._run_ensemble_sync") as mock_ens,
        ):
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "model": "haiku",
                    "models": ["sonnet", "haiku"],
                },
            )
            mock_run.assert_not_called()
            mock_ens.assert_not_called()
            data = json.loads(result[0][0].text)
            assert data["error_code"] == "INVALID_TOOL_ARG"
            assert "mutually exclusive" in data["error"]

    @pytest.mark.asyncio
    async def test_panel_timeout_returns_clear_envelope(self):
        """hq-6j40: ``asyncio.TimeoutError`` must surface as a clear envelope.

        Without an explicit catch, ``wait_for`` raises ``TimeoutError``
        with empty ``str(exc)`` and FastMCP relays
        ``"Error executing tool run_panel: "`` with no context.
        """
        import asyncio

        async def _boom(*_a, **_kw):
            raise asyncio.TimeoutError

        with patch("althing.mcp.server._run_panel_async", side_effect=_boom):
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "model": "haiku",
                },
            )
            data = json.loads(result[0][0].text)
            assert data["error_code"] == "PANEL_TIMEOUT"
            assert "timed out" in data["error"]
            assert data["timeout_seconds"] > 0
            assert data["error"]  # never empty


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    @pytest.mark.asyncio
    async def test_focus_group_prompt(self):
        result = await mcp.get_prompt("focus_group", {"topic": "remote work tools"})
        text = result.messages[0].content.text
        assert "remote work tools" in text
        assert "run_panel" in text

    @pytest.mark.asyncio
    async def test_name_test_prompt(self):
        result = await mcp.get_prompt(
            "name_test",
            {
                "names": "Acme, Zenith, Spark",
                "context": "a new task manager",
            },
        )
        text = result.messages[0].content.text
        assert "Acme, Zenith, Spark" in text
        assert "task manager" in text

    @pytest.mark.asyncio
    async def test_concept_test_prompt(self):
        result = await mcp.get_prompt(
            "concept_test",
            {
                "concept": "AI-powered code review",
                "target_audience": "senior developers",
            },
        )
        text = result.messages[0].content.text
        assert "AI-powered code review" in text
        assert "senior developers" in text


# ---------------------------------------------------------------------------
# Instrument-pack tools (F3-B)
# ---------------------------------------------------------------------------


class TestInstrumentPackTools:
    """The 3 new instrument-pack tools mirror the persona-pack equivalents."""

    @pytest.mark.asyncio
    async def test_list_builtins_only(self):
        result = await mcp.call_tool("list_instrument_packs", {})
        data = json.loads(result[0][0].text)
        assert all(p.get("source") == "bundled" for p in data)
        assert len(data) >= 1  # at least one bundled pack

    @pytest.mark.asyncio
    async def test_save_then_list_then_get(self):
        body = {
            "name": "Demo",
            "version": "1.0.0",
            "description": "demo pack",
            "author": "test",
            "instrument": {
                "version": 1,
                "questions": [{"text": "Hi?"}],
            },
        }
        save_res = await mcp.call_tool(
            "save_instrument_pack",
            {
                "name": "demo",
                "content": body,
            },
        )
        meta = json.loads(save_res[0][0].text)
        assert meta["id"] == "demo"
        assert meta["version"] == "1.0.0"

        list_res = await mcp.call_tool("list_instrument_packs", {})
        listed = json.loads(list_res[0][0].text)
        saved_ids = [p["id"] for p in listed if p.get("source") != "bundled"]
        assert "demo" in saved_ids

        get_res = await mcp.call_tool("get_instrument_pack", {"name": "demo"})
        loaded = json.loads(get_res[0][0].text)
        assert loaded["id"] == "demo"
        assert loaded["instrument"]["questions"][0]["text"] == "Hi?"

    @pytest.mark.asyncio
    async def test_save_rejects_invalid_instrument(self):
        from mcp.server.fastmcp.exceptions import ToolError

        bad = {"name": "Bad", "instrument": {"version": 1}}  # no questions/rounds
        with pytest.raises(ToolError):
            await mcp.call_tool(
                "save_instrument_pack",
                {
                    "name": "bad",
                    "content": bad,
                },
            )


# ---------------------------------------------------------------------------
# run_panel branching surface
# ---------------------------------------------------------------------------


class TestRunPanelInstrument:
    """run_panel accepts inline instrument and instrument_pack inputs."""

    @pytest.mark.asyncio
    async def test_inline_instrument_routes_to_multi_round(self):
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "A"}],
                    "instrument": {"version": 1, "questions": [{"text": "Hello?"}]},
                },
            )
            assert mock_run.called
            instrument_arg = mock_run.call_args[0][1]
            from althing.instrument import Instrument

            assert isinstance(instrument_arg, Instrument)
            assert len(instrument_arg.rounds) == 1

    @pytest.mark.asyncio
    async def test_instrument_pack_loads_then_routes(self):
        # Save a pack first via the data layer.
        from althing.mcp.data import save_instrument_pack as _save

        _save(
            "p1",
            {
                "name": "P1",
                "instrument": {"version": 1, "questions": [{"text": "Q?"}]},
            },
        )
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "A"}],
                    "instrument_pack": "p1",
                },
            )
            assert mock_run.called

    @pytest.mark.asyncio
    async def test_no_questions_no_instrument_returns_error(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "A"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_attachment_typo_field_rejected_at_wire_boundary(self):
        """hq-jviv: caller-supplied attachment typos must surface as a
        clean error through the MCP wire boundary, not silently propagate.

        The v1.0.4 hardening promised strict validation on the
        attachment surface; before the fix this only fired on the
        AttachmentRef CAS-ref shape (refs.json) — the *bank* attachment
        shape that callers actually pass through ``run_panel`` accepted
        unknown fields. This test pins the contract at the wire layer.
        """
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "T", "background": "test"}],
                "instrument": {
                    "version": 3,
                    "attachments": {
                        "img": {
                            "type": "image",
                            "media_type": "image/png",
                            "source": {"type": "base64", "data": "AAAA"},
                            "typo_field": "should fail strict",
                        }
                    },
                    "rounds": [
                        {
                            "name": "v",
                            "questions": [{"text": "Describe.", "attachments": ["img"]}],
                        }
                    ],
                },
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data, f"Expected error, got: {data}"
        assert "typo_field" in data["error"], f"Error must name the offending field; got: {data['error']!r}"


# ---------------------------------------------------------------------------
# extend_panel docstring contract
# ---------------------------------------------------------------------------


class TestExtendPanelContract:
    """extend_panel must document the 'ad-hoc round, not DAG re-entry' rule."""

    def test_docstring_spells_out_contract(self):
        from althing.mcp import server

        doc = server.extend_panel.__doc__ or ""
        # Both halves of the contract must be present.
        assert "ad-hoc" in doc
        assert "not" in doc and "DAG" in doc


# ---------------------------------------------------------------------------
# run_panel variants parameter
# ---------------------------------------------------------------------------


class TestRunPanelVariants:
    """Test run_panel's variants parameter for robustness analysis."""

    @pytest.mark.asyncio
    async def test_variants_param_accepted(self):
        """run_panel should accept variants param and forward to _run_panel_async."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "variants": 3,
                },
            )
            # variants should be forwarded as keyword argument
            kwargs = mock_run.call_args[1]
            assert kwargs.get("variants") == 3

    @pytest.mark.asyncio
    async def test_variants_zero_no_robustness(self):
        """variants=0 (default) should not include robustness data."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": [], "rounds": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                },
            )
            kwargs = mock_run.call_args[1]
            assert kwargs.get("variants", 0) == 0

    @pytest.mark.asyncio
    async def test_variants_invalid_returns_error(self):
        """variants > 20 should return an error."""
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "variants": 25,
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "variants" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_variants_negative_returns_error(self):
        """variants < 0 should return an error."""
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "variants": -1,
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data


# ---------------------------------------------------------------------------
# list_panel_results variant_count
# ---------------------------------------------------------------------------


class TestListPanelResultsVariantCount:
    """list_panel_results should include variant_count when present."""

    @pytest.mark.asyncio
    async def test_variant_count_in_listing(self):
        """Results saved with variant_count should include it in listing."""
        from althing.mcp.data import save_panel_result

        save_panel_result(
            results=[{"persona": "A", "responses": [], "usage": {}, "cost": "$0", "error": None}],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0.00",
            persona_count=1,
            question_count=1,
            variant_count=5,
        )
        result = await mcp.call_tool("list_panel_results", {})
        data = json.loads(result[0][0].text)
        assert len(data) == 1
        assert data[0]["variant_count"] == 5

    @pytest.mark.asyncio
    async def test_no_variant_count_when_zero(self):
        """Results without variants should not include variant_count."""
        from althing.mcp.data import save_panel_result

        save_panel_result(
            results=[{"persona": "A", "responses": [], "usage": {}, "cost": "$0", "error": None}],
            model="haiku",
            total_usage={"input_tokens": 0, "output_tokens": 0},
            total_cost="$0.00",
            persona_count=1,
            question_count=1,
        )
        result = await mcp.call_tool("list_panel_results", {})
        data = json.loads(result[0][0].text)
        assert len(data) == 1
        assert "variant_count" not in data[0]


# ---------------------------------------------------------------------------
# _compute_variant_data
# ---------------------------------------------------------------------------


class TestComputeVariantData:
    """Test the robustness computation from variant results."""

    def test_compute_variant_data_basic(self):
        from althing.mcp.server import _compute_variant_data

        result_dicts = [
            # Base persona
            {
                "persona": "Alice",
                "responses": [{"question": "Q1", "response": "agree", "error": False}],
                "usage": {},
                "cost": "$0",
                "error": None,
            },
            # Variant
            {
                "persona": "Alice (v0)",
                "responses": [{"question": "Q1", "response": "agree", "error": False}],
                "usage": {},
                "cost": "$0",
                "error": None,
            },
            {
                "persona": "Alice (v1)",
                "responses": [{"question": "Q1", "response": "disagree", "error": False}],
                "usage": {},
                "cost": "$0",
                "error": None,
            },
        ]
        variant_names = {"Alice (v0)", "Alice (v1)"}
        variant_mapping = {"Alice (v0)": "Alice", "Alice (v1)": "Alice"}
        questions = [{"text": "Do you agree?"}]

        data = _compute_variant_data(result_dicts, variant_names, variant_mapping, 2, questions)

        assert data["variant_count"] == 2
        assert len(data["robustness_scores"]) == 1
        assert len(data["per_persona_robustness"]) == 1
        assert data["per_persona_robustness"][0]["persona"] == "Alice"
        assert data["per_persona_robustness"][0]["k_variants"] == 2
        # One variant agreed, one disagreed -> 0.5 robustness
        assert data["per_persona_robustness"][0]["robustness"] == 0.5

    def test_compute_variant_data_no_variants(self):
        from althing.mcp.server import _compute_variant_data

        result_dicts = [
            {
                "persona": "Alice",
                "responses": [{"question": "Q1", "response": "agree", "error": False}],
                "usage": {},
                "cost": "$0",
                "error": None,
            },
        ]
        data = _compute_variant_data(result_dicts, set(), {}, 0, [{"text": "Q1"}])

        assert data["variant_count"] == 0
        assert data["robustness_scores"] == []
        assert data["per_persona_robustness"] == []


# ---------------------------------------------------------------------------
# Weighted model spec rejection at the MCP boundary (sp-2rj8)
# ---------------------------------------------------------------------------


class TestWeightedModelSpecRejection:
    """MCP rejects CLI-style ``name:weight`` entries with a clear error.

    The CLI's ``--models haiku:0.25,gpt-4o-mini:0.25`` grammar is not
    parsed at the MCP boundary — each string was being treated as a
    raw model name, so ``"haiku:0.25"`` routed to a nonexistent model
    and produced a silent empty panel. These tests pin the boundary
    behavior so future regressions fail loudly.
    """

    def test_looks_like_weighted_model_spec_detects_weights(self):
        from althing.mcp.server import _looks_like_weighted_model_spec

        assert _looks_like_weighted_model_spec("haiku:0.25") is True
        assert _looks_like_weighted_model_spec("gpt-4o-mini:0.5") is True
        assert _looks_like_weighted_model_spec("claude-sonnet-4-6:1") is True

    def test_looks_like_weighted_model_spec_allows_real_identifiers(self):
        from althing.mcp.server import _looks_like_weighted_model_spec

        # Local model prefixes are preserved.
        assert _looks_like_weighted_model_spec("ollama:llama3") is False
        assert _looks_like_weighted_model_spec("ollama:llama3:8b") is False
        assert _looks_like_weighted_model_spec("local:phi3") is False
        # OpenRouter non-numeric tail suffixes.
        assert _looks_like_weighted_model_spec("mistralai/mistral-nemo:free") is False
        assert _looks_like_weighted_model_spec("anthropic/claude-3.5-sonnet:beta") is False
        # Plain aliases.
        assert _looks_like_weighted_model_spec("haiku") is False
        assert _looks_like_weighted_model_spec("gpt-4o-mini") is False
        assert _looks_like_weighted_model_spec("anthropic/claude-3-5-sonnet-20241022") is False

    @pytest.mark.asyncio
    async def test_run_panel_rejects_weighted_models_list(self):
        """The exact sp-2rj8 repro: weighted ``models`` list must error out."""
        with patch("althing.mcp.server._run_ensemble_sync") as mock_ens:
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "models": [
                        "haiku:0.25",
                        "gpt-4o-mini:0.25",
                        "gemini-flash-lite:0.25",
                        "qwen3-plus:0.25",
                    ],
                },
            )
            mock_ens.assert_not_called()
            data = json.loads(result[0][0].text)
            assert "error" in data
            msg = data["error"]
            assert "haiku:0.25" in msg
            assert "gpt-4o-mini:0.25" in msg
            assert "not supported via MCP" in msg

    @pytest.mark.asyncio
    async def test_run_panel_accepts_plain_alias_ensemble(self):
        """Plain alias lists still reach the ensemble runner unchanged."""
        with patch("althing.mcp.server._run_ensemble_sync") as mock_ens:
            mock_ens.return_value = {
                "per_model_results": {},
                "cost_breakdown": {},
                "models": ["haiku", "gpt-4o-mini"],
                "total_usage": {},
            }
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "models": ["haiku", "gpt-4o-mini"],
                },
            )
            mock_ens.assert_called_once()
            data = json.loads(result[0][0].text)
            assert "error" not in data

    @pytest.mark.asyncio
    async def test_run_panel_rejects_weighted_single_model(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "model": "haiku:0.5",
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "haiku:0.5" in data["error"]

    @pytest.mark.asyncio
    async def test_run_panel_rejects_weighted_synthesis_model(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "synthesis_model": "sonnet:1.0",
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "sonnet:1.0" in data["error"]

    @pytest.mark.asyncio
    async def test_run_panel_rejects_weighted_persona_models(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "persona_models": {"Alice": "haiku:0.25"},
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "haiku:0.25" in data["error"]

    @pytest.mark.asyncio
    async def test_run_panel_accepts_ollama_prefix_model(self):
        """``ollama:llama3`` is a legitimate model id and must not be rejected."""
        with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            result = await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                    "model": "ollama:llama3",
                },
            )
            mock_run.assert_called_once()
            data = json.loads(result[0][0].text)
            assert "error" not in data

    @pytest.mark.asyncio
    async def test_run_prompt_rejects_weighted_model(self):
        result = await mcp.call_tool(
            "run_prompt",
            {"prompt": "hello", "model": "haiku:0.5"},
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "haiku:0.5" in data["error"]

    @pytest.mark.asyncio
    async def test_run_quick_poll_rejects_weighted_model(self):
        result = await mcp.call_tool(
            "run_quick_poll",
            {"question": "hello?", "model": "haiku:0.5"},
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "haiku:0.5" in data["error"]

    @pytest.mark.asyncio
    async def test_extend_panel_rejects_weighted_model(self):
        result = await mcp.call_tool(
            "extend_panel",
            {
                "result_id": "nonexistent",
                "questions": [{"text": "follow-up?"}],
                "model": "haiku:0.5",
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "haiku:0.5" in data["error"]


# ---------------------------------------------------------------------------
# extend_panel synthesis-failure loudness (sp-0ozi)
# ---------------------------------------------------------------------------


class TestExtendPanelSynthesisLoudness:
    """sp-0ozi: synthesis exceptions must surface a structured
    ``synthesis_error`` in the MCP response envelope, not silently null
    ``synth``. The failure stays non-fatal — panelist results are still
    returned."""

    @pytest.mark.asyncio
    async def test_synthesis_exception_populates_synthesis_error(self, monkeypatch):
        from althing.cost import TokenUsage
        from althing.mcp import server as _server
        from althing.orchestrator import PanelistResult

        fake_existing = {"rounds": [], "path": [], "question_count": 0}
        fake_sessions = {"Alice": object()}
        fake_panelist = PanelistResult(
            # Canonical response shape uses ``response`` (matches the real
            # orchestrator + every other stub); ``extend_panel`` now runs
            # detect_total_failure over these rows, which treats a row with
            # no clean ``response`` as a failed panelist.
            persona_name="Alice",
            responses=[{"question": "q?", "response": "a"}],
            usage=TokenUsage(),
            model="haiku",
        )

        class BoomSynth(Exception):
            pass

        def _raise(*args, **kwargs):
            raise BoomSynth("upstream 500")

        with (
            patch("althing.mcp.server._data_get_panel_result", return_value=fake_existing),
            patch("althing.mcp.server.load_panel_sessions", return_value=fake_sessions),
            patch(
                "althing.mcp.server.run_panel_parallel",
                return_value=([fake_panelist], {}, fake_sessions),
            ),
            patch("althing.mcp.server.synthesize_panel", side_effect=_raise),
            patch("althing.mcp.server.update_panel_result"),
            patch("althing.mcp.server._get_shared_client", return_value=object()),
        ):
            # Call the tool function directly (ctx=None). mcp.call_tool
            # injects a request-scoped Context that fails outside a request.
            raw = await _server.extend_panel(
                result_id="r-test",
                questions=[{"text": "follow-up?"}],
                model="haiku",
                synthesis=True,
                ctx=None,
            )

        data = json.loads(raw)
        # Panelist results still come through — non-fatal semantic preserved.
        assert data.get("results"), "panelist results must still be returned"
        # synth field is null because synthesis threw.
        assert data.get("synthesis") is None
        # Top-level structured synthesis_error is the new contract.
        err = data.get("synthesis_error")
        assert isinstance(err, dict), "synthesis_error must be a dict on the envelope"
        assert err.get("error_type") == "synthesis_api_error"
        assert "upstream 500" in err.get("message", "")


# ---------------------------------------------------------------------------
# v3 multi-round panel runs through the MCP wire boundary (hq-83ye)
#
# Pins the contract that ``mcp.call_tool("run_panel", ...)`` with a v3
# instrument runs *every* declared round and serializes the per-round
# results into the response envelope. Until v1.0.5 (hq-fjdx) the
# orchestrator silently terminated linear v3 runs after round 1 and the
# synthesis layer masked it because a single broad LLM response covers
# many topics holistically. This shipped through every v1.0.x release
# because no CI test exercised the *MCP-routed* multi-round path.
#
# The stubs below replace the LLM-touching seam — ``run_panel_parallel``
# and ``synthesize_panel`` — so the orchestrator + MCP wire layers run
# end-to-end on synthetic data. Anything between the wire boundary and
# the LLM provider is real code.
# ---------------------------------------------------------------------------


def _stub_panelist_results_factory(default_themes: list[str] | None = None):
    """Return a stub for ``althing.orchestrator.run_panel_parallel``.

    Each call produces one ``PanelistResult`` per persona with one
    response per question and a small non-zero usage so the cost rollup
    has something to aggregate across rounds.
    """
    from althing.cost import TokenUsage
    from althing.orchestrator import PanelistResult

    def _fake_run_panel_parallel(
        client,
        personas,
        questions,
        model,
        system_prompt_fn,
        question_prompt_fn,
        max_workers=None,
        response_schema=None,
        sessions=None,
        extract_schema=None,
        temperature=None,
        top_p=None,
        seed=None,
        persona_models=None,
        panel_shared_attachments=None,
    ):
        results = [
            PanelistResult(
                persona_name=p.get("name", "anon"),
                responses=[{"question": q.get("text", ""), "response": "ok"} for q in questions],
                usage=TokenUsage(input_tokens=5, output_tokens=3),
                model=(persona_models or {}).get(p.get("name", "anon"), model),
            )
            for p in personas
        ]
        new_sessions = dict(sessions or {})
        for p in personas:
            new_sessions.setdefault(p.get("name", "anon"), object())
        return results, {}, new_sessions

    return _fake_run_panel_parallel


def _stub_synthesize_factory(themes: list[str] | None = None):
    """Return a stub for the ``synthesize_panel`` import in ``_runners``.

    The orchestrator's router consumes ``themes`` from the synthesis
    output, so each test customizes the themes to drive a specific path.
    """
    from althing.cost import TokenUsage as CostTokenUsage
    from althing.synthesis import SynthesisResult

    themes = list(themes or ["pricing pain"])

    def _fake_synthesize_panel(
        client,
        panelist_results,
        questions,
        *,
        model=None,
        panelist_model=None,
        custom_prompt=None,
        temperature=None,
        seed=None,
        **_kwargs,
    ):
        return SynthesisResult(
            summary="stub summary",
            themes=list(themes),
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="stub recommendation",
            usage=CostTokenUsage(input_tokens=2, output_tokens=1),
            model=model or panelist_model or "stub-model",
        )

    return _fake_synthesize_panel


class _StubMcpContext:
    """Minimal Context double for direct invocation of MCP tools.

    ``mcp.call_tool`` only injects a real Context inside an active
    request, so direct unit tests against the tool function pass this
    stub instead. The orchestrator only uses ``report_progress``; other
    Context methods are intentionally unimplemented so missed coverage
    fails loudly.
    """

    async def report_progress(self, *_args, **_kwargs):
        return None


class TestRunPanelMultiRoundV3:
    """End-to-end MCP wire tests for v3 multi-round panel runs (hq-83ye).

    These tests do not mock ``_run_panel_async_instrument`` — the whole
    point of this gap is that prior tests stopped at the wire layer and
    never exercised the orchestrator's per-round loop through the MCP
    tool. The tool function is invoked directly with a stub Context
    because ``mcp.call_tool`` requires an active request to inject one.
    Everything from the tool entry point down to (but excluding) the
    LLM call is real code here.
    """

    PERSONAS = [{"name": "Alice", "background": "x"}, {"name": "Bob", "background": "y"}]

    V3_BRANCHING = {
        "version": 3,
        "rounds": [
            {
                "name": "intro",
                "questions": [{"text": "What hurts?"}],
                "route_when": [
                    {
                        "if": {"field": "themes", "op": "contains", "value": "pricing"},
                        "goto": "probe_pricing",
                    },
                    {"else": "wrap_up"},
                ],
            },
            {
                "name": "probe_pricing",
                "questions": [{"text": "What would feel fair to pay?"}],
            },
            {"name": "wrap_up", "questions": [{"text": "Final thoughts?"}]},
        ],
    }

    V3_LINEAR = {
        "version": 3,
        "rounds": [
            {"name": "first_impressions", "questions": [{"text": "Q1"}]},
            {"name": "brand_fit", "questions": [{"text": "Q2"}]},
            {"name": "ia_hierarchy", "questions": [{"text": "Q3"}]},
        ],
    }

    @pytest.mark.asyncio
    async def test_branching_v3_runs_three_rounds_through_mcp_wire(self):
        """3-round v3 with route_when: MCP envelope reflects every round.

        Pins the per-round shape callers depend on:
        ``rounds`` length matches the path, ``path`` records the routing
        decision for each non-terminal round, ``terminal_round`` is the
        last executed round (not the syntactic last round in the file),
        and ``question_count`` is the sum across executed rounds.
        """
        with (
            patch(
                "althing.orchestrator.run_panel_parallel",
                side_effect=_stub_panelist_results_factory(),
            ),
            patch(
                "althing._runners.synthesize_panel",
                side_effect=_stub_synthesize_factory(themes=["pricing pain"]),
            ),
            patch("althing.mcp.server._shared_client", None),
        ):
            from althing.mcp import server as _server

            raw = await _server.run_panel(
                personas=self.PERSONAS,
                instrument=self.V3_BRANCHING,
                model="haiku",
                ctx=_StubMcpContext(),
            )

        data = json.loads(raw)
        # Path traversed all three rounds in the expected order: intro
        # routes to probe_pricing on the 'pricing' theme, then probe_pricing
        # falls through positionally to wrap_up.
        assert [p["round"] for p in data["path"]] == ["intro", "probe_pricing", "wrap_up"]
        assert data["path"][0]["next"] == "probe_pricing"
        assert "pricing" in data["path"][0]["branch"]
        assert data["path"][-1]["next"] == "__end__"

        # rounds[] mirrors the path 1:1 — three per-round payloads, each
        # with a synthesis dict and a usage breakdown.
        round_names = [r["name"] for r in data["rounds"]]
        assert round_names == ["intro", "probe_pricing", "wrap_up"]
        for rd in data["rounds"]:
            assert rd["synthesis"] is not None
            assert rd["usage"]["input_tokens"] >= 0

        # question_count sums the per-round questions actually executed.
        assert data["question_count"] == 3
        assert data["terminal_round"] == "wrap_up"

        # Cost rollup reflects panelist usage across all three rounds.
        # The stubbed PanelistResult contributes 5 input + 3 output tokens
        # per persona per round; with 2 personas × 3 rounds + per-round
        # synthesis (2 in / 1 out × 3) + final synthesis (2 in / 1 out)
        # the floor for cumulative input tokens is well above zero.
        assert data["total_usage"]["input_tokens"] > 0
        assert data["total_usage"]["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_linear_v3_runs_all_rounds_through_mcp_wire(self):
        """Linear v3 (no route_when, no depends_on): every round executes.

        Regression for hq-fjdx surfaced through the MCP boundary. Before
        the orchestrator fix, the MCP wrapper happily returned a single-
        round payload for a 3-round linear instrument and the synthesis
        layer hid the truncation. This test would have caught the bug.
        """
        with (
            patch(
                "althing.orchestrator.run_panel_parallel",
                side_effect=_stub_panelist_results_factory(),
            ),
            patch(
                "althing._runners.synthesize_panel",
                side_effect=_stub_synthesize_factory(),
            ),
            patch("althing.mcp.server._shared_client", None),
        ):
            from althing.mcp import server as _server

            raw = await _server.run_panel(
                personas=self.PERSONAS,
                instrument=self.V3_LINEAR,
                model="haiku",
                ctx=_StubMcpContext(),
            )

        data = json.loads(raw)
        executed = [r["name"] for r in data["rounds"]]
        assert executed == ["first_impressions", "brand_fit", "ia_hierarchy"]

        # Linear path: each non-terminal next is the positional successor,
        # last entry routes to the __end__ sentinel, branch tag is "linear".
        nexts = [p["next"] for p in data["path"]]
        assert nexts == ["brand_fit", "ia_hierarchy", "__end__"]
        assert all(p["branch"] == "linear" for p in data["path"])
        assert data["terminal_round"] == "ia_hierarchy"
        assert data["question_count"] == 3

    @pytest.mark.asyncio
    async def test_v3_round_with_zero_panelist_responses_pins_envelope_shape(self):
        """Pin: a round that returns zero panelist responses still emits
        a well-formed envelope and the run continues to the next round.

        Today's behavior (pre-fix for any future zero-panelist hardening):
        the round records an empty ``results`` list, synthesis runs on
        the empty list, and the path log advances normally. If we ever
        decide to raise on this case, this test's assertion needs to
        flip — but until then, callers can rely on the envelope shape.

        ``detail="full"`` is requested explicitly because ``run_panel``
        now defaults to the compact ``detail="summary"`` envelope, which
        drops ``rounds[].results`` (retrievable via ``get_panel_result``).
        This test pins the per-round transcript shape, so it opts into full.
        """
        from althing.cost import TokenUsage

        def _empty_then_normal(
            client,
            personas,
            questions,
            model,
            system_prompt_fn,
            question_prompt_fn,
            max_workers=None,
            response_schema=None,
            sessions=None,
            extract_schema=None,
            temperature=None,
            top_p=None,
            seed=None,
            persona_models=None,
            panel_shared_attachments=None,
        ):
            from althing.orchestrator import PanelistResult

            # First round: zero panelist responses (the failure surface).
            # Subsequent rounds: normal stub results so the test pins
            # behavior across the boundary, not just at the empty round.
            if not getattr(_empty_then_normal, "_called", False):
                _empty_then_normal._called = True
                return [], {}, dict(sessions or {})
            results = [
                PanelistResult(
                    persona_name=p.get("name", "anon"),
                    responses=[{"question": q.get("text", ""), "response": "ok"} for q in questions],
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model=model,
                )
                for p in personas
            ]
            return results, {}, dict(sessions or {})

        with (
            patch(
                "althing.orchestrator.run_panel_parallel",
                side_effect=_empty_then_normal,
            ),
            patch(
                "althing._runners.synthesize_panel",
                side_effect=_stub_synthesize_factory(),
            ),
            patch("althing.mcp.server._shared_client", None),
        ):
            from althing.mcp import server as _server

            raw = await _server.run_panel(
                personas=self.PERSONAS,
                instrument=self.V3_LINEAR,
                model="haiku",
                detail="full",
                ctx=_StubMcpContext(),
            )

        data = json.loads(raw)
        # Envelope is well-formed even with an empty first round.
        assert "rounds" in data and "path" in data
        # The empty first round serialises with results=[].
        first_round = data["rounds"][0]
        assert first_round["name"] == "first_impressions"
        assert first_round["results"] == []
        # The run did not abort — subsequent rounds still ran.
        assert len(data["rounds"]) >= 2
        assert data["rounds"][1]["results"], "follow-on rounds still produce panelist results"
