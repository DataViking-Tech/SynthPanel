"""Tests for synth_panel.mcp.server — tool/resource/prompt registration and data tools."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, ANY

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point data dir at temp for all tests."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))


from synth_panel.mcp.server import mcp, MCP_DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Server registration
# ---------------------------------------------------------------------------

class TestServerRegistration:
    """Verify that tools, resources, and prompts are registered."""

    def test_default_model_is_haiku(self):
        assert MCP_DEFAULT_MODEL == "haiku"

    @pytest.mark.asyncio
    async def test_tools_registered(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "run_prompt",
            "run_panel",
            "run_quick_poll",
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
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-1",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hello back!")],
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        with patch("synth_panel.mcp.server.LLMClient") as MockClient:
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
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-2",
            model="claude-haiku-4-5-20251001",
            content=[TextBlock(text="Hi")],
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )
        with patch("synth_panel.mcp.server.LLMClient") as MockClient:
            MockClient.return_value.send.return_value = mock_response
            result = await mcp.call_tool("run_prompt", {"prompt": "Hi"})
            # Verify the request used 'haiku' model (MCP default)
            call_args = MockClient.return_value.send.call_args
            assert call_args[0][0].model == "haiku"

    @pytest.mark.asyncio
    async def test_run_prompt_custom_model(self):
        from synth_panel.llm.models import CompletionResponse, TextBlock, TokenUsage

        mock_response = CompletionResponse(
            id="resp-3",
            model="claude-sonnet-4-6",
            content=[TextBlock(text="Hi")],
            usage=TokenUsage(input_tokens=5, output_tokens=2),
        )
        with patch("synth_panel.mcp.server.LLMClient") as MockClient:
            MockClient.return_value.send.return_value = mock_response
            result = await mcp.call_tool("run_prompt", {
                "prompt": "Hi",
                "model": "sonnet",
            })
            call_args = MockClient.return_value.send.call_args
            assert call_args[0][0].model == "sonnet"


class TestDataTools:
    """Test tools that don't require LLM calls."""

    @pytest.mark.asyncio
    async def test_list_persona_packs_empty(self):
        result = await mcp.call_tool("list_persona_packs", {})
        # call_tool returns a list of content blocks
        text = result[0][0].text
        data = json.loads(text)
        assert data == []

    @pytest.mark.asyncio
    async def test_save_and_get_persona_pack(self):
        # Save
        save_result = await mcp.call_tool("save_persona_pack", {
            "name": "Test Pack",
            "personas": [{"name": "Alice"}, {"name": "Bob"}],
            "pack_id": "test-1",
        })
        saved = json.loads(save_result[0][0].text)
        assert saved["id"] == "test-1"
        assert saved["persona_count"] == 2

        # Get
        get_result = await mcp.call_tool("get_persona_pack", {"pack_id": "test-1"})
        pack = json.loads(get_result[0][0].text)
        assert pack["name"] == "Test Pack"
        assert len(pack["personas"]) == 2

        # List
        list_result = await mcp.call_tool("list_persona_packs", {})
        packs = json.loads(list_result[0][0].text)
        assert len(packs) == 1

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
        from synth_panel.mcp.data import save_persona_pack as _save
        _save("Test Pack", personas, pack_id)

    @pytest.mark.asyncio
    async def test_pack_id_only(self):
        """pack_id alone should resolve personas from storage."""
        self._save_pack("demo-pack", [{"name": "Alice"}, {"name": "Bob"}])
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool("run_panel", {
                "questions": [{"text": "Hello?"}],
                "pack_id": "demo-pack",
            })
            args = mock_run.call_args
            personas_used = args[0][0]
            assert len(personas_used) == 2
            assert personas_used[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_pack_id_merges_with_inline(self):
        """Inline personas come first, pack personas appended."""
        self._save_pack("merge-pack", [{"name": "Charlie"}])
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool("run_panel", {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "pack_id": "merge-pack",
            })
            personas_used = mock_run.call_args[0][0]
            assert len(personas_used) == 2
            assert personas_used[0]["name"] == "Alice"
            assert personas_used[1]["name"] == "Charlie"

    @pytest.mark.asyncio
    async def test_no_personas_no_pack_id_returns_error(self):
        """Neither personas nor pack_id should return an error."""
        result = await mcp.call_tool("run_panel", {
            "questions": [{"text": "Hello?"}],
        })
        data = json.loads(result[0][0].text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_pack_id_raises(self):
        """Non-existent pack_id should raise ToolError wrapping FileNotFoundError."""
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="Persona pack not found"):
            await mcp.call_tool("run_panel", {
                "questions": [{"text": "Hello?"}],
                "pack_id": "nonexistent",
            })

    @pytest.mark.asyncio
    async def test_inline_personas_without_pack_id(self):
        """Traditional usage: inline personas only, no pack_id."""
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool("run_panel", {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
            })
            personas_used = mock_run.call_args[0][0]
            assert len(personas_used) == 1
            assert personas_used[0]["name"] == "Alice"


# ---------------------------------------------------------------------------
# Instrument pack tools (sp-yiz)
# ---------------------------------------------------------------------------

class TestInstrumentPackTools:
    """Round-trip the three new instrument-pack MCP tools."""

    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = await mcp.call_tool("list_instrument_packs", {})
        data = json.loads(result[0][0].text)
        assert data == []

    @pytest.mark.asyncio
    async def test_save_get_list_round_trip(self):
        body = {
            "name": "Pricing Discovery",
            "version": "1.0",
            "description": "Branching pricing probe",
            "author": "test",
            "questions": [{"text": "What would you pay?"}],
        }
        save_result = await mcp.call_tool("save_instrument_pack", {
            "name": "pricing-discovery",
            "content": body,
        })
        saved = json.loads(save_result[0][0].text)
        assert saved["id"] == "pricing-discovery"
        assert saved["version"] == "1.0"
        assert saved["type"] == "instrument"

        get_result = await mcp.call_tool("get_instrument_pack", {
            "name": "pricing-discovery",
        })
        pack = json.loads(get_result[0][0].text)
        assert pack["name"] == "Pricing Discovery"
        assert pack["questions"][0]["text"] == "What would you pay?"

        list_result = await mcp.call_tool("list_instrument_packs", {})
        listed = json.loads(list_result[0][0].text)
        assert len(listed) == 1
        assert listed[0]["id"] == "pricing-discovery"

    @pytest.mark.asyncio
    async def test_get_missing_raises(self):
        with pytest.raises(Exception):
            await mcp.call_tool("get_instrument_pack", {"name": "nope"})


# ---------------------------------------------------------------------------
# run_panel response shape (sp-yiz: path + warnings keys)
# ---------------------------------------------------------------------------

class TestRunPanelResponseShape:
    """run_panel return dict gains path + warnings keys (F3-B contract)."""

    @pytest.mark.asyncio
    async def test_response_has_path_and_warnings(self):
        from synth_panel.mcp import server as srv
        from synth_panel.cost import TokenUsage as CTU, ZERO_USAGE

        async def fake_run_panel_async(personas, questions, model, ctx, response_schema, **kw):
            return {
                "result_id": "fake",
                "model": model,
                "persona_count": len(personas),
                "question_count": len(questions),
                "panelist_cost": "$0.00",
                "synthesis": None,
                "total_cost": "$0.00",
                "total_usage": ZERO_USAGE.to_dict(),
                "results": [],
                "path": [{"round": "default", "branch": None, "next": "__end__"}],
                "warnings": [],
            }

        with patch.object(srv, "_run_panel_async", side_effect=fake_run_panel_async):
            result = await mcp.call_tool("run_panel", {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hi?"}],
            })
        data = json.loads(result[0][0].text)
        assert "path" in data
        assert "warnings" in data
        assert isinstance(data["path"], list)
        assert isinstance(data["warnings"], list)
        assert len(data["path"]) >= 1


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
        result = await mcp.get_prompt("name_test", {
            "names": "Acme, Zenith, Spark",
            "context": "a new task manager",
        })
        text = result.messages[0].content.text
        assert "Acme, Zenith, Spark" in text
        assert "task manager" in text

    @pytest.mark.asyncio
    async def test_concept_test_prompt(self):
        result = await mcp.get_prompt("concept_test", {
            "concept": "AI-powered code review",
            "target_audience": "senior developers",
        })
        text = result.messages[0].content.text
        assert "AI-powered code review" in text
        assert "senior developers" in text
