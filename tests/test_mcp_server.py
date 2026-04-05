"""Tests for synth_panel.mcp.server — tool/resource/prompt registration and data tools."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
            "run_panel",
            "run_quick_poll",
            "list_persona_packs",
            "get_persona_pack",
            "save_persona_pack",
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
