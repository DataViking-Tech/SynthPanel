"""Tests for synth_panel.mcp.server — tool/resource/prompt registration and data tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point data dir at temp for all tests."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))


from synth_panel.mcp.server import MCP_DEFAULT_MODEL, mcp

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
            await mcp.call_tool("run_prompt", {"prompt": "Hi"})
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
        from synth_panel.mcp.data import save_persona_pack as _save

        _save("Test Pack", personas, pack_id)

    @pytest.mark.asyncio
    async def test_pack_id_only(self):
        """pack_id alone should resolve personas from storage."""
        self._save_pack("demo-pack", [{"name": "Alice"}, {"name": "Bob"}])
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
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
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
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
    async def test_invalid_pack_id_raises(self):
        """Non-existent pack_id should raise ToolError wrapping FileNotFoundError."""
        from mcp.server.fastmcp.exceptions import ToolError

        with pytest.raises(ToolError, match="Persona pack not found"):
            await mcp.call_tool(
                "run_panel",
                {
                    "questions": [{"text": "Hello?"}],
                    "pack_id": "nonexistent",
                },
            )

    @pytest.mark.asyncio
    async def test_inline_personas_without_pack_id(self):
        """Traditional usage: inline personas only, no pack_id."""
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
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
            "synth_panel.mcp.server._run_panel_async_instrument",
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
            from synth_panel.instrument import Instrument

            assert isinstance(instrument_arg, Instrument)
            assert len(instrument_arg.rounds) == 1

    @pytest.mark.asyncio
    async def test_instrument_pack_loads_then_routes(self):
        # Save a pack first via the data layer.
        from synth_panel.mcp.data import save_instrument_pack as _save

        _save(
            "p1",
            {
                "name": "P1",
                "instrument": {"version": 1, "questions": [{"text": "Q?"}]},
            },
        )
        with patch(
            "synth_panel.mcp.server._run_panel_async_instrument",
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


# ---------------------------------------------------------------------------
# extend_panel docstring contract
# ---------------------------------------------------------------------------


class TestExtendPanelContract:
    """extend_panel must document the 'ad-hoc round, not DAG re-entry' rule."""

    def test_docstring_spells_out_contract(self):
        from synth_panel.mcp import server

        doc = server.extend_panel.__doc__ or ""
        # Both halves of the contract must be present.
        assert "ad-hoc" in doc
        assert "not" in doc and "DAG" in doc


# ---------------------------------------------------------------------------
# Input validation and rate limits on run_panel
# ---------------------------------------------------------------------------


class TestRunPanelValidation:
    """run_panel rejects invalid personas, questions, and oversized inputs."""

    @pytest.mark.asyncio
    async def test_persona_missing_name(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"age": 30}],
                "questions": [{"text": "Hello?"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "name" in data["error"]

    @pytest.mark.asyncio
    async def test_persona_not_a_dict(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": ["just a string"],
                "questions": [{"text": "Hello?"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "dict" in data["error"]

    @pytest.mark.asyncio
    async def test_question_missing_text(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"prompt": "no text key"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "text" in data["error"]

    @pytest.mark.asyncio
    async def test_question_not_a_dict(self):
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": ["just a string"],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "dict" in data["error"]

    @pytest.mark.asyncio
    async def test_too_many_personas(self):
        from synth_panel.mcp.server import MAX_PERSONAS

        personas = [{"name": f"P{i}"} for i in range(MAX_PERSONAS + 1)]
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": personas,
                "questions": [{"text": "Hello?"}],
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "Too many personas" in data["error"]

    @pytest.mark.asyncio
    async def test_too_many_questions(self):
        from synth_panel.mcp.server import MAX_QUESTIONS

        questions = [{"text": f"Q{i}?"} for i in range(MAX_QUESTIONS + 1)]
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": questions,
            },
        )
        data = json.loads(result[0][0].text)
        assert "error" in data
        assert "Too many questions" in data["error"]

    @pytest.mark.asyncio
    async def test_valid_input_passes_validation(self):
        """Valid personas + questions pass validation and reach the runner."""
        with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"results": []}
            await mcp.call_tool(
                "run_panel",
                {
                    "personas": [{"name": "Alice"}],
                    "questions": [{"text": "Hello?"}],
                },
            )
            assert mock_run.called


# ---------------------------------------------------------------------------
# Synthesis error surfacing
# ---------------------------------------------------------------------------


class TestSynthesisErrorSurfacing:
    """Synthesis failures should be logged and surfaced, not silently swallowed."""

    def test_constants_exported(self):
        from synth_panel.mcp.server import MAX_PERSONAS, MAX_QUESTIONS

        assert MAX_PERSONAS == 100
        assert MAX_QUESTIONS == 50
