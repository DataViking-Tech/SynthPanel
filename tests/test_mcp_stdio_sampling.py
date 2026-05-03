"""Integration test — spawn the MCP server over stdio and verify that
the sampling fallback round-trips when the server has no BYOK creds.

Fills a gap left by :mod:`tests.test_mcp_sampling`, which only exercises
the tool handlers directly against a mocked Context. Here we run the
real :func:`synth_panel.mcp.server.serve` subprocess, connect a
:class:`mcp.ClientSession` that advertises the ``sampling`` capability
(via a ``sampling_callback``), and assert the server emits
``sampling/createMessage`` and gets a usable response back.

The subprocess runs with every provider key unset so the decision in
:func:`synth_panel.mcp.sampling.decide_mode` must pick ``sampling`` —
this is the exact scenario that previously crashed ``run_quick_poll``
with a KeyError stack trace (sp-5no).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    TextContent,
)

from mcp import ClientSession, StdioServerParameters


def _server_env() -> dict[str, str]:
    """Return a subprocess env with every provider key scrubbed so the
    server is forced to resolve run_quick_poll via sampling."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "XAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ):
        env.pop(var, None)
    return env


def _locate_server_entry() -> StdioServerParameters:
    """Resolve how to spawn the server — prefer the installed console
    script, fall back to ``python -m synth_panel.mcp.server``."""
    entry = shutil.which("synthpanel")
    if entry:
        return StdioServerParameters(command=entry, args=["mcp-serve"], env=_server_env())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "synth_panel", "mcp-serve"],
        env=_server_env(),
    )


async def _sampling_callback(
    context: RequestContext[ClientSession, Any],
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """Minimal sampling responder — echoes a canned string so the test
    asserts on known output. The real integration point is that the
    server issued a ``sampling/createMessage`` request at all."""
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="host-sampled-response"),
        model="host-agent-stub",
        stopReason="endTurn",
    )


@pytest.mark.asyncio
async def test_stdio_quick_poll_routes_through_sampling():
    """End-to-end: spawn the server, call run_quick_poll with no BYOK
    creds, and confirm it emits sampling/createMessage and returns the
    host-sampled content."""
    params = _locate_server_entry()

    async with (
        stdio_client(params) as (read, write),
        ClientSession(
            read,
            write,
            sampling_callback=_sampling_callback,
        ) as session,
    ):
        await session.initialize()

        # Server declared tools — confirm run_quick_poll is present.
        tools = await session.list_tools()
        tool_names = {t.name for t in tools.tools}
        assert "run_quick_poll" in tool_names, tool_names

        # Run the poll with no BYOK creds. The server must route
        # through sampling and emit sampling/createMessage, which
        # our callback answers.
        response = await session.call_tool(
            "run_quick_poll",
            {
                "question": "Is the sky blue?",
                "personas": [{"name": "Alice"}],
                "synthesis": False,
            },
        )

        # Extract text payload from the tool response.
        assert response.content, "tool returned empty content"
        text_block = next(
            (block for block in response.content if getattr(block, "type", None) == "text"),
            None,
        )
        assert text_block is not None, "no text block in tool response"
        payload = json.loads(text_block.text)

        # Either the server successfully ran sampling end-to-end…
        if "error" not in payload:
            assert payload["mode"] == "sampling", payload
            assert payload["persona_count"] == 1
            # The host-sampled response bubbles up through the
            # response aggregation — must not be empty/KeyError.
            assert payload["results"], payload
            first_answer = payload["results"][0]["responses"][0]["answer"]
            assert first_answer == "host-sampled-response", payload
        else:  # …or it rejected with a structured, user-actionable message.
            # A rejection is only acceptable for a capability/config
            # reason, never an unhandled KeyError on "usage".
            assert "usage" not in payload["error"].lower(), payload


@pytest.mark.asyncio
async def test_stdio_initialize_advertises_sampling_and_version():
    """sp-lsc/sp-a59 regression: the server's initialize handshake must
    declare that it uses MCP sampling — both at the top level of
    ``capabilities`` (so inspectors enumerating top-level keys see it)
    and under ``experimental`` (back-compat nesting) — and must report
    the synthpanel package version in serverInfo rather than leaking
    the MCP SDK version through FastMCP's default behaviour."""
    import synth_panel

    params = _locate_server_entry()

    async with (
        stdio_client(params) as (read, write),
        ClientSession(
            read,
            write,
            sampling_callback=_sampling_callback,
        ) as session,
    ):
        init_result = await session.initialize()

        assert init_result.serverInfo.name == "synthpanel"
        assert init_result.serverInfo.version == synth_panel.__version__, (
            f"serverInfo.version should be the synthpanel package version "
            f"({synth_panel.__version__}); got {init_result.serverInfo.version}. "
            f"FastMCP defaults to importlib.metadata.version('mcp') when the "
            f"underlying Server.version is unset, which leaks the SDK version."
        )

        experimental = init_result.capabilities.experimental or {}
        assert "sampling" in experimental, (
            f"Server must advertise the 'sampling' experimental capability "
            f"so MCP clients can surface the dependency in their UI. "
            f"Got capabilities.experimental = {experimental!r}"
        )

        # sp-a59: top-level advertisement too — ServerCapabilities is
        # ``extra="allow"`` so the key round-trips even though the MCP
        # spec defines sampling as a client capability. Many hosts and
        # MCP inspectors enumerate top-level keys only, and will miss
        # sampling if it's hidden under ``experimental``.
        top_level = init_result.capabilities.model_dump(exclude_none=True)
        assert "sampling" in top_level, (
            f"Server must advertise 'sampling' at the top level of "
            f"ServerCapabilities (alongside prompts/resources/tools). "
            f"Got top-level capability keys = {sorted(top_level)!r}"
        )


@pytest.mark.asyncio
async def test_stdio_quick_poll_without_personas_uses_defaults():
    """sp-lsc regression: run_quick_poll must work with zero configuration
    — omitting personas falls back to the built-in diverse persona set so
    first-run users aren't forced to hand-craft a personas list."""
    params = _locate_server_entry()

    async with (
        stdio_client(params) as (read, write),
        ClientSession(
            read,
            write,
            sampling_callback=_sampling_callback,
        ) as session,
    ):
        await session.initialize()

        response = await session.call_tool(
            "run_quick_poll",
            {
                "question": "Is the sky blue?",
                "synthesis": False,
            },
        )

        assert response.content
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        assert text_block is not None
        payload = json.loads(text_block.text)

        # Must not be a 'personas field required' validation error — that
        # was the exact regression introduced by sp-5no.
        assert "error" not in payload or "personas" not in payload["error"].lower(), payload

        if "error" not in payload:
            assert payload["mode"] == "sampling"
            assert payload["persona_count"] >= 1


@pytest.mark.asyncio
async def test_stdio_run_panel_routes_through_sampling():
    """Companion: the same fallback must also be wired for run_panel,
    since sp-5no audit specifically called out the usage KeyError on the
    multi-question path."""
    params = _locate_server_entry()

    async with (
        stdio_client(params) as (read, write),
        ClientSession(
            read,
            write,
            sampling_callback=_sampling_callback,
        ) as session,
    ):
        await session.initialize()

        response = await session.call_tool(
            "run_panel",
            {
                "questions": [{"text": "What do you think?"}],
                "personas": [{"name": "Alice"}],
                "synthesis": False,
            },
        )

        assert response.content
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            None,
        )
        assert text_block is not None
        payload = json.loads(text_block.text)

        assert "error" not in payload, payload
        assert payload["mode"] == "sampling"
        assert payload["persona_count"] == 1
        assert payload["question_count"] == 1
        assert payload["results"][0]["responses"][0]["answer"] == "host-sampled-response"
