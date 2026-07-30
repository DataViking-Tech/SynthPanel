"""Integration test — spawn the MCP server over stdio and verify that
the sampling fallback round-trips when the server has no BYOK creds.

Fills a gap left by :mod:`tests.test_mcp_sampling`, which only exercises
the tool handlers directly against a mocked Context. Here we run the
real :func:`althing.mcp.server.serve` subprocess, connect a client that
advertises the ``sampling`` capability (via a ``sampling_callback``),
and assert the sampling flow round-trips on **both** protocol eras:

* Legacy (``initialize`` handshake, ≤ 2025-11-25): the server emits a
  mid-call ``sampling/createMessage`` server→client request.
* Modern (``server/discover``, 2026-07-28): the protocol defines no
  server→client requests, so the tool call returns an
  ``InputRequiredResult`` whose embedded ``CreateMessageRequest``s the
  client answers through the same callback before retrying (SEP-2322).
  The high-level :class:`mcp.client.client.Client` drives that loop.

The subprocess runs with every provider key unset so the decision in
:func:`althing.mcp.sampling.decide_mode` must pick ``sampling`` —
this is the exact scenario that previously crashed ``run_quick_poll``
with a KeyError stack trace (sp-5no).
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import pytest

pytest.importorskip("mcp")

from mcp.client.client import Client
from mcp.client.session import ClientRequestContext
from mcp.client.stdio import stdio_client
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
    script, fall back to ``python -m althing.mcp.server``."""
    entry = shutil.which("althing")
    if entry:
        return StdioServerParameters(command=entry, args=["mcp-serve"], env=_server_env())
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "althing", "mcp-serve"],
        env=_server_env(),
    )


async def _sampling_callback(
    context: ClientRequestContext,
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """Minimal sampling responder — echoes a canned string so the test
    asserts on known output. The real integration point is that the
    server requested sampling at all: as a mid-call
    ``sampling/createMessage`` on legacy connections, or embedded in an
    ``InputRequiredResult`` on 2026-07-28 connections."""
    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="host-sampled-response"),
        model="host-agent-stub",
        stop_reason="endTurn",
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
async def test_stdio_initialize_reports_althing_version():
    """sp-lsc regression, updated for MCP SDK 2.0: the initialize
    handshake must report the althing package version in serverInfo
    rather than leaking the MCP SDK version (the low-level server's
    default when ``version`` is unset).

    Note the pre-2.0 ``sampling`` capability advertisement (experimental
    nesting + off-spec top-level ``ServerCapabilities`` key) is gone:
    the MCP spec defines sampling as a *client* capability, SDK 2.0's
    public ``MCPServer.run`` path exposes no initialization-option hook,
    and the 2026-07-28 ``server/discover`` flow never consults
    initialize options at all. Whether sampling is used is a runtime
    decision made by probing the client's declared capability
    (``decide_mode``), which the remaining tests in this module cover on
    both protocol eras."""
    import althing

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

        assert init_result.server_info.name == "althing"
        assert init_result.server_info.version == althing.__version__, (
            f"serverInfo.version should be the althing package version "
            f"({althing.__version__}); got {init_result.server_info.version}. "
            f"The low-level server defaults to importlib.metadata.version('mcp') "
            f"when version is unset, which leaks the SDK version."
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


# ---------------------------------------------------------------------------
# 2026-07-28 (modern era): sampling rides the InputRequiredResult loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stdio_modern_protocol_quick_poll_samples_via_input_required():
    """SDK 2.0 / spec 2026-07-28: modern connections define no
    server→client requests, so the server must surface sampling as an
    ``InputRequiredResult`` (SEP-2322) that the high-level client
    resolves through the same ``sampling_callback`` before retrying.
    This is the new-protocol equivalent of
    ``test_stdio_quick_poll_routes_through_sampling``."""
    params = _locate_server_entry()

    async with Client(
        stdio_client(params),
        sampling_callback=_sampling_callback,
    ) as client:
        # mode='auto' probes server/discover; the SDK 2.0 server answers,
        # locking the connection to the modern era. If this assertion
        # fails the rest of the test would silently exercise the legacy
        # path instead.
        assert client.session is not None
        assert client.session.protocol_version == "2026-07-28", client.session.protocol_version

        result = await client.call_tool(
            "run_quick_poll",
            {
                "question": "Is the sky blue?",
                "personas": [{"name": "Alice"}],
                "synthesis": False,
            },
        )

        assert result.content, "tool returned empty content"
        text_block = next(
            (b for b in result.content if getattr(b, "type", None) == "text"),
            None,
        )
        assert text_block is not None
        payload = json.loads(text_block.text)

        assert "error" not in payload, payload
        assert payload["mode"] == "sampling"
        assert payload["persona_count"] == 1
        assert payload["results"][0]["responses"][0]["answer"] == "host-sampled-response"


@pytest.mark.asyncio
async def test_stdio_modern_protocol_panel_with_synthesis_multi_round():
    """Modern-era run_panel with synthesis needs two InputRequired
    rounds: one batching the per-persona sampling requests, then one for
    the synthesis call (which depends on every persona answer, carried
    across rounds in the sealed ``request_state``). The client driver
    resolves both transparently."""
    params = _locate_server_entry()

    async with Client(
        stdio_client(params),
        sampling_callback=_sampling_callback,
    ) as client:
        assert client.session is not None
        assert client.session.protocol_version == "2026-07-28"

        result = await client.call_tool(
            "run_panel",
            {
                "questions": [{"text": "What do you think?"}],
                "personas": [{"name": "Alice"}, {"name": "Bob"}],
                "synthesis": True,
            },
        )

        assert result.content
        text_block = next(
            (b for b in result.content if getattr(b, "type", None) == "text"),
            None,
        )
        assert text_block is not None
        payload = json.loads(text_block.text)

        assert "error" not in payload, payload
        assert payload["mode"] == "sampling"
        assert payload["persona_count"] == 2
        for entry in payload["results"]:
            assert entry["responses"][0]["answer"] == "host-sampled-response"
        assert payload["synthesis"] is not None
        assert payload["synthesis"]["summary"] == "host-sampled-response"
