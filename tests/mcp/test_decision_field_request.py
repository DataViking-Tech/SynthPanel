"""AC-3: decision-field request validation in MCP handlers.

Verifies that ``run_panel`` / ``run_quick_poll`` / ``extend_panel`` apply
the v1.0.0 frozen-contract validator (AC-2) to ``decision_being_informed``
and return the typed error envelope on contract violations. ``run_prompt``
is sub-decisional (per SPEC §12.1) and must NOT advertise the field.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point data dir at temp and stub a provider key for BYOK paths."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")


from synth_panel.mcp.server import mcp

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"
_PANEL_TOOLS = ("run_panel", "run_quick_poll", "extend_panel")


def _payload(text: str) -> dict:
    return json.loads(text)


def _call_args_for(tool: str, *, decision: str | None) -> dict:
    """Build a minimum-viable kwargs dict for each panel-running tool."""
    if tool == "run_panel":
        args: dict = {
            "personas": [{"name": "Alice"}],
            "questions": [{"text": "Hello?"}],
        }
    elif tool == "run_quick_poll":
        args = {"question": "What do you think?"}
    elif tool == "extend_panel":
        args = {"result_id": "any", "questions": [{"text": "Follow-up?"}]}
    else:
        raise AssertionError(f"unhandled tool {tool!r}")
    if decision is not None:
        args["decision_being_informed"] = decision
    return args


# ---------------------------------------------------------------------------
# Tool-schema introspection — run_prompt must not surface the field.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_prompt_rejects_decision_field():
    """``run_prompt`` is sub-decisional and must not advertise the field.

    SPEC §12.1: ``decision_being_informed`` lives on ``run_panel`` /
    ``run_quick_poll`` / ``extend_panel`` and explicitly *not* on
    ``run_prompt``. The MCP-advertised input schema is the contract a
    client introspects; if the field shows up there we have leaked the
    decision contract into a sub-decisional tool.
    """
    tools = await mcp.list_tools()
    schema = next(t for t in tools if t.name == "run_prompt").inputSchema
    properties = schema.get("properties", {}) or {}
    assert "decision_being_informed" not in properties


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_panel_tools_advertise_decision_field(tool: str):
    tools = await mcp.list_tools()
    schema = next(t for t in tools if t.name == tool).inputSchema
    properties = schema.get("properties", {}) or {}
    assert "decision_being_informed" in properties, f"{tool} must advertise decision_being_informed in its input schema"


# ---------------------------------------------------------------------------
# Empty-after-trim — typed MISSING_DECISION envelope.
#
# A bare-omitted field is the v1.0.x grace path (handled by AC-4's
# ``apply_legacy_grace``); AC-3 only owns the constraint surface, so
# these tests cover the "provided but empty/whitespace" case where the
# validator-core fires.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_whitespace_only_decision_treated_as_missing(tool: str):
    result = await mcp.call_tool(tool, _call_args_for(tool, decision="          "))
    data = _payload(result[0][0].text)
    assert data["error_code"] == "MISSING_DECISION"
    assert data["field_path"] == "decision_being_informed"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_empty_string_decision_treated_as_missing(tool: str):
    result = await mcp.call_tool(tool, _call_args_for(tool, decision=""))
    data = _payload(result[0][0].text)
    assert data["error_code"] == "MISSING_DECISION"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_omitted_decision_is_pass_through(tool: str):
    """Bare-omitted field is the AC-4 grace path; AC-3 must not reject it.

    Under v1.0.x the missing-field path goes through ``apply_legacy_grace``
    which synthesises ``"unspecified-legacy-call"``. AC-3's constraint
    validator must therefore *not* short-circuit when the field is absent,
    or every legacy call regresses to a contract error. ``run_panel`` and
    ``run_quick_poll`` are exercised here; ``extend_panel`` requires a
    real ``result_id`` to reach its post-validation path and is covered
    indirectly by the existing extend_panel test suite.
    """
    if tool == "extend_panel":
        pytest.skip("extend_panel post-validation path needs a real result_id")
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {"results": []}
        result = await mcp.call_tool(tool, _call_args_for(tool, decision=None))
        data = _payload(result[0][0].text)
        assert data.get("error_code") not in {
            "MISSING_DECISION",
            "INVALID_TOOL_ARG",
            "DECISION_TOO_LONG",
        }, f"omitted field must pass through; got {data}"


# ---------------------------------------------------------------------------
# Length / shape violations — typed envelopes for each rule.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_decision_too_long_returns_typed_envelope(tool: str):
    result = await mcp.call_tool(tool, _call_args_for(tool, decision="x" * 281))
    data = _payload(result[0][0].text)
    assert data["error_code"] == "DECISION_TOO_LONG"
    assert data["field_path"] == "decision_being_informed"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_short_after_trim_returns_invalid_tool_arg(tool: str):
    result = await mcp.call_tool(tool, _call_args_for(tool, decision="  short  "))
    data = _payload(result[0][0].text)
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert data["field_path"] == "decision_being_informed"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", _PANEL_TOOLS)
async def test_newline_in_decision_rejected(tool: str):
    result = await mcp.call_tool(tool, _call_args_for(tool, decision="Decide whether\nto ship anything yet."))
    data = _payload(result[0][0].text)
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert "newline" in data["message"].lower()


# ---------------------------------------------------------------------------
# Boundary conditions on the trimmed length.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["run_panel", "run_quick_poll"])
async def test_minimum_length_accepted(tool: str):
    """Exactly 12 chars after trim must pass the length gate.

    Excludes ``extend_panel`` because its post-validation path requires a
    real ``result_id`` to load — the boundary semantics are exercised at
    the validator-core level (AC-2 unit tests).
    """
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_panel:
        mock_panel.return_value = {"results": []}
        result = await mcp.call_tool(tool, _call_args_for(tool, decision="exactly12chr"))
        data = _payload(result[0][0].text)
        assert data.get("field_path") != "decision_being_informed", (
            f"length-12 decision must not produce a decision-field envelope; got {data}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["run_panel", "run_quick_poll"])
async def test_maximum_length_accepted(tool: str):
    """Exactly 280 chars after trim must pass the length gate."""
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_panel:
        mock_panel.return_value = {"results": []}
        result = await mcp.call_tool(tool, _call_args_for(tool, decision="x" * 280))
        data = _payload(result[0][0].text)
        assert data.get("error_code") not in {
            "DECISION_TOO_LONG",
            "MISSING_DECISION",
        }


# ---------------------------------------------------------------------------
# Valid decision flows past the gate into the normal handler path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_panel_valid_decision_reaches_runner():
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = {"results": []}
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "Hello?"}],
                "decision_being_informed": _VALID_DECISION,
            },
        )
        assert mock_run.called, "valid decision must not short-circuit"
        data = _payload(result[0][0].text)
        assert "error_code" not in data
