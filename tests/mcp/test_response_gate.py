"""AC-9: response-side gate.

The MCP server must run :func:`validate_response` over every artifact that
claims to be a v1.0.0 ``panel_verdict`` (i.e. carries ``schema_version``)
before that artifact leaves the server. Bogus flag codes, missing required
fields, or schema-version drift are swapped for the typed error envelope
on egress; nothing non-conformant escapes.

The gate is intentionally a no-op for legacy, non-verdict return shapes
(``{"results": [...], ...}`` etc.) so the contract is enforced without
breaking pre-contract callers.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")


from synth_panel.mcp.server import mcp
from synth_panel.structured.validate import apply_response_gate

_VALID_DECISION = "choosing launch tier price"


def _verdict(**overrides):
    base = {
        "headline": "Cohort splits on $79; price is the dominant objection.",
        "convergence": 0.62,
        "dissent_count": 1,
        "top_3_verbatims": [],
        "flags": [{"code": "low_convergence", "severity": "warn"}],
        "extension": [],
        "full_transcript_uri": "panel-result://abc",
        "meta": {"decision_being_informed": _VALID_DECISION},
        "schema_version": "1.0.0",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Unit-level: the gate helper itself
# ---------------------------------------------------------------------------


def test_gate_passes_through_non_verdict_payload() -> None:
    """Legacy MCP shapes (no ``schema_version``) must traverse the gate
    unchanged. The gate triggers only on artifacts claiming to be v1.0.0
    ``panel_verdict``s; everything else is the contract's responsibility-
    free zone."""
    legacy = {"results": [], "result_id": "abc", "warnings": []}
    assert apply_response_gate(legacy) is legacy


def test_gate_passes_through_valid_verdict() -> None:
    artifact = _verdict()
    assert apply_response_gate(artifact) is artifact


def test_gate_blocks_invalid_flag() -> None:
    """A flag code outside the closed enum is rejected; the gate replaces
    the artifact with the typed error envelope dict."""
    artifact = _verdict(flags=[{"code": "totally_made_up", "severity": "warn"}])
    out = apply_response_gate(artifact)
    assert out is not artifact
    assert out["error_code"] == "INVALID_FLAG"
    assert out["field_path"] == "flags[0].code"
    assert out["schema_version"] == "1.0.0"
    assert out["retry_safe"] is False


def test_gate_blocks_schema_version_drift() -> None:
    artifact = _verdict(schema_version="0.12.0")
    out = apply_response_gate(artifact)
    assert out["error_code"] == "SCHEMA_DRIFT"
    assert out["field_path"] == "schema_version"
    assert out["retry_safe"] is False


def test_gate_blocks_missing_required_field() -> None:
    artifact = _verdict()
    del artifact["headline"]
    out = apply_response_gate(artifact)
    assert out["error_code"] == "SCHEMA_DRIFT"
    assert out["field_path"] == "headline"
    assert out["retry_safe"] is False


# ---------------------------------------------------------------------------
# Integration: the gate is wired into the MCP run_panel return path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_flag_blocks_egress() -> None:
    """End-to-end: when the panel runner produces a panel_verdict carrying
    a flag outside the closed enum, the MCP egress yields the typed
    INVALID_FLAG envelope, not the malformed artifact.

    This is the AC-9 contract: ``validate_response`` runs at the response
    boundary; non-conformant artifacts can never reach the caller.
    """
    bad = _verdict(flags=[{"code": "totally_made_up", "severity": "warn"}])
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = bad
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "How do you feel?"}],
            },
        )

    data = json.loads(result[0][0].text)
    assert data["error_code"] == "INVALID_FLAG"
    assert data["field_path"] == "flags[0].code"
    assert data["schema_version"] == "1.0.0"
    assert data["retry_safe"] is False
    # The malformed verdict's headline must NOT have escaped the boundary.
    assert "headline" not in data


@pytest.mark.asyncio
async def test_valid_verdict_passes_egress_unchanged() -> None:
    """A conformant panel_verdict traverses the gate intact: same flags,
    same headline, same schema_version. The gate must not mutate or
    repackage successful artifacts."""
    good = _verdict()
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = good
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "How do you feel?"}],
            },
        )

    data = json.loads(result[0][0].text)
    assert data["schema_version"] == "1.0.0"
    assert data["headline"] == good["headline"]
    assert data["flags"] == good["flags"]


@pytest.mark.asyncio
async def test_legacy_shape_unaffected() -> None:
    """Pre-contract MCP responses (no ``schema_version``) must still flow
    through unchanged so the AC-9 wiring doesn't regress shipped clients."""
    legacy = {"results": [], "result_id": "xyz", "warnings": []}
    with patch("synth_panel.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = legacy
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "How do you feel?"}],
            },
        )

    data = json.loads(result[0][0].text)
    assert data == legacy
