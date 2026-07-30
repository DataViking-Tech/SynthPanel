"""AC-6/AC-8/AC-9 emission: ``panel_verdict`` on the success path.

* Successful BYOK panel runs attach a schema-valid ``panel_verdict``
  alongside ``synthesis`` and stamp ``schema_version: "1.0.0"`` on the
  envelope, which arms the AC-9 response gate for real.
* A deliberately malformed verdict is swapped for the typed error envelope
  at egress (the gate is no longer a no-op on success responses).
* AC-8: structured-output 3-strike exhaustion routes through
  ``exhausted_retry_outcome`` — typed ``SCHEMA_DRIFT`` error by default,
  degraded verdict with a ``schema_drift`` warn flag under
  ``ALTHING_DRIFT_DEGRADE=1`` (docs/structured-polling.md).
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
    monkeypatch.delenv("ALTHING_SCHEMA_MIN", raising=False)
    monkeypatch.delenv("ALTHING_DRIFT_DEGRADE", raising=False)


from althing.mcp.server import mcp
from althing.structured.validate import apply_response_gate, validate_response

from .test_decision_wiring import (
    _stub_run_panel_parallel_with_sessions,
    _stub_synthesize_panel,
    _StubMcpContext,
)

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"


def _verdict(**overrides):
    base = {
        "headline": "Cohort splits on $79.",
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


async def _run_panel_tool(**response_overrides):
    from althing.mcp import server as _server

    with (
        patch(
            "althing.orchestrator.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(**response_overrides),
        ),
        patch(
            "althing._runners.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(**response_overrides),
        ),
        patch("althing._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("althing.mcp.server._shared_client", None),
    ):
        raw = await _server.run_panel(
            personas=[{"name": "Alice"}, {"name": "Bob"}],
            questions=[{"text": "What would you pay?"}],
            model="haiku",
            decision_being_informed=_VALID_DECISION,
            ctx=_StubMcpContext(),
        )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Success path: verdict present, schema-valid, alongside synthesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_envelope_carries_schema_valid_verdict():
    data = await _run_panel_tool()

    assert "error_code" not in data
    verdict = data["panel_verdict"]
    assert validate_response(verdict) is None
    # Alongside synthesis, not replacing it.
    assert data["synthesis"] is not None
    # Envelope-level version stamp arms the response gate.
    assert data["schema_version"] == "1.0.0"
    # Verdict internals derived from the run.
    assert verdict["full_transcript_uri"] == f"panel-result://{data['result_id']}"
    assert verdict["headline"]
    # 2 panelists → small_n block flag from the AC-5 raiser.
    assert {"code": "small_n", "severity": "block"} in verdict["flags"]
    # Verbatims are deterministic first-text-response picks.
    assert verdict["top_3_verbatims"] and verdict["top_3_verbatims"][0]["persona_id"] == "Alice"


@pytest.mark.asyncio
async def test_v3_instrument_success_envelope_carries_verdict():
    from althing.mcp import server as _server

    instrument = {
        "version": 3,
        "rounds": [
            {"name": "r1", "questions": [{"text": "Q1"}]},
            {"name": "r2", "questions": [{"text": "Q2"}]},
        ],
    }
    with (
        patch(
            "althing.orchestrator.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch(
            "althing._runners.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch("althing._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("althing.mcp.server._shared_client", None),
    ):
        raw = await _server.run_panel(
            personas=[{"name": "Alice"}, {"name": "Bob"}],
            instrument=instrument,
            model="haiku",
            decision_being_informed=_VALID_DECISION,
            ctx=_StubMcpContext(),
        )
    data = json.loads(raw)

    assert data["schema_version"] == "1.0.0"
    assert validate_response(data["panel_verdict"]) is None
    assert data["panel_verdict"]["meta"]["decision_being_informed"] == _VALID_DECISION


# ---------------------------------------------------------------------------
# AC-9: the gate actually validates success envelopes now
# ---------------------------------------------------------------------------


def test_gate_passes_envelope_with_valid_nested_verdict():
    envelope = {"results": [], "panel_verdict": _verdict(), "schema_version": "1.0.0"}
    assert apply_response_gate(envelope) is envelope


def test_gate_blocks_envelope_with_malformed_nested_verdict():
    envelope = {
        "results": [],
        "panel_verdict": _verdict(flags=[{"code": "totally_made_up", "severity": "warn"}]),
        "schema_version": "1.0.0",
    }
    out = apply_response_gate(envelope)
    assert out is not envelope
    assert out["error_code"] == "INVALID_FLAG"
    assert out["retry_safe"] is False


def test_gate_blocks_envelope_with_wrong_top_level_schema_version():
    envelope = {"results": [], "panel_verdict": _verdict(), "schema_version": "0.12.0"}
    out = apply_response_gate(envelope)
    assert out["error_code"] == "SCHEMA_DRIFT"
    assert out["field_path"] == "schema_version"


def test_gate_passes_through_typed_error_envelopes():
    err = {
        "error_code": "SCHEMA_DRIFT",
        "message": "boom",
        "schema_version": "1.0.0",
        "retry_safe": True,
    }
    assert apply_response_gate(err) is err


@pytest.mark.asyncio
async def test_malformed_envelope_verdict_blocked_at_egress():
    """End-to-end: a success envelope whose nested verdict violates the
    contract is replaced by the typed error envelope on the wire."""
    bad_envelope = {
        "result_id": "abc",
        "results": [],
        "panel_verdict": _verdict(flags=[{"code": "totally_made_up", "severity": "warn"}]),
        "schema_version": "1.0.0",
    }
    with patch("althing.mcp.server._run_panel_async", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = bad_envelope
        result = await mcp.call_tool(
            "run_panel",
            {
                "personas": [{"name": "Alice"}],
                "questions": [{"text": "How do you feel?"}],
                "decision_being_informed": _VALID_DECISION,
            },
        )
    data = json.loads(result.content[0].text)
    assert data["error_code"] == "INVALID_FLAG"
    assert "panel_verdict" not in data


# ---------------------------------------------------------------------------
# AC-8: structured-retry exhaustion routes through exhausted_retry_outcome
# ---------------------------------------------------------------------------

_FALLBACK_RESPONSE = {
    "response": {"_error": "3 strikes", "_fallback": True},
    "structured": True,
    "is_fallback": True,
}


@pytest.mark.asyncio
async def test_structured_exhaustion_returns_schema_drift_error_by_default():
    data = await _run_panel_tool(**_FALLBACK_RESPONSE)

    assert data["error_code"] == "SCHEMA_DRIFT"
    assert data["schema_version"] == "1.0.0"
    # Pre-exhaustion drift is retryable with different stimulus.
    assert data["retry_safe"] is True
    assert "ALTHING_DRIFT_DEGRADE" in data["message"]
    assert "panel_verdict" not in data


@pytest.mark.asyncio
async def test_structured_exhaustion_degrades_with_flag_when_enabled(monkeypatch):
    monkeypatch.setenv("ALTHING_DRIFT_DEGRADE", "1")
    data = await _run_panel_tool(**_FALLBACK_RESPONSE)

    assert "error_code" not in data
    verdict = data["panel_verdict"]
    assert {"code": "schema_drift", "severity": "warn"} in verdict["flags"]
    assert validate_response(verdict) is None
