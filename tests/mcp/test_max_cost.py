"""GH#576: the ``max_cost`` budget ceiling on the panel-running MCP tools.

The CLI's ``--max-cost`` projected-total spend gate previously had no MCP
analog — agents could only budget-gate by piloting small, reading in-band
``total_cost``, and scaling manually. ``run_panel`` / ``run_quick_poll`` /
``extend_panel`` now accept ``max_cost`` (USD) and wire it into the same
:class:`~synth_panel.cost.CostGate` machinery the CLI uses, with the same
soft-halt semantics: in-flight panelists finish, no new panelists start,
synthesis is skipped, and the response is a valid *partial* envelope with
``run_invalid: true``, ``cost_exceeded: true``, ``abort_reason:
"cost_exceeded"``, ``halted_at_panelist``, the ``cost_gate`` snapshot, and
an agent-legible ``resume`` block.

Only the LLM boundary (``run_panel_parallel`` / ``synthesize_panel``) is
stubbed; the MCP tool functions, ``run_panel_sync``, and the CostGate run
for real.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    monkeypatch.delenv("SYNTHPANEL_DRIFT_DEGRADE", raising=False)


from .test_decision_wiring import _stub_synthesize_panel, _StubMcpContext

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"
_PERSONAS = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Cara"}]
_QUESTIONS = [{"text": "What would you pay?"}]


def _gate_aware_run_panel_parallel(per_panelist_cost: float, captured: dict):
    """Stub for ``run_panel_parallel`` that honors the cost gate for real.

    Emulates the orchestrator's soft-halt loop: each persona "completes"
    in order, its cost is recorded against the gate, and once the gate
    halts no further personas are dispatched — exactly the contract
    ``run_panel_parallel`` documents. The real :class:`CostGate` does the
    projection math; nothing about the trip is faked.
    """
    from synth_panel.cost import TokenUsage
    from synth_panel.orchestrator import PanelistResult
    from synth_panel.persistence import ConversationMessage, Session

    def _fake(client=None, personas=None, questions=None, model=None, sessions=None, cost_gate=None, **kwargs):
        captured["cost_gate"] = cost_gate
        captured["personas"] = list(personas or [])
        results = []
        out_sessions = dict(sessions or {})
        for p in personas or []:
            if cost_gate is not None and cost_gate.should_halt():
                break
            name = p.get("name", "anon")
            results.append(
                PanelistResult(
                    persona_name=name,
                    responses=[{"question": (questions or [{}])[0].get("text", ""), "response": "I would pay $49."}],
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model=model,
                )
            )
            sess = out_sessions.get(name)
            if not isinstance(sess, Session):
                sess = Session()
            sess.push_message(ConversationMessage(role="user", content=[{"type": "text", "text": "q"}]))
            sess.push_message(ConversationMessage(role="assistant", content=[{"type": "text", "text": "a"}]))
            out_sessions[name] = sess
            if cost_gate is not None:
                # Record a fixed priced cost per panelist (the real path
                # prices pr.usage; the gate math under test is identical).
                cost_gate.record(per_panelist_cost)
        return results, {}, out_sessions

    return _fake


async def _run_panel_tool(max_cost, per_panelist_cost: float = 10.0, captured: dict | None = None, **extra):
    from synth_panel.mcp import server as _server

    captured = captured if captured is not None else {}
    synth_mock = MagicMock(side_effect=_stub_synthesize_panel)
    with (
        patch(
            "synth_panel._runners.run_panel_parallel",
            side_effect=_gate_aware_run_panel_parallel(per_panelist_cost, captured),
        ),
        patch("synth_panel._runners.synthesize_panel", synth_mock),
        patch("synth_panel.mcp.server._shared_client", None),
        patch("synth_panel.mcp.server.LLMClient", MagicMock()),
    ):
        raw = await _server.run_panel(
            personas=list(_PERSONAS),
            questions=list(_QUESTIONS),
            model="haiku",
            max_cost=max_cost,
            decision_being_informed=_VALID_DECISION,
            ctx=_StubMcpContext(),
            **extra,
        )
    return json.loads(raw), captured, synth_mock


# ---------------------------------------------------------------------------
# Param plumbing
# ---------------------------------------------------------------------------


class TestMaxCostPlumbing:
    @pytest.mark.asyncio
    async def test_run_panel_builds_gate_from_max_cost(self):
        """max_cost reaches run_panel_parallel as a real CostGate sized to the panel."""
        from synth_panel.cost import CostGate

        data, captured, _synth = await _run_panel_tool(max_cost=50.0, per_panelist_cost=0.01)
        gate = captured["cost_gate"]
        assert isinstance(gate, CostGate)
        assert gate.max_cost_usd == 50.0
        assert gate.total_panelists == len(_PERSONAS)
        # Generous ceiling: run completes normally, no partial markers.
        assert "cost_exceeded" not in data
        assert data.get("run_invalid") is not True
        assert data.get("synthesis") is not None

    @pytest.mark.asyncio
    async def test_run_panel_without_max_cost_passes_no_gate(self):
        data, captured, _synth = await _run_panel_tool(max_cost=None)
        assert captured["cost_gate"] is None
        assert "cost_exceeded" not in data
        assert "cost_gate" not in data

    @pytest.mark.asyncio
    async def test_run_quick_poll_builds_gate(self):
        from synth_panel.cost import CostGate
        from synth_panel.mcp import server as _server

        captured: dict = {}
        with (
            patch(
                "synth_panel._runners.run_panel_parallel",
                side_effect=_gate_aware_run_panel_parallel(0.01, captured),
            ),
            patch("synth_panel._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
            patch("synth_panel.mcp.server._shared_client", None),
            patch("synth_panel.mcp.server.LLMClient", MagicMock()),
        ):
            raw = await _server.run_quick_poll(
                question="What would you pay?",
                personas=list(_PERSONAS),
                model="haiku",
                max_cost=25.0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        data = json.loads(raw)
        gate = captured["cost_gate"]
        assert isinstance(gate, CostGate)
        assert gate.max_cost_usd == 25.0
        assert gate.total_panelists == len(_PERSONAS)
        assert "cost_exceeded" not in data


# ---------------------------------------------------------------------------
# Gate-trip envelope shape
# ---------------------------------------------------------------------------


class TestGateTripEnvelope:
    @pytest.mark.asyncio
    async def test_run_panel_trip_is_typed_partial(self):
        # $10/panelist against a $5 cap for 3 personas: the first completion
        # projects $30 > $5 and halts — 1 completed, 2 never dispatched.
        data, _captured, _synth = await _run_panel_tool(max_cost=5.0, per_panelist_cost=10.0)

        assert data["run_invalid"] is True
        assert data["cost_exceeded"] is True
        assert data["abort_reason"] == "cost_exceeded"
        assert data["halted_at_panelist"] == 1

        snap = data["cost_gate"]
        assert snap["halted"] is True
        assert snap["max_cost_usd"] == 5.0
        assert snap["running_cost_usd"] == pytest.approx(10.0)
        assert snap["projected_total_usd"] == pytest.approx(30.0)
        assert snap["completed"] == 1
        assert snap["total_panelists"] == 3

        # Spend so far, the cap, and how to resume — all agent-legible.
        resume = data["resume"]
        assert resume["completed_panelists"] == ["Alice"]
        assert resume["remaining_personas"] == ["Bob", "Cara"]
        assert resume["partial_result_id"] == data["result_id"]
        assert "get_panel_result" in resume["how_to_resume"]
        assert any("cost_exceeded" in w for w in data.get("warnings", []))

    @pytest.mark.asyncio
    async def test_trip_skips_synthesis(self):
        data, _captured, synth_mock = await _run_panel_tool(max_cost=5.0, per_panelist_cost=10.0)
        synth_mock.assert_not_called()
        assert data.get("synthesis") is None

    @pytest.mark.asyncio
    async def test_trip_composes_with_detail_summary_and_verdict(self):
        """The partial envelope still carries panel_verdict and honors detail."""
        data, _captured, _synth = await _run_panel_tool(max_cost=5.0, per_panelist_cost=10.0, detail="summary")
        assert data.get("detail") == "summary"
        assert "results" not in data  # transcript dropped under summary
        assert isinstance(data.get("panel_verdict"), dict)
        assert data.get("schema_version") == "1.0.0"
        # The persisted partial is retrievable.
        assert data.get("transcript_uri") == f"panel-result://{data['result_id']}"

    @pytest.mark.asyncio
    async def test_trip_full_detail_returns_partial_transcript(self):
        data, _captured, _synth = await _run_panel_tool(max_cost=5.0, per_panelist_cost=10.0, detail="full")
        assert [r["persona"] for r in data["results"]] == ["Alice"]

    @pytest.mark.asyncio
    async def test_run_quick_poll_trip_envelope(self):
        from synth_panel.mcp import server as _server

        captured: dict = {}
        with (
            patch(
                "synth_panel._runners.run_panel_parallel",
                side_effect=_gate_aware_run_panel_parallel(10.0, captured),
            ),
            patch("synth_panel._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
            patch("synth_panel.mcp.server._shared_client", None),
            patch("synth_panel.mcp.server.LLMClient", MagicMock()),
        ):
            raw = await _server.run_quick_poll(
                question="What would you pay?",
                personas=list(_PERSONAS),
                model="haiku",
                max_cost=5.0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        data = json.loads(raw)
        assert data["run_invalid"] is True
        assert data["cost_exceeded"] is True
        assert data["abort_reason"] == "cost_exceeded"
        assert data["cost_gate"]["halted"] is True
        assert data["resume"]["remaining_personas"] == ["Bob", "Cara"]


# ---------------------------------------------------------------------------
# extend_panel
# ---------------------------------------------------------------------------


def _extend_fake_parallel(per_panelist_cost: float, captured: dict):
    from synth_panel.cost import TokenUsage
    from synth_panel.orchestrator import PanelistResult

    def _fake(client=None, personas=None, questions=None, model=None, sessions=None, cost_gate=None, **kwargs):
        captured["cost_gate"] = cost_gate
        results = []
        for p in personas or []:
            if cost_gate is not None and cost_gate.should_halt():
                break
            results.append(
                PanelistResult(
                    persona_name=p.get("name", "anon"),
                    responses=[{"question": "follow-up?", "response": "sure"}],
                    usage=TokenUsage(input_tokens=5, output_tokens=3),
                    model=model,
                )
            )
            if cost_gate is not None:
                cost_gate.record(per_panelist_cost)
        return results, {}, dict(sessions or {})

    return _fake


class TestExtendPanelMaxCost:
    async def _call(self, max_cost, per_panelist_cost=10.0):
        from synth_panel.mcp import server as _server

        captured: dict = {}
        fake_existing = {"rounds": [], "path": [], "question_count": 0}
        fake_sessions = {"Alice": object(), "Bob": object(), "Cara": object()}
        synth_mock = MagicMock(side_effect=_stub_synthesize_panel)
        with (
            patch("synth_panel.mcp.server._data_get_panel_result", return_value=fake_existing),
            patch("synth_panel.mcp.server.load_panel_sessions", return_value=fake_sessions),
            patch(
                "synth_panel.mcp.server.run_panel_parallel",
                side_effect=_extend_fake_parallel(per_panelist_cost, captured),
            ),
            patch("synth_panel.mcp.server.synthesize_panel", synth_mock),
            patch("synth_panel.mcp.server.update_panel_result"),
            patch("synth_panel.mcp.server._get_shared_client", return_value=object()),
        ):
            raw = await _server.extend_panel(
                result_id="r-576",
                questions=[{"text": "follow-up?"}],
                model="haiku",
                max_cost=max_cost,
                decision_being_informed=_VALID_DECISION,
                ctx=None,
            )
        return json.loads(raw), captured, synth_mock

    @pytest.mark.asyncio
    async def test_gate_plumbed_and_no_trip(self):
        from synth_panel.cost import CostGate

        data, captured, _synth = await self._call(max_cost=100.0, per_panelist_cost=0.01)
        gate = captured["cost_gate"]
        assert isinstance(gate, CostGate)
        assert gate.max_cost_usd == 100.0
        assert gate.total_panelists == 3
        assert "cost_exceeded" not in data

    @pytest.mark.asyncio
    async def test_trip_envelope_and_synthesis_skip(self):
        data, _captured, synth_mock = await self._call(max_cost=5.0, per_panelist_cost=10.0)
        assert data["run_invalid"] is True
        assert data["cost_exceeded"] is True
        assert data["abort_reason"] == "cost_exceeded"
        assert data["halted_at_panelist"] == 1
        assert data["cost_gate"]["halted"] is True
        assert data["resume"]["partial_result_id"] == "r-576"
        assert data["resume"]["remaining_personas"] == ["Bob", "Cara"]
        synth_mock.assert_not_called()
        assert data.get("synthesis") is None
        # The partial round is still appended/returned.
        assert [r["persona"] for r in data["results"]] == ["Alice"]

    @pytest.mark.asyncio
    async def test_no_max_cost_passes_no_gate(self):
        data, captured, _synth = await self._call(max_cost=None, per_panelist_cost=10.0)
        assert captured["cost_gate"] is None
        assert "cost_exceeded" not in data


# ---------------------------------------------------------------------------
# Boundary validation and unsupported combinations
# ---------------------------------------------------------------------------


class TestMaxCostValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [0, -1, -0.5])
    async def test_non_positive_rejected_run_panel(self, bad):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.run_panel(
                personas=list(_PERSONAS),
                questions=list(_QUESTIONS),
                max_cost=bad,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"

    @pytest.mark.asyncio
    async def test_non_positive_rejected_run_quick_poll(self):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.run_quick_poll(
                question="Q?",
                max_cost=0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"

    @pytest.mark.asyncio
    async def test_non_positive_rejected_extend_panel(self):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.extend_panel(
                result_id="r-576",
                questions=[{"text": "q?"}],
                max_cost=-3,
                decision_being_informed=_VALID_DECISION,
                ctx=None,
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"

    @pytest.mark.asyncio
    async def test_rejected_with_ensemble_models(self):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.run_panel(
                personas=list(_PERSONAS),
                questions=list(_QUESTIONS),
                models=["haiku", "gpt-4o-mini"],
                max_cost=5.0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"
        assert "ensemble" in data["message"]

    @pytest.mark.asyncio
    async def test_rejected_with_instrument(self):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.run_panel(
                personas=list(_PERSONAS),
                instrument={"questions": [{"text": "q?"}]},
                max_cost=5.0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"
        assert "instrument" in data["message"]

    @pytest.mark.asyncio
    async def test_rejected_with_variants(self):
        from synth_panel.mcp import server as _server

        data = json.loads(
            await _server.run_panel(
                personas=list(_PERSONAS),
                questions=list(_QUESTIONS),
                variants=2,
                max_cost=5.0,
                decision_being_informed=_VALID_DECISION,
                ctx=_StubMcpContext(),
            )
        )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"
        assert "variants" in data["message"]

    @pytest.mark.asyncio
    async def test_rejected_in_sampling_mode_run_panel(self):
        from synth_panel.mcp import server as _server
        from synth_panel.mcp.sampling import SamplingDecision

        with patch(
            "synth_panel.mcp.server._decide_sampling_mode",
            return_value=SamplingDecision(mode="sampling"),
        ):
            data = json.loads(
                await _server.run_panel(
                    personas=list(_PERSONAS),
                    questions=list(_QUESTIONS),
                    max_cost=5.0,
                    decision_being_informed=_VALID_DECISION,
                    ctx=_StubMcpContext(),
                )
            )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"
        assert "sampling" in data["message"]

    @pytest.mark.asyncio
    async def test_rejected_in_sampling_mode_run_quick_poll(self):
        from synth_panel.mcp import server as _server
        from synth_panel.mcp.sampling import SamplingDecision

        with patch(
            "synth_panel.mcp.server._decide_sampling_mode",
            return_value=SamplingDecision(mode="sampling"),
        ):
            data = json.loads(
                await _server.run_quick_poll(
                    question="Q?",
                    personas=list(_PERSONAS),
                    max_cost=5.0,
                    decision_being_informed=_VALID_DECISION,
                    ctx=_StubMcpContext(),
                )
            )
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "max_cost"
        assert "sampling" in data["message"]
