"""P1-1 wiring: ``decision_being_informed`` flows through the whole signal path.

Covers the previously-dormant v1.0.0 contract plumbing:

* AC-4 grace — omitting the field synthesizes ``"unspecified-legacy-call"``
  and surfaces a ``W_DECISION_MISSING`` nudge in the response ``warnings[]``;
  ``ALTHING_SCHEMA_MIN>=1.1.0`` flips omission into a typed
  ``MISSING_DECISION`` reject.
* AC-7 — the decision is persisted in the saved result JSON and stamped on
  the persisted panelist sessions (and their JSONL transcript rows).
* AC-6 — the decision is echoed verbatim at
  ``panel_verdict.meta.decision_being_informed`` in the response envelope.

These tests run the real ``_run_panel_async`` path (only the LLM boundary —
``run_panel_parallel`` / ``synthesize_panel`` — is stubbed) so the wiring is
exercised end to end through the MCP tool functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")
    monkeypatch.delenv("ALTHING_SCHEMA_MIN", raising=False)
    monkeypatch.delenv("ALTHING_DRIFT_DEGRADE", raising=False)


from althing.mcp.compat import LEGACY_DECISION_PLACEHOLDER, W_DECISION_MISSING
from althing.structured.validate import validate_response

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"


class _StubMcpContext:
    async def report_progress(self, *_args, **_kwargs):
        return None


def _stub_run_panel_parallel_with_sessions(**response_overrides):
    """Stub for ``run_panel_parallel`` producing real Session objects.

    Unlike the sentinel-based stub in test_mcp_server.py, this returns
    genuine :class:`~althing.persistence.Session` instances so the
    AC-7 stamping/persistence path is exercised for real.
    """
    from althing.cost import TokenUsage
    from althing.orchestrator import PanelistResult
    from althing.persistence import ConversationMessage, Session

    def _fake(
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
        attachment_bank=None,
        allow_empty_attachments=False,
    ):
        results = []
        out_sessions = dict(sessions or {})
        for p in personas:
            name = p.get("name", "anon")
            resp: dict = {"question": questions[0].get("text", ""), "response": "I would pay $49."}
            resp.update(response_overrides)
            results.append(
                PanelistResult(
                    persona_name=name,
                    responses=[resp],
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
        return results, {}, out_sessions

    return _fake


def _stub_synthesize_panel(*_args, **_kwargs):
    from althing.cost import TokenUsage as CostTokenUsage
    from althing.synthesis import SynthesisResult

    return SynthesisResult(
        summary="Panel leans positive on the new tier.",
        themes=["pricing"],
        agreements=[],
        disagreements=[],
        surprises=[],
        recommendation="Ship the $49 tier.",
        usage=CostTokenUsage(input_tokens=2, output_tokens=1),
        model="stub-model",
    )


async def _run_panel_tool(decision: str | None, **extra):
    from althing.mcp import server as _server

    kwargs: dict = {
        "personas": [{"name": "Alice"}, {"name": "Bob"}],
        "questions": [{"text": "What would you pay?"}],
        "model": "haiku",
        "ctx": _StubMcpContext(),
    }
    if decision is not None:
        kwargs["decision_being_informed"] = decision
    kwargs.update(extra)

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
        raw = await _server.run_panel(**kwargs)
    return json.loads(raw)


def _saved_result(tmp_path: Path, result_id: str) -> dict:
    return json.loads((tmp_path / "results" / f"{result_id}.json").read_text(encoding="utf-8"))


def _saved_sessions(tmp_path: Path, result_id: str) -> list[dict]:
    sdir = tmp_path / "results" / f"{result_id}.sessions"
    assert sdir.is_dir(), f"expected persisted sessions at {sdir}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(sdir.glob("*.json"))]


# ---------------------------------------------------------------------------
# AC-4 grace: omitted decision → placeholder + warnings[] nudge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_omitted_decision_synthesizes_placeholder_and_warns(tmp_path):
    data = await _run_panel_tool(decision=None)

    assert "error_code" not in data
    # The nudge rides in the response warnings.
    assert any(W_DECISION_MISSING in w for w in data["warnings"])
    # The placeholder is echoed verbatim at the contract location.
    assert data["panel_verdict"]["meta"]["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER
    # And persisted in the saved result JSON.
    saved = _saved_result(tmp_path, data["result_id"])
    assert saved["decision_being_informed"] == LEGACY_DECISION_PLACEHOLDER


@pytest.mark.asyncio
async def test_schema_min_1_1_0_hard_rejects_omitted_decision(monkeypatch):
    monkeypatch.setenv("ALTHING_SCHEMA_MIN", "1.1.0")
    data = await _run_panel_tool(decision=None)

    assert data["error_code"] == "MISSING_DECISION"
    assert data["field_path"] == "decision_being_informed"
    assert data["schema_version"] == "1.0.0"
    assert data["retry_safe"] is False


@pytest.mark.asyncio
async def test_supplied_decision_produces_no_grace_warning(tmp_path):
    data = await _run_panel_tool(decision=_VALID_DECISION)

    assert not any(W_DECISION_MISSING in w for w in data.get("warnings", []))


# ---------------------------------------------------------------------------
# AC-7: persistence — saved result JSON, session stamp, JSONL rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplied_decision_persisted_in_saved_result(tmp_path):
    data = await _run_panel_tool(decision=_VALID_DECISION)

    saved = _saved_result(tmp_path, data["result_id"])
    assert saved["decision_being_informed"] == _VALID_DECISION


@pytest.mark.asyncio
async def test_supplied_decision_stamped_on_persisted_sessions(tmp_path):
    data = await _run_panel_tool(decision=_VALID_DECISION)

    session_dicts = _saved_sessions(tmp_path, data["result_id"])
    assert len(session_dicts) == 2
    for sd in session_dicts:
        assert sd["decision_being_informed"] == _VALID_DECISION


@pytest.mark.asyncio
async def test_stamped_session_jsonl_rows_carry_decision(tmp_path):
    """Every JSONL transcript row is self-describing (persistence AC-7)."""
    from althing.persistence import Session

    data = await _run_panel_tool(decision=_VALID_DECISION)

    session_dicts = _saved_sessions(tmp_path, data["result_id"])
    sess = Session.from_dict(session_dicts[0])
    rows = [json.loads(line) for line in sess.to_jsonl().strip().splitlines()]
    assert rows, "expected at least the session_meta row"
    for row in rows:
        assert row["decision_being_informed"] == _VALID_DECISION


# ---------------------------------------------------------------------------
# AC-6: response-envelope echo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supplied_decision_echoed_in_verdict_meta():
    data = await _run_panel_tool(decision=_VALID_DECISION)

    verdict = data["panel_verdict"]
    assert verdict["meta"]["decision_being_informed"] == _VALID_DECISION
    assert verdict["schema_version"] == "1.0.0"
    assert data["schema_version"] == "1.0.0"
    assert validate_response(verdict) is None


@pytest.mark.asyncio
async def test_quick_poll_byok_carries_contract_fields(tmp_path):
    from althing.mcp import server as _server

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
        raw = await _server.run_quick_poll(
            question="What would you pay?",
            personas=[{"name": "Alice"}],
            model="haiku",
            decision_being_informed=_VALID_DECISION,
            ctx=_StubMcpContext(),
        )
    data = json.loads(raw)

    assert data["mode"] == "byok"
    assert data["schema_version"] == "1.0.0"
    assert data["panel_verdict"]["meta"]["decision_being_informed"] == _VALID_DECISION
    saved = _saved_result(tmp_path, data["result_id"])
    assert saved["decision_being_informed"] == _VALID_DECISION


# ---------------------------------------------------------------------------
# extend_panel: full loop against sessions persisted by run_panel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_panel_records_extension_decision(tmp_path):
    from althing.mcp import server as _server

    first = await _run_panel_tool(decision=_VALID_DECISION)
    result_id = first["result_id"]

    extension_decision = "Should the follow-up round change our launch copy?"
    with (
        patch(
            "althing.mcp.server.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch("althing.mcp.server.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("althing.mcp.server._shared_client", None),
    ):
        raw = await _server.extend_panel(
            result_id=result_id,
            questions=[{"text": "And the copy?"}],
            model="haiku",
            decision_being_informed=extension_decision,
            ctx=None,
        )
    data = json.loads(raw)

    assert data["schema_version"] == "1.0.0"
    assert data["panel_verdict"]["meta"]["decision_being_informed"] == extension_decision
    assert validate_response(data["panel_verdict"]) is None

    # The extension round records its own decision; the original result's
    # top-level decision keeps describing the original run.
    saved = _saved_result(tmp_path, result_id)
    assert saved["decision_being_informed"] == _VALID_DECISION
    assert saved["rounds"][-1]["decision_being_informed"] == extension_decision

    # Sessions are re-persisted with the freshest decision stamp.
    for sd in _saved_sessions(tmp_path, result_id):
        assert sd["decision_being_informed"] == extension_decision
