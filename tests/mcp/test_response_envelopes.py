"""Response-envelope shape for the panel-running MCP tools.

Covers the compact-by-default + typed-envelope work:

* ``detail="summary"`` (the run-tool default) drops the per-panelist
  transcripts while keeping synthesis / panel_verdict / poll_summary /
  metadata / costs; ``detail="full"`` returns them.
* ``per_model_results`` never carries a second copy of the transcript
  (the canonical copy is ``rounds[].results``).
* The flat-questions BYOK path emits the same top-level ``results`` +
  ``terminal_round`` keys the instrument/sampling paths do.
* ``run_quick_poll`` grows a ``pack_id`` and returns typed envelopes on
  an unknown pack or a total model failure instead of raising.
* ``extend_panel`` returns a typed ``INVALID_TOOL_ARG`` for an unknown
  ``result_id`` instead of a raw ``FileNotFoundError``.

Only the LLM boundary (``run_panel_parallel`` / ``synthesize_panel``) is
stubbed; the MCP tool functions run for real.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")
    monkeypatch.delenv("SYNTHPANEL_SCHEMA_MIN", raising=False)
    monkeypatch.delenv("SYNTHPANEL_DRIFT_DEGRADE", raising=False)


from .test_decision_wiring import (
    _stub_run_panel_parallel_with_sessions,
    _stub_synthesize_panel,
    _StubMcpContext,
)

_VALID_DECISION = "Should we ship the new pricing tier next quarter?"


def _stub_all_failed_panelists(error: str = "model 'bogus-alias-xyz' not found (HTTP 400)"):
    """Stub ``run_panel_parallel`` returning an all-failed panel.

    Every persona comes back with ``error`` set and no clean response, so
    :func:`synth_panel._runners.detect_total_failure` classifies the run as
    a total wipeout — the exact shape a knowingly-bad model alias produces.
    """
    from synth_panel.cost import TokenUsage
    from synth_panel.orchestrator import PanelistResult

    def _fake(client=None, personas=None, questions=None, model=None, sessions=None, **_kwargs):
        results = [
            PanelistResult(
                persona_name=p.get("name", "anon"),
                responses=[],
                usage=TokenUsage(),
                error=error,
                model=model,
            )
            for p in (personas or [])
        ]
        return results, {}, dict(sessions or {})

    return _fake


async def _run_panel(detail: str | None = None, **extra):
    from synth_panel.mcp import server as _server

    kwargs: dict = {
        "personas": [{"name": "Alice"}, {"name": "Bob"}],
        "questions": [{"text": "What would you pay?"}],
        "model": "haiku",
        "decision_being_informed": _VALID_DECISION,
        "ctx": _StubMcpContext(),
    }
    if detail is not None:
        kwargs["detail"] = detail
    kwargs.update(extra)

    with (
        patch(
            "synth_panel.orchestrator.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch(
            "synth_panel._runners.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch("synth_panel._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("synth_panel.mcp.server._shared_client", None),
    ):
        raw = await _server.run_panel(**kwargs)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# detail=summary vs full
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_is_the_default_and_omits_transcripts():
    data = await _run_panel()  # no detail arg → summary default

    assert data["detail"] == "summary"
    # The bulky per-panelist transcript is gone from both surfaces.
    assert "results" not in data
    assert data["results_omitted"] is True
    for rd in data["rounds"]:
        assert "results" not in rd
        assert "result_count" in rd
    # But the decision-grade signal is all retained.
    assert data["synthesis"] is not None
    assert data["panel_verdict"]["meta"]["decision_being_informed"] == _VALID_DECISION
    assert "poll_summary" in data
    assert data["metadata"] is not None
    assert data["result_id"]
    assert data["total_cost"]
    # And the omission is self-describing — the caller is told where to look.
    assert data["transcript_uri"] == f"panel-result://{data['result_id']}"


@pytest.mark.asyncio
async def test_full_detail_keeps_every_transcript_row():
    data = await _run_panel(detail="full")

    assert data.get("detail") != "summary"
    assert "results_omitted" not in data
    # Top-level mirror + per-round transcript both present and populated.
    assert len(data["results"]) == 2
    assert data["rounds"][0]["results"], "full detail keeps rounds[].results"
    assert data["results"][0]["persona"] == "Alice"


@pytest.mark.asyncio
async def test_invalid_detail_value_returns_typed_envelope():
    data = await _run_panel(detail="verbose")
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert data["field_path"] == "detail"


# ---------------------------------------------------------------------------
# No transcript duplication in per_model_results (regardless of detail)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_model_results_carries_no_duplicate_transcript():
    data = await _run_panel(detail="full")

    per_model = data["per_model_results"]
    assert per_model, "single-model panels still emit a one-entry rollup"
    for model_name, entry in per_model.items():
        # usage/cost breakdown is retained...
        assert "usage" in entry
        assert "cost" in entry
        # ...but the full panelist transcript is NOT copied here.
        assert "results" not in entry, f"{model_name} must not duplicate the transcript"
        # A cheap reference replaces it.
        assert entry["result_count"] == 2
        assert entry["personas"] == ["Alice", "Bob"]


# ---------------------------------------------------------------------------
# Envelope uniformity: flat-questions path has results + terminal_round
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flat_questions_path_has_results_and_terminal_round():
    data = await _run_panel(detail="full")

    assert data["terminal_round"] == "default"
    assert isinstance(data["results"], list) and data["results"]


@pytest.mark.asyncio
async def test_flat_questions_summary_still_reports_terminal_round():
    data = await _run_panel()  # summary

    # terminal_round survives summary; only the transcript is dropped.
    assert data["terminal_round"] == "default"
    assert "results" not in data


# ---------------------------------------------------------------------------
# run_quick_poll: pack_id + typed envelopes
# ---------------------------------------------------------------------------


async def _run_quick_poll(**kwargs):
    from synth_panel.mcp import server as _server

    kwargs.setdefault("decision_being_informed", _VALID_DECISION)
    kwargs.setdefault("ctx", _StubMcpContext())
    with (
        patch(
            "synth_panel.orchestrator.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch(
            "synth_panel._runners.run_panel_parallel",
            side_effect=_stub_run_panel_parallel_with_sessions(),
        ),
        patch("synth_panel._runners.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("synth_panel.mcp.server._shared_client", None),
    ):
        raw = await _server.run_quick_poll(**kwargs)
    return json.loads(raw)


@pytest.mark.asyncio
async def test_quick_poll_pack_id_resolves_personas():
    from synth_panel.mcp.data import save_persona_pack

    save_persona_pack(
        "Trio",
        [{"name": "Pat"}, {"name": "Quinn"}, {"name": "Rae"}],
        pack_id="trio",
    )
    data = await _run_quick_poll(question="Is this clear?", pack_id="trio", model="haiku")

    assert "error_code" not in data
    assert data["mode"] == "byok"
    # The three pack personas ran (not the 3-persona built-in default —
    # verify by name via the persisted transcript).
    assert data["persona_count"] == 3
    from synth_panel.mcp.data import get_panel_result

    saved = get_panel_result(data["result_id"])
    assert sorted(r["persona"] for r in saved["results"]) == ["Pat", "Quinn", "Rae"]


@pytest.mark.asyncio
async def test_quick_poll_pack_id_merges_with_inline_personas():
    from synth_panel.mcp.data import save_persona_pack

    save_persona_pack("One", [{"name": "Zed"}], pack_id="one")
    data = await _run_quick_poll(
        question="Thoughts?",
        personas=[{"name": "Ada"}],
        pack_id="one",
        model="haiku",
    )
    assert data["persona_count"] == 2  # inline + pack


@pytest.mark.asyncio
async def test_quick_poll_unknown_pack_id_returns_typed_envelope():
    data = await _run_quick_poll(question="Hi?", pack_id="does-not-exist", model="haiku")
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert data["field_path"] == "pack_id"
    assert "does-not-exist" in data["message"]


@pytest.mark.asyncio
async def test_quick_poll_bad_alias_returns_typed_envelope_not_exception():
    from synth_panel.mcp import server as _server

    with (
        patch(
            "synth_panel._runners.run_panel_parallel",
            side_effect=_stub_all_failed_panelists(),
        ),
        patch("synth_panel.mcp.server._shared_client", None),
    ):
        raw = await _server.run_quick_poll(
            question="What would you pay?",
            personas=[{"name": "Alice"}],
            model="bogus-alias-xyz",
            decision_being_informed=_VALID_DECISION,
            ctx=_StubMcpContext(),
        )
    data = json.loads(raw)

    # Typed total-failure envelope, NOT a raw FastMCP "Error executing tool".
    assert data["run_invalid"] is True
    assert "total_failure" in data
    assert "bogus-alias-xyz" in data["error"]


@pytest.mark.asyncio
async def test_quick_poll_byok_default_is_summary():
    data = await _run_quick_poll(question="Clear?", personas=[{"name": "Alice"}], model="haiku")
    assert data["detail"] == "summary"
    assert "results" not in data
    assert data["synthesis"] is not None


# ---------------------------------------------------------------------------
# extend_panel: typed envelope for unknown result_id + bad alias
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_panel_unknown_result_id_returns_typed_envelope():
    from synth_panel.mcp import server as _server

    raw = await _server.extend_panel(
        result_id="result-does-not-exist",
        questions=[{"text": "And now?"}],
        model="haiku",
        decision_being_informed=_VALID_DECISION,
        ctx=None,
    )
    data = json.loads(raw)
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert data["field_path"] == "result_id"
    assert "result-does-not-exist" in data["message"]


@pytest.mark.asyncio
async def test_extend_panel_bad_alias_returns_typed_envelope():
    # First create a real saved result with persisted sessions.
    first = await _run_panel()
    result_id = first["result_id"]

    from synth_panel.mcp import server as _server

    with (
        patch(
            "synth_panel.mcp.server.run_panel_parallel",
            side_effect=_stub_all_failed_panelists(),
        ),
        patch("synth_panel.mcp.server.synthesize_panel", side_effect=_stub_synthesize_panel),
        patch("synth_panel.mcp.server._shared_client", None),
    ):
        raw = await _server.extend_panel(
            result_id=result_id,
            questions=[{"text": "And the copy?"}],
            model="bogus-alias-xyz",
            decision_being_informed="Should the follow-up change our copy?",
            ctx=None,
        )
    data = json.loads(raw)
    assert data["run_invalid"] is True
    assert "total_failure" in data


# ---------------------------------------------------------------------------
# get_panel_result honors detail (default full for back-compat)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_panel_result_defaults_to_full_and_can_summarize():
    from synth_panel.mcp import server as _server

    first = await _run_panel(detail="full")
    result_id = first["result_id"]

    # Default (full): the saved transcript is present.
    full_raw = await _server.get_panel_result(result_id=result_id)
    full = json.loads(full_raw)
    assert full["results"], "get_panel_result defaults to full for back-compat"

    # detail=summary: transcript dropped, pointer left behind.
    summ_raw = await _server.get_panel_result(result_id=result_id, detail="summary")
    summ = json.loads(summ_raw)
    assert "results" not in summ
    assert summ["results_omitted"] is True
    assert summ["transcript_uri"] == f"panel-result://{result_id}"


@pytest.mark.asyncio
async def test_get_panel_result_rejects_bad_detail():
    from synth_panel.mcp import server as _server

    raw = await _server.get_panel_result(result_id="whatever", detail="loud")
    data = json.loads(raw)
    assert data["error_code"] == "INVALID_TOOL_ARG"
    assert data["field_path"] == "detail"
