"""GH#562: MCP ``run_panel`` instrument template vars + placeholder fail-fast.

The CLI has had ``--var`` / ``--vars-file`` substitution plus an
unsubstituted-placeholder guard (sp-6yi) since 0.9.5; the MCP path had
neither, so ``run_panel(instrument_pack="pricing-discovery")`` sent literal
``{problem}`` to panelists and silently corrupted results. These tests pin
the MCP parity surface:

* ``vars`` values are substituted into the panelist-visible question text
  (pack and inline ``instrument`` inputs alike);
* leftover ``{placeholder}`` tokens — including when ``vars`` is omitted
  entirely — return a typed ``INVALID_TOOL_ARG`` envelope naming the
  missing keys (the core behavior change: previously such calls
  "succeeded");
* instruments without placeholders are unaffected;
* ``vars`` with plain ``questions`` input is a typed error.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("mcp")


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point data dir at temp; fake BYOK creds so the BYOK path runs."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-placeholder")


from althing.mcp.server import mcp

_TEMPLATED_INSTRUMENT = {
    "version": 1,
    "questions": [
        {"text": "Walk me through the last time you tried to solve {problem}."},
        {"text": "Which of these fits best: {candidates}?"},
    ],
}

_PLAIN_INSTRUMENT = {
    "version": 1,
    "questions": [{"text": "How do you feel about subscriptions?"}],
}


def _save_pack(name: str, instrument: dict) -> None:
    from althing.mcp.data import save_instrument_pack as _save

    _save(name, {"name": name, "instrument": instrument})


def _text(result) -> str:
    return result.content[0].text


async def _call_run_panel(**kwargs):
    args = {"personas": [{"name": "A"}], "decision_being_informed": "testing vars substitution"}
    args.update(kwargs)
    return await mcp.call_tool("run_panel", args)


def _question_texts(instrument_arg) -> list[str]:
    return [q["text"] for rnd in instrument_arg.rounds for q in rnd.questions]


class TestVarsSubstitution:
    @pytest.mark.asyncio
    async def test_vars_substituted_into_pack_question_text(self):
        """End-to-end through the pack path: the Instrument handed to the
        panel runner carries the substituted, panelist-visible text."""
        _save_pack("templated", _TEMPLATED_INSTRUMENT)
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await _call_run_panel(
                instrument_pack="templated",
                vars={"problem": "tracking cloud spend", "candidates": "Wander, Roamly"},
            )
            assert mock_run.called
            texts = _question_texts(mock_run.call_args[0][1])
            assert texts[0] == "Walk me through the last time you tried to solve tracking cloud spend."
            assert texts[1] == "Which of these fits best: Wander, Roamly?"
            assert not any("{" in t for t in texts)

    @pytest.mark.asyncio
    async def test_vars_substituted_into_inline_instrument(self):
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await _call_run_panel(
                instrument=_TEMPLATED_INSTRUMENT,
                vars={"problem": "expense reports", "candidates": "Core, Plus"},
            )
            texts = _question_texts(mock_run.call_args[0][1])
            assert "expense reports" in texts[0]
            assert "Core, Plus" in texts[1]

    @pytest.mark.asyncio
    async def test_bundled_pricing_discovery_pack_substitutes(self):
        """The real bundled pack that motivated GH#562 renders cleanly."""
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await _call_run_panel(
                instrument_pack="pricing-discovery",
                vars={"problem": "sharing large design files"},
            )
            assert mock_run.called
            texts = _question_texts(mock_run.call_args[0][1])
            assert any("sharing large design files" in t for t in texts)
            assert not any("{problem}" in t for t in texts)


class TestPlaceholderFailFast:
    @pytest.mark.asyncio
    async def test_pack_without_vars_returns_typed_error(self):
        """The core GH#562 fix: a placeholder-bearing pack with no vars is a
        typed error, not a 'successful' run with literal {problem} text."""
        _save_pack("templated", _TEMPLATED_INSTRUMENT)
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            result = await _call_run_panel(instrument_pack="templated")
            assert not mock_run.called
        data = json.loads(_text(result))
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "vars"
        assert data["retry_safe"] is False
        # Names every missing placeholder and shows an example vars payload.
        assert "candidates" in data["message"]
        assert "problem" in data["message"]
        assert "vars=" in data["message"]

    @pytest.mark.asyncio
    async def test_partial_vars_names_only_missing_keys(self):
        _save_pack("templated", _TEMPLATED_INSTRUMENT)
        result = await _call_run_panel(
            instrument_pack="templated",
            vars={"problem": "expense reports"},
        )
        data = json.loads(_text(result))
        assert data["error_code"] == "INVALID_TOOL_ARG"
        # Only the still-missing key is named; the supplied one is not.
        assert "candidates" in data["message"]
        assert "problem" not in data["message"]

    @pytest.mark.asyncio
    async def test_inline_instrument_without_vars_returns_typed_error(self):
        result = await _call_run_panel(instrument=_TEMPLATED_INSTRUMENT)
        data = json.loads(_text(result))
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert "problem" in data["message"]

    @pytest.mark.asyncio
    async def test_vars_with_plain_questions_is_typed_error(self):
        result = await _call_run_panel(
            questions=[{"text": "Do you like {thing}?"}],
            vars={"thing": "surveys"},
        )
        data = json.loads(_text(result))
        assert data["error_code"] == "INVALID_TOOL_ARG"
        assert data["field_path"] == "vars"


class TestNoPlaceholderInstrumentsUnaffected:
    @pytest.mark.asyncio
    async def test_plain_pack_runs_without_vars(self):
        _save_pack("plain", _PLAIN_INSTRUMENT)
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await _call_run_panel(instrument_pack="plain")
            assert mock_run.called
            texts = _question_texts(mock_run.call_args[0][1])
            assert texts == ["How do you feel about subscriptions?"]

    @pytest.mark.asyncio
    async def test_plain_pack_ignores_extra_vars(self):
        """Extra keys are harmless — parity with the CLI, where --var
        entries that match no placeholder are simply unused."""
        _save_pack("plain", _PLAIN_INSTRUMENT)
        with patch(
            "althing.mcp.server._run_panel_async_instrument",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"rounds": [], "path": [], "warnings": []}
            await _call_run_panel(instrument_pack="plain", vars={"unused": "x"})
            assert mock_run.called
            texts = _question_texts(mock_run.call_args[0][1])
            assert texts == ["How do you feel about subscriptions?"]

    @pytest.mark.asyncio
    async def test_plain_questions_without_vars_unaffected(self):
        with patch(
            "althing.mcp.server._run_panel_async",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = {"results": [], "warnings": []}
            await _call_run_panel(questions=[{"text": "Plain question?"}])
            assert mock_run.called
