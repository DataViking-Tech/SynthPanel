"""sp-avmm: synthesis failure must fail loud, not silently return success.

Covers the three call sites the bead calls out:

* ``cmd_panel_run`` (non-ensemble CLI path) — previously caught the
  exception, logged a WARN, then emitted a result with ``synthesis: null``
  and exit 0. Now: pre-flight context check + structured ``synthesis_error``
  + ``run_invalid: true`` + exit 2.
* ``handle_panel_synthesize`` (re-synthesize saved result) — previously
  exited 1 with a bare stderr line. Now: exit 2 and a structured envelope
  in JSON/NDJSON mode so MCP/CI consumers see the same shape.
* ``run_panel_sync`` (MCP/SDK sync runner) — previously set
  ``{"synthesis_error": "Synthesis failed — see logs for details."}``
  without flagging ``run_invalid``. Now: structured payload and, via the
  server/sdk envelope, ``run_invalid=True``.
"""

from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.main import main

# --- helpers --------------------------------------------------------------


def _write_personas(path, n: int = 2) -> str:
    names = [f"Panelist{i}" for i in range(n)]
    body = "personas:\n" + "\n".join(f"  - name: {name}" for name in names) + "\n"
    path.write_text(body)
    return str(path)


def _write_instrument(path, question: str = "Q1") -> str:
    path.write_text(f"instrument:\n  questions:\n    - text: {question}\n")
    return str(path)


def _panelist_results(n: int = 2, *, answer: str = "A clean answer"):
    from synth_panel.orchestrator import PanelistResult

    return [
        PanelistResult(
            persona_name=f"Panelist{i}",
            responses=[{"question": "Q1", "response": answer}],
            usage=ZERO_USAGE,
            model="sonnet",
        )
        for i in range(n)
    ]


# --- sp-avmm: CLI non-ensemble path ---------------------------------------


class TestCliNonEnsembleSynthesisFailure:
    """commands.py:1113-1126 — synthesis exception must not silently skip."""

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_synthesis_api_error_sets_run_invalid(self, _mock_client, mock_run, mock_synth, capsys, tmp_path):
        """When synthesize_panel raises an API error, the run must exit 2,
        set run_invalid=True, and carry a structured synthesis_error payload.
        """
        from synth_panel.orchestrator import WorkerRegistry

        mock_run.return_value = (_panelist_results(2), WorkerRegistry(), {})
        mock_synth.side_effect = RuntimeError("Anthropic 400: prompt too long (262000 > 200000)")

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
            ]
        )
        assert code == 2, "sp-avmm requires exit code 2 on synthesis failure"
        captured = capsys.readouterr()
        assert "synthesis call failed" in captured.err.lower()
        assert "262000" in captured.err or "262k" in captured.err or "200000" in captured.err

        payload = json.loads(captured.out)
        assert payload["run_invalid"] is True
        err = payload["synthesis_error"]
        assert isinstance(err, dict), "synthesis_error must be a structured dict"
        assert err["error_type"] == "synthesis_api_error"
        assert "message" in err
        assert "suggested_fix" in err
        mock_synth.assert_called_once()

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_preflight_rejects_oversized_input(self, _mock_client, mock_run, mock_synth, capsys, tmp_path):
        """When the estimated prompt overflows the synthesis model's context,
        the run must fail BEFORE synthesize_panel is called.

        sp-exu6: must pass ``--synthesis-strategy=single`` explicitly.
        Under ``auto`` (the new default behavior after sp-exu6), overflow
        would instead route to map-reduce."""
        from synth_panel.orchestrator import PanelistResult, WorkerRegistry

        # Build one giant panelist response that comfortably overflows
        # haiku's 200k (with 8k headroom). 800_000 chars ≈ 200k tokens.
        giant = "A" * 800_000
        huge_results = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[{"question": "Q1", "response": giant}],
                usage=ZERO_USAGE,
                model="haiku",
            )
            for i in range(2)
        ]
        mock_run.return_value = (huge_results, WorkerRegistry(), {})

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
                "--synthesis-model",
                "haiku",
                "--synthesis-strategy",
                "single",
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "pre-flight" in captured.err.lower() or "exceeds" in captured.err.lower()
        mock_synth.assert_not_called(), "pre-flight must short-circuit BEFORE the API call"

        payload = json.loads(captured.out)
        assert payload["run_invalid"] is True
        err = payload["synthesis_error"]
        assert err["error_type"] == "synthesis_context_overflow"
        diag = err["diagnostic"]
        assert diag["context_window"] == 200_000
        assert diag["estimated_tokens"] > diag["effective_limit"]


# --- sp-avmm: handle_panel_synthesize (re-synthesize) path ----------------


class TestResynthesizeSynthesisFailure:
    """commands.py:2305 — panel synthesize must fail loud too."""

    @patch("synth_panel.cli.commands.synthesize_panel")
    def test_resynthesize_api_error_emits_structured_payload(self, mock_synth, capsys, tmp_path):
        from synth_panel.mcp.data import save_panel_result

        mock_synth.side_effect = RuntimeError("provider 429: rate-limited")

        saved = save_panel_result(
            results=[
                {
                    "persona": "Alice",
                    "responses": [{"question": "Q1", "response": "An answer"}],
                    "usage": ZERO_USAGE.to_dict(),
                    "cost": "$0.00",
                    "error": None,
                }
            ],
            model="sonnet",
            total_usage=ZERO_USAGE.to_dict(),
            total_cost="$0.00",
            persona_count=1,
            question_count=1,
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "synthesize",
                saved,
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["run_invalid"] is True
        err = payload["synthesis_error"]
        assert err["error_type"] == "synthesis_api_error"
        assert "suggested_fix" in err


# --- sp-avmm: run_panel_sync (MCP/SDK) ------------------------------------


class TestRunPanelSyncSynthesisFailure:
    """_runners.py:420-435 — synthesis_dict gains structured payload and the
    MCP server surfaces run_invalid=True at the envelope top-level."""

    @patch("synth_panel._runners.synthesize_panel")
    @patch("synth_panel._runners.run_panel_parallel")
    def test_synthesis_api_error_returns_structured_payload(self, mock_run, mock_synth):
        from synth_panel._runners import run_panel_sync

        mock_run.return_value = (_panelist_results(2), None, {})
        mock_synth.side_effect = RuntimeError("Anthropic 400: prompt too long")

        client = MagicMock()
        _panelists, _result_dicts, _panelist_usage, _panelist_cost, synthesis_dict, _variants = run_panel_sync(
            client=client,
            personas=[{"name": "Panelist0"}, {"name": "Panelist1"}],
            questions=[{"text": "Q1"}],
            model="sonnet",
            synthesis=True,
        )
        assert synthesis_dict is not None
        err = synthesis_dict.get("synthesis_error")
        assert isinstance(err, dict)
        assert err["error_type"] == "synthesis_api_error"
        assert "message" in err
        assert "suggested_fix" in err

    @patch("synth_panel._runners.synthesize_panel")
    @patch("synth_panel._runners.run_panel_parallel")
    def test_preflight_rejects_before_api_call(self, mock_run, mock_synth):
        """Pre-flight check must short-circuit before synthesize_panel is called."""
        from synth_panel._runners import run_panel_sync
        from synth_panel.orchestrator import PanelistResult

        giant = "A" * 800_000
        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name=f"P{i}",
                    responses=[{"question": "Q1", "response": giant}],
                    usage=ZERO_USAGE,
                    model="haiku",
                )
                for i in range(2)
            ],
            None,
            {},
        )

        client = MagicMock()
        _pr, _rd, _pu, _pc, synthesis_dict, _v = run_panel_sync(
            client=client,
            personas=[{"name": "P0"}, {"name": "P1"}],
            questions=[{"text": "Q1"}],
            model="haiku",
            synthesis=True,
            synthesis_model="haiku",
        )
        mock_synth.assert_not_called()
        assert synthesis_dict is not None
        err = synthesis_dict["synthesis_error"]
        assert err["error_type"] == "synthesis_context_overflow"
        diag = err["diagnostic"]
        assert diag["context_window"] == 200_000


# --- sp-avmm: context window table ----------------------------------------


class TestContextWindowResolution:
    """The pre-flight check's model→window lookup is the load-bearing
    table. Guard the key lookups used by the acceptance criteria."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("haiku", 200_000),
            ("claude-haiku-4-5-20251001", 200_000),
            ("sonnet", 200_000),
            ("claude-sonnet-4-6", 200_000),
            ("opus", 200_000),
            ("gemini-2.5-flash", 1_000_000),
            ("gemini-2.5-pro", 1_000_000),
            ("gemini-2.5-flash-lite", 1_000_000),
            ("qwen3-32b", 131_072),
            ("deepseek-v3-0324", 128_000),
        ],
    )
    def test_known_aliases_resolve(self, model, expected):
        from synth_panel._runners import _resolve_context_window

        window, is_default = _resolve_context_window(model)
        assert window == expected
        assert is_default is False

    def test_unknown_model_falls_back_to_default(self):
        from synth_panel._runners import _resolve_context_window

        window, is_default = _resolve_context_window("totally-made-up-99b")
        assert window == 128_000
        assert is_default is True


# --- sy-549: openrouter/anthropic/* synthesis routing ---------------------


class TestSynthesisOpenRouterRouting:
    """sy-549: a synthesis model of the form ``openrouter/anthropic/*`` must
    route through OpenRouter (OPENROUTER_API_KEY) for the WHOLE synthesis call
    chain — including the structured-output engine's final-strike escalation,
    which previously jumped to the bare ``sonnet`` alias on the *direct*
    Anthropic provider and demanded ANTHROPIC_API_KEY an OpenRouter-only
    caller never set.
    """

    def test_openrouter_synthesis_never_demands_anthropic_key(self, monkeypatch):
        from synth_panel.llm.client import LLMClient
        from synth_panel.synthesis import synthesize_panel

        # OpenRouter-only environment.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        seen_urls: list[str] = []

        def fake_post(url, **kwargs):
            seen_urls.append(url)
            resp = MagicMock()
            resp.status_code = 200
            # Return an empty tool input so the structured engine exhausts its
            # retries and reaches the final-strike escalation branch — the exact
            # path that used to cross over to the Anthropic provider.
            resp.json.return_value = {
                "id": "x",
                "model": "anthropic/claude-haiku-4.5",
                "role": "assistant",
                "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "t", "name": "synthesize", "input": {}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            return resp

        monkeypatch.setattr("httpx.post", fake_post)

        results = _panelist_results(2)
        # Should NOT raise "Missing API key for Anthropic".
        synth = synthesize_panel(
            LLMClient(),
            results,
            [{"text": "Q1"}],
            model="openrouter/anthropic/claude-haiku-4.5",
            panelist_model="openrouter/anthropic/claude-haiku-4.5",
        )
        # Every call (including the escalation strike) hit OpenRouter.
        assert seen_urls, "synthesis made no LLM call"
        assert all("openrouter.ai" in u for u in seen_urls)
        # The synthesis itself is a fallback (empty tool input every time),
        # surfacing the failure rather than pretending success.
        assert synth.is_fallback is True


# --- gh#571: native-provider synthesis escalation must stay in-family ------


class TestSynthesisNativeGeminiRouting:
    """gh#571 (sy-549 class): a native ``gemini-*`` synthesis run with ONLY
    GEMINI_API_KEY set must keep the WHOLE synthesis call chain — including
    the structured-output engine's final-strike escalation — on the Gemini
    provider. Previously the final strike escalated to the bare ``sonnet``
    alias on the direct Anthropic provider and failed the run with
    "Missing API key for Anthropic"."""

    def test_gemini_synthesis_never_demands_anthropic_key(self, monkeypatch):
        from synth_panel.llm.client import LLMClient
        from synth_panel.synthesis import synthesize_panel

        # Gemini-only environment (the live gh#571 repro shape).
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        seen_urls: list[str] = []
        seen_models: list[str] = []

        def fake_post(url, **kwargs):
            seen_urls.append(url)
            seen_models.append(kwargs.get("json", {}).get("model", ""))
            resp = MagicMock()
            resp.status_code = 200
            # Empty tool arguments every time so the structured engine
            # exhausts its retries and reaches the final-strike escalation
            # branch — the exact path that used to cross to Anthropic.
            resp.json.return_value = {
                "id": "x",
                "model": "gemini-2.5-flash",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "t",
                                    "type": "function",
                                    "function": {"name": "synthesize", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
            return resp

        monkeypatch.setattr("httpx.post", fake_post)

        results = _panelist_results(2)
        # Should NOT raise "Missing API key for Anthropic".
        synth = synthesize_panel(
            LLMClient(),
            results,
            [{"text": "Q1"}],
            model="gemini-2.5-flash",
            panelist_model="gemini-2.5-flash",
        )
        # Every call (including the escalation strike) hit Gemini.
        assert seen_urls, "synthesis made no LLM call"
        # Regex host check (not a bare substring test on a URL literal) so
        # CodeQL's py/incomplete-url-substring-sanitization heuristic does
        # not misread this test assertion as URL sanitization logic.
        assert all(re.search(r"generativelanguage\.googleapis\.com", u) for u in seen_urls)
        # The escalation strike stayed in the Gemini family.
        assert all(m.startswith("gemini-") for m in seen_models)
        assert "gemini-2.5-pro" in seen_models
        # The synthesis itself is a fallback (empty tool input every time),
        # surfacing the failure rather than pretending success.
        assert synth.is_fallback is True


# --- sy-549: fallback synthesis must surface a marker, not save silently --


class TestSynthesisFallbackSurfaced:
    """sy-549 (2): a fallback synthesis (judge exhausted retries / partial
    schema) must not be persisted as a silent success — it carries an
    ``is_fallback`` marker in to_dict() and fails the run loudly."""

    def test_to_dict_marks_fallback(self):
        from synth_panel.synthesis import SynthesisResult

        sr = SynthesisResult(
            summary="Synthesis failed — see error field.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            is_fallback=True,
            error="judge exhausted retries",
        )
        d = sr.to_dict()
        assert d["is_fallback"] is True
        assert d["error"] == "judge exhausted retries"

    def test_healthy_synthesis_has_no_fallback_marker(self):
        from synth_panel.synthesis import SynthesisResult

        sr = SynthesisResult(
            summary="All good",
            themes=["t"],
            agreements=["a"],
            disagreements=[],
            surprises=[],
            recommendation="ship it",
        )
        d = sr.to_dict()
        assert "is_fallback" not in d
        assert "error" not in d

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_inrun_fallback_sets_run_invalid(self, _mock_client, mock_run, mock_synth, capsys, tmp_path):
        from synth_panel.orchestrator import WorkerRegistry
        from synth_panel.synthesis import SynthesisResult

        mock_run.return_value = (_panelist_results(2), WorkerRegistry(), {})
        mock_synth.return_value = SynthesisResult(
            summary="Synthesis failed — see error field.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            is_fallback=True,
            error="judge exhausted retries",
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
            ]
        )
        assert code == 2
        captured = capsys.readouterr()
        assert "incomplete" in captured.err.lower()
        payload = json.loads(captured.out)
        assert payload["run_invalid"] is True
        err = payload["synthesis_error"]
        assert err["error_type"] == "synthesis_fallback"

    @patch("synth_panel.cli.commands.synthesize_panel")
    def test_resynthesize_fallback_exits_two(self, mock_synth, capsys, tmp_path):
        from synth_panel.mcp.data import save_panel_result
        from synth_panel.synthesis import SynthesisResult

        mock_synth.return_value = SynthesisResult(
            summary="Synthesis failed — see error field.",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            is_fallback=True,
            error="judge exhausted retries",
        )
        saved = save_panel_result(
            results=[
                {
                    "persona": "Alice",
                    "responses": [{"question": "Q1", "response": "An answer"}],
                    "usage": ZERO_USAGE.to_dict(),
                    "cost": "$0.00",
                    "error": None,
                }
            ],
            model="sonnet",
            total_usage=ZERO_USAGE.to_dict(),
            total_cost="$0.00",
            persona_count=1,
            question_count=1,
        )

        code = main(["--output-format", "json", "panel", "synthesize", saved])
        assert code == 2
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["run_invalid"] is True
        assert payload["synthesis_error"]["error_type"] == "synthesis_fallback"


# --- sp-avmm: detect_synthesis_context_overflow diagnostic shape ----------


class TestSynthesisOverflowDetector:
    def test_returns_none_when_fits(self):
        from synth_panel._runners import detect_synthesis_context_overflow
        from synth_panel.orchestrator import PanelistResult

        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[{"question": "Q", "response": "A short answer"}],
                usage=ZERO_USAGE,
            )
        ]
        assert (
            detect_synthesis_context_overflow(
                results,
                [{"text": "Q"}],
                synthesis_model="haiku",
            )
            is None
        )

    def test_returns_diagnostic_when_overflows(self):
        from synth_panel._runners import detect_synthesis_context_overflow
        from synth_panel.orchestrator import PanelistResult

        giant = "x" * 900_000
        results = [
            PanelistResult(
                persona_name="Alice",
                responses=[{"question": "Q", "response": giant}],
                usage=ZERO_USAGE,
            )
        ]
        diag = detect_synthesis_context_overflow(
            results,
            [{"text": "Q"}],
            synthesis_model="haiku",
        )
        assert diag is not None
        assert diag["context_window"] == 200_000
        assert diag["synthesis_model"] == "haiku"
        assert diag["effective_limit"] == 200_000 - 8_000
        assert diag["estimated_tokens"] > diag["effective_limit"]
