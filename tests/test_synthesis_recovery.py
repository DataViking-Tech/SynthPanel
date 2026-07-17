"""Tests for the synthesis failure recovery ladder (sp-rcvr).

Covers: the overflow/transient/fatal classifier, each ladder rung with a
mocked synthesis layer, the map-reduce trigger on a downstream 400 with an
oversized estimate, the OpenRouter reroute request shape, the `panel
synthesize` global --model warning, and the guided last-resort message.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx
import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock
from synth_panel.llm.providers.openrouter import (
    OpenRouterProvider,
    _openrouter_error_from_response,
)
from synth_panel.orchestrator import PanelistResult
from synth_panel.synthesis import STRATEGY_MAP_REDUCE, SynthesisResult
from synth_panel.synthesis_recovery import (
    RecoveryContext,
    SynthesisFailureClass,
    SynthesisRecoveryError,
    classify_synthesis_failure,
    downstream_provider_of,
    format_recovery_command,
    provider_slug,
    recover_synthesis_failure,
    suggest_recovery_model,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _panelists(n: int = 2, answer: str = "I like it.") -> list[PanelistResult]:
    return [
        PanelistResult(
            persona_name=f"Persona {i}",
            responses=[{"question": "What do you think?", "response": answer}],
            usage=ZERO_USAGE,
        )
        for i in range(n)
    ]


_QUESTIONS = [{"text": "What do you think?"}]


def _ok_result(**kwargs: Any) -> SynthesisResult:
    defaults: dict[str, Any] = dict(
        summary="fine",
        themes=["t"],
        agreements=[],
        disagreements=[],
        surprises=[],
        recommendation="ship it",
    )
    defaults.update(kwargs)
    return SynthesisResult(**defaults)


def _ctx(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> RecoveryContext:
    # No rung should ever sleep for real in tests.
    monkeypatch.setattr("synth_panel.synthesis_recovery.time.sleep", lambda _s: None)
    defaults: dict[str, Any] = dict(
        client=object(),  # never touched: synthesis fns are monkeypatched
        panelist_results=_panelists(),
        questions=_QUESTIONS,
        synthesis_model="openrouter/anthropic/claude-haiku-4.5",
    )
    defaults.update(kwargs)
    return RecoveryContext(**defaults)


def _or_400(downstream: str | None = "Azure") -> LLMError:
    label = f"OpenRouter (downstream: {downstream})" if downstream else "OpenRouter"
    return LLMError(
        f"{label} API error 400: Provider returned error",
        LLMErrorCategory.BAD_REQUEST,
        status_code=400,
        downstream_provider=downstream,
    )


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class TestClassifier:
    def test_oversized_estimate_classifies_as_overflow(self) -> None:
        # Plain 400 with no keywords, but the estimate exceeds the window.
        cls = classify_synthesis_failure(_or_400(), estimated_tokens=250_000, context_window=200_000)
        assert cls is SynthesisFailureClass.OVERFLOW

    @pytest.mark.parametrize(
        "message",
        [
            "prompt is too long: 214321 tokens > 200000 maximum",
            "This request exceeds the maximum context length",
            "input is too long for requested model",
            "context_length_exceeded: reduce your prompt",
            "too many input tokens for this model",
        ],
    )
    def test_keyword_heuristics_classify_as_overflow(self, message: str) -> None:
        exc = LLMError(message, LLMErrorCategory.BAD_REQUEST, status_code=400)
        cls = classify_synthesis_failure(exc, estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.OVERFLOW

    @pytest.mark.parametrize(
        ("category", "status"),
        [
            (LLMErrorCategory.RATE_LIMIT, 429),
            (LLMErrorCategory.SERVER_ERROR, 500),
            (LLMErrorCategory.SERVER_ERROR, 529),
            (LLMErrorCategory.TRANSPORT, None),
        ],
    )
    def test_transient_categories(self, category: LLMErrorCategory, status: int | None) -> None:
        exc = LLMError("boom", category, status_code=status)
        cls = classify_synthesis_failure(exc, estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.TRANSIENT

    def test_retries_exhausted_wrapping_transient_is_transient(self) -> None:
        inner = LLMError("503 upstream", LLMErrorCategory.SERVER_ERROR, status_code=503)
        outer = LLMError("Retries exhausted after 3 attempts: 503", LLMErrorCategory.RETRIES_EXHAUSTED, cause=inner)
        cls = classify_synthesis_failure(outer, estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.TRANSIENT

    def test_plain_400_is_fatal(self) -> None:
        cls = classify_synthesis_failure(_or_400(), estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.FATAL

    def test_auth_error_is_fatal(self) -> None:
        exc = LLMError("bad key", LLMErrorCategory.AUTHENTICATION, status_code=401)
        cls = classify_synthesis_failure(exc, estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.FATAL

    def test_non_llm_error_is_fatal(self) -> None:
        cls = classify_synthesis_failure(ValueError("bug"), estimated_tokens=1_000, context_window=200_000)
        assert cls is SynthesisFailureClass.FATAL


# ---------------------------------------------------------------------------
# Downstream provider extraction + slugs + suggestions
# ---------------------------------------------------------------------------


class TestDownstreamExtraction:
    def test_structured_attribute_wins(self) -> None:
        assert downstream_provider_of(_or_400("Azure")) == "Azure"

    def test_attribute_found_through_cause_chain(self) -> None:
        inner = _or_400("Azure")
        outer = LLMError("Retries exhausted: x", LLMErrorCategory.RETRIES_EXHAUSTED, cause=inner)
        assert downstream_provider_of(outer) == "Azure"

    def test_message_fallback(self) -> None:
        exc = RuntimeError("OpenRouter (downstream: Amazon Bedrock) API error 400: Provider returned error")
        assert downstream_provider_of(exc) == "Amazon Bedrock"

    def test_none_when_absent(self) -> None:
        assert downstream_provider_of(LLMError("Anthropic API error 400: nope", LLMErrorCategory.BAD_REQUEST)) is None

    @pytest.mark.parametrize(
        ("name", "slug"),
        [("Azure", "azure"), ("Amazon Bedrock", "amazon-bedrock"), ("google-vertex", "google-vertex")],
    )
    def test_provider_slug(self, name: str, slug: str) -> None:
        assert provider_slug(name) == slug

    def test_suggest_recovery_model_openrouter(self) -> None:
        assert suggest_recovery_model("openrouter/anthropic/claude-haiku-4.5") == "openrouter/google/gemini-2.5-flash"

    def test_suggest_recovery_model_gemini_family(self) -> None:
        assert suggest_recovery_model("gemini-2.5-flash-lite") == "gemini-2.5-pro"

    def test_suggest_recovery_model_default(self) -> None:
        assert suggest_recovery_model("sonnet") == "gemini-2.5-flash-lite"

    def test_format_recovery_command(self) -> None:
        cmd = format_recovery_command("abc123", "openrouter/anthropic/claude-haiku-4.5")
        assert cmd == "synthpanel panel synthesize abc123 --synthesis-model openrouter/google/gemini-2.5-flash"

    def test_format_recovery_command_placeholder_id(self) -> None:
        assert "<result-id>" in format_recovery_command(None, "sonnet")


# ---------------------------------------------------------------------------
# Ladder rungs (mocked synthesis layer)
# ---------------------------------------------------------------------------


class TestLadderRungs:
    def test_transient_error_retried_once_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_single(*args: Any, **kwargs: Any) -> SynthesisResult:
            calls.append(kwargs)
            return _ok_result()

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", fake_single)
        ctx = _ctx(monkeypatch)
        exc = LLMError("overloaded", LLMErrorCategory.SERVER_ERROR, status_code=529)
        result = recover_synthesis_failure(exc, ctx)
        assert result.summary == "fine"
        assert len(calls) == 1
        assert calls[0].get("provider_routing") is None
        err = capsys.readouterr().err
        assert "rung=retry" in err

    def test_transient_retry_is_bounded_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[Any] = []

        def always_529(*args: Any, **kwargs: Any) -> SynthesisResult:
            calls.append(kwargs)
            raise LLMError("overloaded", LLMErrorCategory.SERVER_ERROR, status_code=529)

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", always_529)
        ctx = _ctx(monkeypatch, synthesis_model="sonnet")
        with pytest.raises(SynthesisRecoveryError):
            recover_synthesis_failure(LLMError("overloaded", LLMErrorCategory.SERVER_ERROR, status_code=529), ctx)
        # Only ONE ladder-level retry (no downstream → no reroute rung).
        assert len(calls) == 1

    def test_overflow_routes_into_map_reduce(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mr_calls: list[dict[str, Any]] = []

        def fake_mapreduce(*args: Any, **kwargs: Any) -> SynthesisResult:
            mr_calls.append(kwargs)
            return _ok_result(strategy=STRATEGY_MAP_REDUCE)

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel_mapreduce", fake_mapreduce)
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel",
            lambda *a, **k: pytest.fail("single-pass call must not be retried on overflow"),
        )
        ctx = _ctx(monkeypatch)
        exc = LLMError(
            "OpenRouter (downstream: Azure) API error 400: prompt is too long",
            LLMErrorCategory.BAD_REQUEST,
            status_code=400,
            downstream_provider="Azure",
        )
        result = recover_synthesis_failure(exc, ctx)
        assert result.strategy == STRATEGY_MAP_REDUCE
        assert len(mr_calls) == 1
        assert "rung=map-reduce" in capsys.readouterr().err

    def test_downstream_400_with_oversized_estimate_triggers_map_reduce(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The production bug shape: pre-flight passed nothing, downstream 400'd,
        and the token estimate exceeds the model window — must fire map-reduce."""
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.estimate_single_pass_tokens",
            lambda *a, **k: 250_000,
        )
        mr_kwargs: dict[str, Any] = {}

        def fake_mapreduce(*args: Any, **kwargs: Any) -> SynthesisResult:
            mr_kwargs.update(kwargs)
            return _ok_result(strategy=STRATEGY_MAP_REDUCE)

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel_mapreduce", fake_mapreduce)
        ctx = _ctx(monkeypatch)  # claude-haiku → 200k window
        result = recover_synthesis_failure(_or_400("Azure"), ctx)
        assert result.strategy == STRATEGY_MAP_REDUCE
        # True documented-window overflow: normal map-reduce planning
        # already splits, so no artificial cap is applied.
        assert mr_kwargs["context_limit_override"] is None

    def test_explicit_single_strategy_skips_map_reduce_and_says_so(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel_mapreduce",
            lambda *a, **k: pytest.fail("map-reduce must not run when strategy=single"),
        )
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.estimate_single_pass_tokens",
            lambda *a, **k: 250_000,
        )
        ctx = _ctx(monkeypatch, allow_map_reduce=False)
        with pytest.raises(SynthesisRecoveryError) as ei:
            recover_synthesis_failure(_or_400(None), ctx)
        assert ei.value.map_reduce_blocked_by_strategy is True
        assert "--synthesis-strategy=single" in str(ei.value)

    def test_reroute_rung_excludes_failed_downstream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, Any]] = []

        def fake_single(*args: Any, **kwargs: Any) -> SynthesisResult:
            calls.append(kwargs)
            return _ok_result()

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", fake_single)
        ctx = _ctx(monkeypatch)
        # Plain 400, small prompt: FATAL class, downstream known → reroute.
        result = recover_synthesis_failure(_or_400("Azure"), ctx)
        assert result.summary == "fine"
        assert len(calls) == 1
        assert calls[0]["provider_routing"] == {"ignore": ["azure"]}

    def test_reroute_failure_then_suspected_overflow_map_reduce(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reroute 400s too; a large (but pre-flight-passing) prompt then
        falls back to map-reduce as suspected downstream overflow."""
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.estimate_single_pass_tokens",
            lambda *a, **k: 150_000,  # under haiku's 200k documented window
        )

        def reroute_also_400s(*args: Any, **kwargs: Any) -> SynthesisResult:
            raise _or_400("Google")

        mr_kwargs: dict[str, Any] = {}

        def fake_mapreduce(*args: Any, **kwargs: Any) -> SynthesisResult:
            mr_kwargs.update(kwargs)
            return _ok_result(strategy=STRATEGY_MAP_REDUCE)

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", reroute_also_400s)
        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel_mapreduce", fake_mapreduce)
        ctx = _ctx(monkeypatch)
        result = recover_synthesis_failure(_or_400("Azure"), ctx)
        assert result.strategy == STRATEGY_MAP_REDUCE
        # The prompt FIT the documented window, so the map-reduce pass must
        # be capped below it — half the rejected estimate — or it would just
        # re-send the same oversized chunk to the same downstream.
        assert mr_kwargs["context_limit_override"] == 75_000
        assert "suspected downstream context overflow" in capsys.readouterr().err

    def test_reroute_failure_small_prompt_exhausts_ladder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def always_400(*args: Any, **kwargs: Any) -> SynthesisResult:
            raise _or_400("Azure")

        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", always_400)
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel_mapreduce",
            lambda *a, **k: pytest.fail("small prompt must not trigger suspected-overflow map-reduce"),
        )
        ctx = _ctx(monkeypatch)  # tiny panel → tiny estimate
        with pytest.raises(SynthesisRecoveryError):
            recover_synthesis_failure(_or_400("Azure"), ctx)

    def test_fatal_without_downstream_goes_straight_to_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel",
            lambda *a, **k: pytest.fail("no rung applies to a fatal non-OR error"),
        )
        ctx = _ctx(monkeypatch, synthesis_model="sonnet")
        exc = LLMError("Anthropic API error 400: invalid request", LLMErrorCategory.BAD_REQUEST, status_code=400)
        with pytest.raises(SynthesisRecoveryError) as ei:
            recover_synthesis_failure(exc, ctx)
        assert ei.value.rungs  # classification line is always logged

    def test_last_resort_message_names_exact_recovery_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel",
            lambda *a, **k: (_ for _ in ()).throw(_or_400("Azure")),
        )
        ctx = _ctx(monkeypatch, result_id="panel-42")
        with pytest.raises(SynthesisRecoveryError) as ei:
            recover_synthesis_failure(_or_400("Azure"), ctx)
        msg = str(ei.value)
        assert "synthpanel panel synthesize panel-42 --synthesis-model openrouter/google/gemini-2.5-flash" in msg
        assert ei.value.suggested_command.endswith("openrouter/google/gemini-2.5-flash")


class TestDownstreamRevealedLimit:
    def test_capped_at_half_estimate_when_fits_documented_window(self) -> None:
        from synth_panel.synthesis_recovery import _downstream_revealed_limit

        assert _downstream_revealed_limit(150_000, 200_000) == 75_000

    def test_floored_at_suspected_threshold(self) -> None:
        from synth_panel.synthesis_recovery import _downstream_revealed_limit

        assert _downstream_revealed_limit(40_000, 200_000) == 32_000

    def test_none_for_true_documented_overflow(self) -> None:
        from synth_panel.synthesis_recovery import _downstream_revealed_limit

        assert _downstream_revealed_limit(250_000, 200_000) is None


class TestContextLimitOverridePlanning:
    def test_override_forces_sub_chunking(self) -> None:
        """With a small override, a question whose responses fit the documented
        window is still split into per-batch calls + inner reduce."""
        from synth_panel.synthesis import synthesize_panel_mapreduce

        long_answer = "word " * 6_400  # ~32k chars → ~8k tokens per panelist
        panelists = _panelists(2, answer=long_answer)

        class _CountingClient:
            def __init__(self) -> None:
                self.call_count = 0

            def send(self, request: Any, **kwargs: Any) -> Any:
                from synth_panel.llm.models import (
                    CompletionResponse,
                    ToolInvocationBlock,
                )
                from synth_panel.llm.models import (
                    TokenUsage as LLMTokenUsage,
                )

                self.call_count += 1
                return CompletionResponse(
                    id=f"synth-{self.call_count}",
                    model="claude-haiku-4-5",
                    content=[
                        ToolInvocationBlock(
                            id="tc1",
                            name="synthesize",
                            input={
                                "summary": "s",
                                "themes": ["t"],
                                "agreements": [],
                                "disagreements": [],
                                "surprises": [],
                                "recommendation": "r",
                            },
                        )
                    ],
                    usage=LLMTokenUsage(input_tokens=10, output_tokens=5),
                )

        # Without an override this panel is 1 map + 1 reduce (2 calls).
        baseline = _CountingClient()
        synthesize_panel_mapreduce(baseline, panelists, _QUESTIONS, model="claude-haiku-4-5")  # type: ignore[arg-type]
        assert baseline.call_count == 2

        # ~10k-token cap: each panelist (~8k tokens + scaffold) gets its own
        # batch → 2 batch maps + inner reduce + outer reduce = 4 calls.
        capped = _CountingClient()
        synthesize_panel_mapreduce(
            capped,  # type: ignore[arg-type]
            panelists,
            _QUESTIONS,
            model="claude-haiku-4-5",
            context_limit_override=11_000,
        )
        assert capped.call_count == 4


# ---------------------------------------------------------------------------
# OpenRouter: downstream metadata parsing + reroute request shape
# ---------------------------------------------------------------------------


def _fake_response(status: int, body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=body,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )


class TestOpenRouterErrorMetadata:
    def test_downstream_provider_attribute_set(self) -> None:
        resp = _fake_response(
            400,
            {
                "error": {
                    "code": 400,
                    "message": "Provider returned error",
                    "metadata": {"provider_name": "Azure"},
                }
            },
        )
        err = _openrouter_error_from_response(resp)
        assert err.downstream_provider == "Azure"
        assert "downstream: Azure" in str(err)
        assert err.status_code == 400

    def test_no_metadata_leaves_downstream_none(self) -> None:
        resp = _fake_response(400, {"error": {"code": 400, "message": "bad request"}})
        err = _openrouter_error_from_response(resp)
        assert err.downstream_provider is None


class TestRerouteRequestShape:
    @pytest.fixture()
    def provider(self, monkeypatch: pytest.MonkeyPatch) -> OpenRouterProvider:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        return OpenRouterProvider()

    def _request(self, model: str) -> CompletionRequest:
        return CompletionRequest(
            model=model,
            max_tokens=64,
            messages=[InputMessage(role="user", content=[TextBlock(text="hi")])],
            provider_routing={"ignore": ["azure"]},
        )

    def test_openai_transport_body_carries_provider_object(
        self, provider: OpenRouterProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> httpx.Response:
            captured["json"] = kwargs["json"]
            return _fake_response(
                200,
                {
                    "id": "gen-1",
                    "model": "google/gemini-2.5-flash",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        monkeypatch.setattr("synth_panel.llm.providers.openrouter.httpx.post", fake_post)
        provider.send(self._request("openrouter/google/gemini-2.5-flash"))
        assert captured["json"]["provider"] == {"ignore": ["azure"]}

    def test_anthropic_passthrough_body_carries_provider_object(
        self, provider: OpenRouterProvider, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> httpx.Response:
            captured["url"] = url
            captured["json"] = kwargs["json"]
            return _fake_response(
                200,
                {
                    "id": "msg-1",
                    "model": "anthropic/claude-haiku-4.5",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        monkeypatch.setattr("synth_panel.llm.providers.openrouter.httpx.post", fake_post)
        provider.send(self._request("openrouter/anthropic/claude-haiku-4.5"))
        assert captured["url"].endswith("/v1/messages")
        assert captured["json"]["provider"] == {"ignore": ["azure"]}

    def test_no_provider_object_when_unset(self, provider: OpenRouterProvider, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        def fake_post(url: str, **kwargs: Any) -> httpx.Response:
            captured["json"] = kwargs["json"]
            return _fake_response(
                200,
                {
                    "id": "gen-1",
                    "model": "google/gemini-2.5-flash",
                    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            )

        monkeypatch.setattr("synth_panel.llm.providers.openrouter.httpx.post", fake_post)
        req = CompletionRequest(
            model="openrouter/google/gemini-2.5-flash",
            max_tokens=64,
            messages=[InputMessage(role="user", content=[TextBlock(text="hi")])],
        )
        provider.send(req)
        assert "provider" not in captured["json"]


# ---------------------------------------------------------------------------
# panel synthesize: global --model warning + ladder wiring
# ---------------------------------------------------------------------------


def _write_saved_result(tmp_path: Any) -> Any:
    result = {
        "id": "saved-1",
        "model": "openrouter/anthropic/claude-haiku-4.5",
        "questions": _QUESTIONS,
        "results": [
            {
                "persona": "Persona 0",
                "responses": [{"question": "What do you think?", "response": "Nice."}],
                "usage": {},
            }
        ],
    }
    p = tmp_path / "saved-1.json"
    p.write_text(json.dumps(result), encoding="utf-8")
    return p


class TestPanelSynthesizeCommand:
    def _args(self, result: str, **kwargs: Any) -> argparse.Namespace:
        ns = argparse.Namespace(result=result, synthesis_model=None, synthesis_prompt=None, model=None)
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def test_global_model_flag_warns(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli import commands as cmd
        from synth_panel.cli.output import OutputFormat

        path = _write_saved_result(tmp_path)
        monkeypatch.setattr(
            cmd,
            "synthesize_panel",
            lambda *a, **k: _ok_result(model="openrouter/anthropic/claude-haiku-4.5"),
        )
        monkeypatch.setattr("synth_panel.mcp.data.save_panel_synthesis", lambda *a, **k: "sidecar.json")
        rc = cmd.handle_panel_synthesize(self._args(str(path), model="opus"), OutputFormat.TEXT)
        assert rc == 0
        err = capsys.readouterr().err
        assert "ignored by 'panel synthesize'" in err
        assert "--synthesis-model opus" in err

    def test_no_warning_without_global_model(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli import commands as cmd
        from synth_panel.cli.output import OutputFormat

        path = _write_saved_result(tmp_path)
        monkeypatch.setattr(
            cmd,
            "synthesize_panel",
            lambda *a, **k: _ok_result(model="openrouter/anthropic/claude-haiku-4.5"),
        )
        monkeypatch.setattr("synth_panel.mcp.data.save_panel_synthesis", lambda *a, **k: "sidecar.json")
        rc = cmd.handle_panel_synthesize(self._args(str(path)), OutputFormat.TEXT)
        assert rc == 0
        assert "ignored by 'panel synthesize'" not in capsys.readouterr().err

    def test_failed_synthesis_runs_ladder_and_reports_guided_command(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli import commands as cmd
        from synth_panel.cli.output import OutputFormat

        path = _write_saved_result(tmp_path)

        def always_400(*args: Any, **kwargs: Any) -> SynthesisResult:
            raise _or_400("Azure")

        monkeypatch.setattr(cmd, "synthesize_panel", always_400)
        # Ladder's own calls go through synthesis_recovery's import.
        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", always_400)
        monkeypatch.setattr("synth_panel.synthesis_recovery.time.sleep", lambda _s: None)
        monkeypatch.setattr(
            "synth_panel.synthesis_recovery.synthesize_panel_mapreduce",
            lambda *a, **k: (_ for _ in ()).throw(_or_400("Azure")),
        )
        rc = cmd.handle_panel_synthesize(self._args(str(path)), OutputFormat.TEXT)
        assert rc == 2
        err = capsys.readouterr().err
        # Guided fallback names the exact recovery command with the saved id.
        assert "synthpanel panel synthesize saved-1 --synthesis-model openrouter/google/gemini-2.5-flash" in err

    def test_ladder_reroute_recovers_panel_synthesize(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from synth_panel.cli import commands as cmd
        from synth_panel.cli.output import OutputFormat

        path = _write_saved_result(tmp_path)

        def primary_400(*args: Any, **kwargs: Any) -> SynthesisResult:
            raise _or_400("Azure")

        def reroute_ok(*args: Any, **kwargs: Any) -> SynthesisResult:
            assert kwargs.get("provider_routing") == {"ignore": ["azure"]}
            return _ok_result(model="openrouter/anthropic/claude-haiku-4.5")

        monkeypatch.setattr(cmd, "synthesize_panel", primary_400)
        monkeypatch.setattr("synth_panel.synthesis_recovery.synthesize_panel", reroute_ok)
        monkeypatch.setattr("synth_panel.mcp.data.save_panel_synthesis", lambda *a, **k: "sidecar.json")
        rc = cmd.handle_panel_synthesize(self._args(str(path)), OutputFormat.TEXT)
        assert rc == 0
        assert "rung=reroute" in capsys.readouterr().err
