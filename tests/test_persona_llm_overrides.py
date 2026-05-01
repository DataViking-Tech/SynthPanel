"""Tests for per-persona ``llm_overrides`` (sp-4loufu / SP#347).

Covers four guarantees that together make the feature usable:

1. Validation rejects out-of-range / wrong-type / typo'd overrides
   *before* the run starts, so a malformed YAML block can't silently
   fall back to the run-level default.
2. Per-persona temperature / top_p / max_tokens reach
   ``CompletionRequest`` for both free-text and structured paths.
3. Run-level CLI flags act as the default; only personas that opt in
   diverge from them.
4. ``llm_overrides.model`` is recognised by the YAML-overrides
   extraction so it routes through ``persona_models`` exactly like
   the legacy top-level ``model`` field.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.cli.commands import assign_models_to_personas
from synth_panel.llm.models import (
    CompletionResponse,
    StopReason,
    TextBlock,
)
from synth_panel.llm.models import (
    TokenUsage as LLMTokenUsage,
)
from synth_panel.orchestrator import (
    get_persona_llm_overrides,
    run_panel_parallel,
    validate_llm_overrides,
)


def _make_text_response(text: str = "ok", usage: LLMTokenUsage | None = None) -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=usage or LLMTokenUsage(input_tokens=5, output_tokens=3),
    )


def _make_capturing_client() -> tuple[MagicMock, list[Any]]:
    """Return a thread-safe mock client that records every CompletionRequest."""
    captured: list[Any] = []
    lock = threading.Lock()

    def send(request):
        with lock:
            captured.append(request)
        return _make_text_response()

    client = MagicMock()
    client.send = MagicMock(side_effect=send)
    return client, captured


def _system_prompt(p: dict[str, Any]) -> str:
    return f"You are {p.get('name', 'Anonymous')}."


def _question_prompt(q: dict[str, Any]) -> str:
    return q.get("text", str(q)) if isinstance(q, dict) else str(q)


# ---------------------------------------------------------------------------
# validate_llm_overrides
# ---------------------------------------------------------------------------


class TestValidateLLMOverrides:
    def test_empty_dict_is_valid(self):
        validate_llm_overrides({})

    def test_temperature_in_range(self):
        validate_llm_overrides({"temperature": 0.0})
        validate_llm_overrides({"temperature": 1.0})
        validate_llm_overrides({"temperature": 2.0})

    @pytest.mark.parametrize("bad", [-0.1, 2.01, 5.0, -1.0])
    def test_temperature_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match="temperature"):
            validate_llm_overrides({"temperature": bad}, persona_name="Alice")

    def test_temperature_wrong_type_rejected(self):
        with pytest.raises(ValueError, match="temperature"):
            validate_llm_overrides({"temperature": "hot"})

    def test_temperature_bool_rejected(self):
        # ``True`` would otherwise pass the float check via int-bool coercion.
        with pytest.raises(ValueError, match="temperature"):
            validate_llm_overrides({"temperature": True})

    def test_top_p_in_range(self):
        validate_llm_overrides({"top_p": 0.0})
        validate_llm_overrides({"top_p": 1.0})

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
    def test_top_p_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match="top_p"):
            validate_llm_overrides({"top_p": bad})

    def test_max_tokens_positive_int(self):
        validate_llm_overrides({"max_tokens": 1})
        validate_llm_overrides({"max_tokens": 100000})

    @pytest.mark.parametrize("bad", [0, -1, 1.5, "100"])
    def test_max_tokens_invalid_rejected(self, bad):
        with pytest.raises(ValueError, match="max_tokens"):
            validate_llm_overrides({"max_tokens": bad})

    def test_model_string(self):
        validate_llm_overrides({"model": "haiku"})

    @pytest.mark.parametrize("bad", ["", "  ", 123, None])
    def test_model_invalid_rejected(self, bad):
        with pytest.raises(ValueError, match="model"):
            validate_llm_overrides({"model": bad})

    def test_unknown_key_rejected(self):
        # Catches typos like ``temperatur:`` that would otherwise
        # silently fall back to the run-level default.
        with pytest.raises(ValueError, match="unknown key"):
            validate_llm_overrides({"temperatur": 0.5})

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="must be a mapping"):
            validate_llm_overrides("temperature: 0.5", persona_name="Alice")  # type: ignore[arg-type]

    def test_persona_name_appears_in_error(self):
        with pytest.raises(ValueError, match="'Alice'"):
            validate_llm_overrides({"temperature": 5.0}, persona_name="Alice")


class TestGetPersonaLLMOverrides:
    def test_missing_block_returns_empty(self):
        assert get_persona_llm_overrides({"name": "Alice"}) == {}

    def test_explicit_empty_block_returns_empty(self):
        assert get_persona_llm_overrides({"name": "Alice", "llm_overrides": {}}) == {}

    def test_returns_copy(self):
        persona = {"name": "Alice", "llm_overrides": {"temperature": 0.3}}
        result = get_persona_llm_overrides(persona)
        result["temperature"] = 0.9
        # Mutating the returned dict must not contaminate the source.
        assert persona["llm_overrides"]["temperature"] == 0.3

    def test_validates_on_read(self):
        with pytest.raises(ValueError, match="temperature"):
            get_persona_llm_overrides({"name": "Alice", "llm_overrides": {"temperature": 5.0}})


# ---------------------------------------------------------------------------
# Override flow through run_panel_parallel
# ---------------------------------------------------------------------------


class TestPersonaOverridesReachRequest:
    def test_temperature_override_reaches_request(self):
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Alice", "llm_overrides": {"temperature": 0.3}},
            {"name": "Bob"},
        ]
        questions = [{"text": "hi?"}]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            temperature=0.7,
            max_workers=1,
        )
        # Pair each captured request to its persona by system prompt
        # (the only stable identifier the request carries here).
        by_persona = {req.system: req for req in captured}
        assert by_persona["You are Alice."].temperature == 0.3
        assert by_persona["You are Bob."].temperature == 0.7

    def test_top_p_override_reaches_request(self):
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Alice", "llm_overrides": {"top_p": 0.95}},
            {"name": "Bob"},
        ]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            top_p=0.5,
            max_workers=1,
        )
        by_persona = {req.system: req for req in captured}
        assert by_persona["You are Alice."].top_p == 0.95
        assert by_persona["You are Bob."].top_p == 0.5

    def test_max_tokens_override_reaches_request(self):
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Alice", "llm_overrides": {"max_tokens": 256}},
            {"name": "Bob"},
        ]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            max_workers=1,
        )
        by_persona = {req.system: req for req in captured}
        assert by_persona["You are Alice."].max_tokens == 256
        # Bob inherits the runtime default (4096) when no override given.
        assert by_persona["You are Bob."].max_tokens == 4096

    def test_no_overrides_uses_run_level(self):
        client, captured = _make_capturing_client()
        personas = [{"name": "Alice"}, {"name": "Bob"}]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            temperature=0.7,
            top_p=0.9,
            max_workers=1,
        )
        for req in captured:
            assert req.temperature == 0.7
            assert req.top_p == 0.9

    def test_mixed_temperatures_panel(self):
        """Three-way split: deliberate / default / impulsive."""
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Deliberate", "llm_overrides": {"temperature": 0.3}},
            {"name": "Default"},
            {"name": "Impulsive", "llm_overrides": {"temperature": 0.9}},
        ]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            temperature=0.7,
            max_workers=1,
        )
        by_persona = {req.system: req for req in captured}
        assert by_persona["You are Deliberate."].temperature == 0.3
        assert by_persona["You are Default."].temperature == 0.7
        assert by_persona["You are Impulsive."].temperature == 0.9

    def test_invalid_overrides_aborts_run(self):
        """A bad override must fail the run before any LLM call happens."""
        client, captured = _make_capturing_client()
        personas = [{"name": "Alice", "llm_overrides": {"temperature": 5.0}}]
        with pytest.raises(ValueError, match="Alice"):
            run_panel_parallel(
                client=client,
                personas=personas,
                questions=[{"text": "hi?"}],
                model="sonnet",
                system_prompt_fn=_system_prompt,
                question_prompt_fn=_question_prompt,
                max_workers=1,
            )
        assert captured == [], "no LLM calls should fire when validation fails"

    def test_unknown_key_aborts_run(self):
        client, _ = _make_capturing_client()
        personas = [{"name": "Alice", "llm_overrides": {"temperatur": 0.5}}]
        with pytest.raises(ValueError, match="unknown key"):
            run_panel_parallel(
                client=client,
                personas=personas,
                questions=[{"text": "hi?"}],
                model="sonnet",
                system_prompt_fn=_system_prompt,
                question_prompt_fn=_question_prompt,
                max_workers=1,
            )

    def test_orchestrator_does_not_read_overrides_model_directly(self):
        """``llm_overrides.model`` is CLI/SDK-extracted, not orchestrator-resolved.

        This mirrors the legacy top-level ``model:`` field, which the CLI
        folds into ``persona_models`` before invoking the orchestrator.
        Calling ``run_panel_parallel`` directly without ``persona_models``
        leaves the override on the floor — consistent across both fields,
        and tested by :class:`TestAssignModelsRecognisesOverridesModel`
        below for the path that actually does the extraction.
        """
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Alice", "llm_overrides": {"model": "haiku"}},
            {"name": "Bob"},
        ]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            max_workers=1,
        )
        for req in captured:
            assert req.model == "sonnet"

    def test_persona_models_arg_routes_overrides_model_when_extracted(self):
        """When the caller (CLI/SDK) extracts ``llm_overrides.model`` into
        ``persona_models``, the orchestrator routes the request accordingly.

        Combined with the test above, this pins down the contract: model
        routing always flows via ``persona_models``, and the orchestrator
        is the single chokepoint that consumes it.
        """
        client, captured = _make_capturing_client()
        personas = [
            {"name": "Alice", "llm_overrides": {"model": "haiku"}},
            {"name": "Bob"},
        ]
        run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            persona_models={"Alice": "haiku"},
            max_workers=1,
        )
        by_persona = {req.system: req.model for req in captured}
        assert by_persona["You are Alice."] == "haiku"
        assert by_persona["You are Bob."] == "sonnet"


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class TestCostTrackingReflectsOverrides:
    def test_per_persona_usage_from_provider_response(self):
        """Each persona's ``usage`` reflects the response their request got.

        This is the building block for the acceptance criterion that
        per-persona cost tracks per-persona overrides: when ``max_tokens``
        is lower for one persona, the provider returns fewer output
        tokens, ``PanelistResult.usage`` carries that, and the cost
        layer prices it accordingly. We use the mock to produce
        per-persona usage directly so the assertion stays focused on
        the orchestrator's bookkeeping.
        """
        # Different usage per persona based on system prompt.
        captured: list[Any] = []
        lock = threading.Lock()

        def send(request):
            with lock:
                captured.append(request)
            if "Alice" in (request.system or ""):
                # Lower max_tokens → simulate a shorter response.
                return _make_text_response(
                    text="short",
                    usage=LLMTokenUsage(input_tokens=10, output_tokens=20),
                )
            return _make_text_response(
                text="much longer response here",
                usage=LLMTokenUsage(input_tokens=10, output_tokens=200),
            )

        client = MagicMock()
        client.send = MagicMock(side_effect=send)

        personas = [
            {"name": "Alice", "llm_overrides": {"max_tokens": 50}},
            {"name": "Bob"},
        ]
        results, _reg, _sess = run_panel_parallel(
            client=client,
            personas=personas,
            questions=[{"text": "hi?"}],
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            max_workers=1,
        )
        by_name = {r.persona_name: r for r in results}
        assert by_name["Alice"].usage.output_tokens == 20
        assert by_name["Bob"].usage.output_tokens == 200
        # And the request actually carried the override max_tokens.
        by_persona = {req.system: req for req in captured}
        assert by_persona["You are Alice."].max_tokens == 50
        assert by_persona["You are Bob."].max_tokens == 4096


# ---------------------------------------------------------------------------
# CLI integration: assign_models_to_personas
# ---------------------------------------------------------------------------


class TestAssignModelsRecognisesOverridesModel:
    def test_llm_overrides_model_treated_as_explicit(self):
        """A persona with ``llm_overrides.model`` is excluded from the
        weighted assignment pool exactly like a top-level ``model``."""
        personas = [
            {"name": "Alice", "llm_overrides": {"model": "haiku"}},
            {"name": "Bob"},
        ]
        result, warnings = assign_models_to_personas(
            personas,
            [("sonnet", 1.0)],
            "sonnet",
        )
        assert result == {"Alice": "haiku", "Bob": "sonnet"}
        assert warnings == []

    def test_top_level_model_wins_over_overrides_model(self):
        """When both are present the legacy top-level field wins.

        That order keeps existing YAML stable: anyone who already had
        ``model:`` set won't see their behaviour change just because
        they also added an ``llm_overrides`` block.
        """
        personas = [
            {
                "name": "Alice",
                "model": "sonnet",
                "llm_overrides": {"model": "haiku"},
            },
        ]
        result, _warnings = assign_models_to_personas(
            personas,
            [("gemini", 1.0)],
            "gemini",
        )
        assert result == {"Alice": "sonnet"}
