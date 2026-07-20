"""hq-swzx: v1.0.3 P2 — typed Pydantic objects post-extraction (map-reduce).

Covers two surfaces:

* :func:`synth_panel.synthesis._typed_or_dict` — the helper that lets
  callers consume ``responses[i]["extraction"]`` whether it's a Pydantic
  ``BaseModel`` (new, attached when ``extract_schema=`` resolved a typed
  model) or a plain dict (legacy / no-model fallback).
* :func:`synth_panel.synthesis.synthesize_panel_mapreduce` — the
  map-boundary validation pass that fails loud when a per-question map
  call returns an unusable partial (explicit fallback, or schema drift
  caught by :class:`PartialSummary`) instead of silently feeding empty
  themes through to reduce.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.models import (
    CompletionResponse,
    TokenUsage,
    ToolInvocationBlock,
)
from synth_panel.orchestrator import PanelistResult
from synth_panel.structured.models import (
    AnnotatedChoice,
    PartialSummary,
    PickOne,
)
from synth_panel.synthesis import (
    MapPhaseFailure,
    SynthesisResult,
    _typed_or_dict,
    synthesize_panel_mapreduce,
)

# --- _typed_or_dict ------------------------------------------------------


class TestTypedOrDict:
    def test_reads_attribute_from_pydantic_model(self):
        m = PickOne(choice="alpha", reasoning="because")
        assert _typed_or_dict(m, "choice") == "alpha"
        assert _typed_or_dict(m, "reasoning") == "because"

    def test_reads_key_from_dict(self):
        d = {"choice": "beta", "reasoning": "legacy"}
        assert _typed_or_dict(d, "choice") == "beta"
        assert _typed_or_dict(d, "reasoning") == "legacy"

    def test_returns_none_on_missing_attr_pydantic(self):
        m = PickOne(choice="alpha")
        # PickOne has no ``extra_field``; helper must not raise.
        assert _typed_or_dict(m, "extra_field") is None

    def test_returns_none_on_missing_key_dict(self):
        assert _typed_or_dict({"choice": "alpha"}, "missing") is None

    def test_returns_none_on_none(self):
        assert _typed_or_dict(None, "anything") is None

    def test_returns_none_on_unsupported_type(self):
        # The orchestrator may legitimately attach ``None`` on extraction
        # failure or omit the field entirely; non-dict / non-BaseModel
        # values must short-circuit cleanly.
        assert _typed_or_dict("a string", "choice") is None
        assert _typed_or_dict(42, "choice") is None
        assert _typed_or_dict(["list"], "choice") is None

    def test_works_for_annotated_choice_attachment_id(self):
        """AnnotatedChoice is the v1.0.0 multimodal-attachment surface."""
        m = AnnotatedChoice(choice="A", attachment_id="att-7")
        assert _typed_or_dict(m, "attachment_id") == "att-7"
        # Dict equivalent
        assert _typed_or_dict({"choice": "A", "attachment_id": "att-7"}, "attachment_id") == "att-7"


# --- map-boundary validation --------------------------------------------


_QUESTIONS = [
    {"text": "What frustrates you?"},
    {"text": "What would you change?"},
]


_PANELISTS = [
    PanelistResult(
        persona_name="Alice",
        responses=[
            {"question": "What frustrates you?", "response": "Slow load times."},
            {"question": "What would you change?", "response": "Faster feedback."},
        ],
        usage=ZERO_USAGE,
    ),
    PanelistResult(
        persona_name="Bob",
        responses=[
            {"question": "What frustrates you?", "response": "Confusing UI."},
            {"question": "What would you change?", "response": "Redesign."},
        ],
        usage=ZERO_USAGE,
    ),
]


def _tool_response(payload: dict, input_tokens: int = 100, output_tokens: int = 50) -> CompletionResponse:
    return CompletionResponse(
        id="synth-x",
        model="claude-sonnet-4-6",
        content=[ToolInvocationBlock(id="tc1", name="synthesize", input=payload)],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_payload(marker: str) -> dict:
    return {
        "summary": f"summary-{marker}",
        "themes": [f"theme-{marker}"],
        "agreements": [f"agree-{marker}"],
        "disagreements": [f"disagree-{marker}"],
        "surprises": [f"surprise-{marker}"],
        "recommendation": f"rec-{marker}",
    }


class _SequenceClient:
    """Sequential mock LLM client (subset of test_synthesis_mapreduce)."""

    def __init__(self, responses: list[CompletionResponse]):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self._lock = threading.Lock()

    def send(self, request, **kwargs):
        with self._lock:
            resp = self._responses[self._idx]
            self._idx += 1
            self.call_count += 1
        return resp


class TestPartialSummaryAcceptsHealthyMaps:
    def test_valid_payloads_pass_validation_and_reduce_runs(self):
        n = len(_QUESTIONS)
        responses = [_tool_response(_make_payload(f"map-{i}")) for i in range(n)]
        responses.append(_tool_response(_make_payload("REDUCE")))
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(client, _PANELISTS, _QUESTIONS, model="sonnet", max_workers=1)
        assert isinstance(result, SynthesisResult)
        # Reduce ran (call_count == n maps + 1 reduce)
        assert client.call_count == n + 1
        # Top-level fields populated from the reduce call
        assert result.summary == "summary-REDUCE"


class TestMapPhaseFailureOnFallback:
    def test_fallback_map_raises_map_phase_failure(self):
        """is_fallback=True from any map call surfaces as MapPhaseFailure."""
        # Patch synthesize_panel so the second map returns is_fallback=True.
        # Caller sees a clean MapPhaseFailure naming the offending question
        # rather than reduce silently consuming "(no summary produced)".
        from synth_panel import synthesis

        def _fake_synth(client, panelists, questions, **kwargs):
            # Inspect the question text to decide which call is which.
            q_text = questions[0].get("text") if isinstance(questions[0], dict) else str(questions[0])
            if q_text == _QUESTIONS[1]["text"]:
                return SynthesisResult(
                    summary="Synthesis failed — see error field.",
                    themes=[],
                    agreements=[],
                    disagreements=[],
                    surprises=[],
                    recommendation="",
                    is_fallback=True,
                    error="upstream 400",
                )
            return SynthesisResult(
                summary=f"summary-for-{q_text}",
                themes=["theme"],
                agreements=["agree"],
                disagreements=["disagree"],
                surprises=["surprise"],
                recommendation="rec",
            )

        with (
            patch.object(synthesis, "synthesize_panel", side_effect=_fake_synth),
            pytest.raises(MapPhaseFailure) as exc_info,
        ):
            synthesize_panel_mapreduce(
                object(),  # client unused under patch
                _PANELISTS,
                _QUESTIONS,
                model="sonnet",
                max_workers=1,
            )
        err = exc_info.value
        assert err.is_fallback is True
        assert err.question_index == 1
        assert "upstream 400" in str(err)

    def test_fallback_first_question_names_index_zero(self):
        """The question_index attribute reports the offending question."""
        from synth_panel import synthesis

        def _fake_synth(client, panelists, questions, **kwargs):
            q_text = questions[0].get("text") if isinstance(questions[0], dict) else str(questions[0])
            if q_text == _QUESTIONS[0]["text"]:
                return SynthesisResult(
                    summary="failed",
                    themes=[],
                    agreements=[],
                    disagreements=[],
                    surprises=[],
                    recommendation="",
                    is_fallback=True,
                    error="bad map",
                )
            return SynthesisResult(
                summary="ok",
                themes=["t"],
                agreements=["a"],
                disagreements=["d"],
                surprises=["s"],
                recommendation="r",
            )

        with (
            patch.object(synthesis, "synthesize_panel", side_effect=_fake_synth),
            pytest.raises(MapPhaseFailure) as exc_info,
        ):
            synthesize_panel_mapreduce(object(), _PANELISTS, _QUESTIONS, model="sonnet", max_workers=1)
        assert exc_info.value.question_index == 0


class TestMapPhaseFailureOnSchemaDrift:
    def test_themes_wrong_type_raises(self):
        """A map result with a wrong-typed field triggers ValidationError."""
        from synth_panel import synthesis

        def _fake_synth(client, panelists, questions, **kwargs):
            # ``themes`` should be list[str]; deliberately set to a dict
            # to trip Pydantic's typed validation.
            return SynthesisResult(
                summary="s",
                themes={"not": "a list"},  # type: ignore[arg-type]
                agreements=["a"],
                disagreements=["d"],
                surprises=["sp"],
                recommendation="r",
            )

        with (
            patch.object(synthesis, "synthesize_panel", side_effect=_fake_synth),
            pytest.raises(MapPhaseFailure) as exc_info,
        ):
            synthesize_panel_mapreduce(object(), _PANELISTS, _QUESTIONS, model="sonnet", max_workers=1)
        err = exc_info.value
        assert err.validation_error is not None
        assert err.is_fallback is False
        # The Pydantic error chain must surface so callers can inspect it.
        assert err.__cause__ is err.validation_error

    def test_summary_wrong_type_raises(self):
        from synth_panel import synthesis

        def _fake_synth(client, panelists, questions, **kwargs):
            return SynthesisResult(
                summary=12345,  # type: ignore[arg-type]
                themes=["t"],
                agreements=["a"],
                disagreements=["d"],
                surprises=["sp"],
                recommendation="r",
            )

        with (
            patch.object(synthesis, "synthesize_panel", side_effect=_fake_synth),
            pytest.raises(MapPhaseFailure),
        ):
            synthesize_panel_mapreduce(object(), _PANELISTS, _QUESTIONS, model="sonnet", max_workers=1)


class TestPartialSummaryModelDirect:
    """Direct unit tests on the Pydantic model itself."""

    def test_round_trip_with_healthy_payload(self):
        p = PartialSummary.model_validate(
            {
                "summary": "s",
                "themes": ["t1", "t2"],
                "agreements": ["a"],
                "disagreements": ["d"],
                "surprises": ["sp"],
                "recommendation": "r",
            }
        )
        assert p.summary == "s"
        assert p.themes == ["t1", "t2"]
        assert p.recommendation == "r"

    def test_empty_lists_are_legitimate(self):
        """Empty lists per field are valid — only schema drift fails.

        The bead's failure mode is "single-persona map failure produces
        empty themes". That happens when ``is_fallback=True``, which is
        caught by the surrounding handler. A *valid* synthesis that
        legitimately found no themes for one question is not a failure.
        """
        p = PartialSummary.model_validate(
            {
                "summary": "no consensus emerged",
                "themes": [],
                "agreements": [],
                "disagreements": [],
                "surprises": [],
                "recommendation": "",
            }
        )
        assert p.themes == []
        assert p.summary == "no consensus emerged"

    def test_missing_required_field_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PartialSummary.model_validate(
                {
                    # ``summary`` missing
                    "themes": ["t"],
                    "agreements": ["a"],
                    "disagreements": ["d"],
                    "surprises": ["sp"],
                    "recommendation": "r",
                }
            )


class TestPartialSummaryStringCoercion:
    """Regression (2026-07-19 live repro): map models sometimes emit the
    list fields as ONE newline-bulleted string ("\\n- A\\n- B") instead of
    a JSON array. The single-pass path tolerates that drift, so the map
    boundary must coerce it rather than fail typed validation."""

    def test_newline_bulleted_string_coerces_to_list(self):
        p = PartialSummary.model_validate(
            {
                "summary": "s",
                "themes": "\n- Distinctiveness and identity\n- Pricing anxiety",
                "agreements": "- All want speed\n- All dislike ads",
                "disagreements": "* One split\n* Another split",
                "surprises": "1. First surprise\n2) Second surprise",
                "recommendation": "r",
            }
        )
        assert p.themes == ["Distinctiveness and identity", "Pricing anxiety"]
        assert p.agreements == ["All want speed", "All dislike ads"]
        assert p.disagreements == ["One split", "Another split"]
        assert p.surprises == ["First surprise", "Second surprise"]

    def test_plain_single_line_string_becomes_one_item(self):
        p = PartialSummary.model_validate(
            {
                "summary": "s",
                "themes": "just one theme",
                "agreements": [],
                "disagreements": [],
                "surprises": [],
                "recommendation": "r",
            }
        )
        assert p.themes == ["just one theme"]

    def test_empty_string_becomes_empty_list(self):
        p = PartialSummary.model_validate(
            {
                "summary": "s",
                "themes": "",
                "agreements": "  \n  ",
                "disagreements": [],
                "surprises": [],
                "recommendation": "r",
            }
        )
        assert p.themes == []
        assert p.agreements == []

    def test_real_lists_pass_through_unchanged(self):
        p = PartialSummary.model_validate(
            {
                "summary": "s",
                "themes": ["- keeps literal dash items intact"],
                "agreements": ["a"],
                "disagreements": ["d"],
                "surprises": ["sp"],
                "recommendation": "r",
            }
        )
        assert p.themes == ["- keeps literal dash items intact"]


class TestMapReduceToleratesStringListFields:
    """End-to-end regression through synthesize_panel_mapreduce: a map
    call whose SynthesisResult carries newline-bulleted strings for the
    list fields must NOT raise MapPhaseFailure; the coerced lists are
    written back onto the map result before reduce."""

    def test_string_valued_map_partial_parses_and_reduce_runs(self):
        from synth_panel import synthesis

        map_results: list[SynthesisResult] = []

        def _fake_synth(client, panelists, questions, **kwargs):
            q_text = questions[0].get("text") if isinstance(questions[0], dict) else str(questions[0])
            if kwargs.get("custom_prompt") == synthesis._REDUCE_PROMPT_TEMPLATE:
                return SynthesisResult(
                    summary="reduced",
                    themes=["cross-question theme"],
                    agreements=["a"],
                    disagreements=["d"],
                    surprises=["s"],
                    recommendation="r",
                )
            res = SynthesisResult(
                summary=f"summary-for-{q_text}",
                themes="\n- Distinctiveness and identity\n- Pricing anxiety",  # type: ignore[arg-type]
                agreements="- broad agreement",  # type: ignore[arg-type]
                disagreements="",  # type: ignore[arg-type]
                surprises="\n- One surprise",  # type: ignore[arg-type]
                recommendation="rec",
            )
            map_results.append(res)
            return res

        with patch.object(synthesis, "synthesize_panel", side_effect=_fake_synth):
            result = synthesize_panel_mapreduce(
                object(),
                _PANELISTS,
                _QUESTIONS,
                model="sonnet",
                max_workers=1,
            )
        assert isinstance(result, SynthesisResult)
        assert result.summary == "reduced"
        # Coerced lists were written back onto each map partial.
        for res in map_results:
            assert res.themes == ["Distinctiveness and identity", "Pricing anxiety"]
            assert res.agreements == ["broad agreement"]
            assert res.disagreements == []
            assert res.surprises == ["One surprise"]
