"""Tests for panel synthesis module."""

from __future__ import annotations

from unittest.mock import MagicMock

from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.models import (
    CompletionResponse,
    TextBlock,
    TokenUsage,
    ToolInvocationBlock,
)
from synth_panel.orchestrator import PanelistResult
from synth_panel.prompts import SYNTHESIS_PROMPT, SYNTHESIS_PROMPT_VERSION
from synth_panel.synthesis import (
    _SYNTHESIS_SCHEMA,
    SynthesisResult,
    _format_panelist_data,
    synthesize_panel,
)

# --- Test fixtures ---

_SYNTHESIS_DATA = {
    "summary": "Panelists agreed on productivity gains but diverged on pricing.",
    "themes": ["productivity", "pricing concerns", "ease of use"],
    "agreements": ["The tool saves time on repetitive tasks"],
    "disagreements": ["Whether the price is justified for small teams"],
    "surprises": ["Several panelists mentioned using it for unintended use cases"],
    "recommendation": "Focus messaging on productivity gains; consider a small-team tier.",
}

_QUESTIONS = [
    {"text": "What do you think of the product?"},
    {"text": "Would you recommend it?"},
]

_PANELIST_RESULTS = [
    PanelistResult(
        persona_name="Alice",
        responses=[
            {"question": "What do you think of the product?", "response": "I love the simplicity."},
            {"question": "Would you recommend it?", "response": "Yes, absolutely."},
        ],
        usage=ZERO_USAGE,
    ),
    PanelistResult(
        persona_name="Bob",
        responses=[
            {"question": "What do you think of the product?", "response": "It's decent but overpriced."},
            {"question": "Would you recommend it?", "response": "Only if they lower the price."},
        ],
        usage=ZERO_USAGE,
    ),
]


def _make_synthesis_response(data: dict) -> CompletionResponse:
    return CompletionResponse(
        id="synth-1",
        model="claude-sonnet-4-6",
        content=[
            ToolInvocationBlock(id="tc1", name="synthesize", input=data),
        ],
        usage=TokenUsage(input_tokens=500, output_tokens=200),
    )


def _make_text_response() -> CompletionResponse:
    return CompletionResponse(
        id="synth-1",
        model="claude-sonnet-4-6",
        content=[TextBlock(text="No tool call here")],
        usage=TokenUsage(input_tokens=500, output_tokens=200),
    )


# --- Tests ---


class TestSynthesizePanel:
    def test_successful_synthesis(self):
        """Happy path: LLM returns valid structured synthesis."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        assert isinstance(result, SynthesisResult)
        assert result.summary == _SYNTHESIS_DATA["summary"]
        assert result.themes == _SYNTHESIS_DATA["themes"]
        assert result.agreements == _SYNTHESIS_DATA["agreements"]
        assert result.disagreements == _SYNTHESIS_DATA["disagreements"]
        assert result.surprises == _SYNTHESIS_DATA["surprises"]
        assert result.recommendation == _SYNTHESIS_DATA["recommendation"]
        assert not result.is_fallback
        assert result.error is None

    def test_usage_tracked(self):
        """Synthesis cost is tracked separately."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        assert result.usage.input_tokens == 500
        assert result.usage.output_tokens == 200
        assert result.cost.total_cost > 0

    def test_default_model_is_sonnet(self):
        """When neither model nor panelist_model is set, defaults to sonnet."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        assert result.model == "sonnet"
        call_args = mock_client.send.call_args[0][0]
        assert call_args.model == "sonnet"

    def test_panelist_model_used_as_default(self):
        """When no explicit model, synthesis defaults to panelist model."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS, panelist_model="haiku")

        assert result.model == "haiku"
        call_args = mock_client.send.call_args[0][0]
        assert call_args.model == "haiku"

    def test_explicit_model_overrides_panelist_model(self):
        """Explicit --synthesis-model overrides panelist model."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(
            mock_client,
            _PANELIST_RESULTS,
            _QUESTIONS,
            model="opus",
            panelist_model="haiku",
        )

        assert result.model == "opus"
        call_args = mock_client.send.call_args[0][0]
        assert call_args.model == "opus"

    def test_custom_model(self):
        """Can override the synthesis model."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS, model="opus")

        assert result.model == "opus"
        call_args = mock_client.send.call_args[0][0]
        assert call_args.model == "opus"

    def test_custom_prompt(self):
        """Can override the synthesis prompt."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)
        custom = "Summarize this panel in haiku form."

        synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS, custom_prompt=custom)

        call_args = mock_client.send.call_args[0][0]
        user_text = call_args.messages[0].content[0].text
        assert custom in user_text
        assert SYNTHESIS_PROMPT not in user_text

    def test_fallback_on_no_tool_call(self):
        """When LLM fails to produce a tool call, returns fallback result."""
        mock_client = MagicMock()
        # All attempts return text-only (no tool call)
        mock_client.send.return_value = _make_text_response()

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        assert result.is_fallback
        assert result.error is not None
        assert result.summary == "Synthesis failed — see error field."
        assert result.themes == []

    def test_prompt_version_in_result(self):
        """synthesis_prompt_version is included in the result."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        assert result.synthesis_prompt_version == SYNTHESIS_PROMPT_VERSION

    def test_to_dict(self):
        """to_dict produces the expected output format."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        result = synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)
        d = result.to_dict()

        assert d["summary"] == _SYNTHESIS_DATA["summary"]
        assert d["themes"] == _SYNTHESIS_DATA["themes"]
        assert d["prompt_version"] == SYNTHESIS_PROMPT_VERSION
        assert "cost" in d
        assert "usage" in d
        assert "model" in d

    def test_uses_tool_use_forcing(self):
        """Verifies the LLM request uses tool-use forcing (tool_choice=specific)."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        call_args = mock_client.send.call_args[0][0]
        assert call_args.tools is not None
        assert len(call_args.tools) == 1
        assert call_args.tools[0].name == "synthesize"
        assert call_args.tool_choice is not None
        assert call_args.tool_choice.name == "synthesize"


class TestCostEstimate:
    def test_cost_estimate_printed_to_stderr(self, capsys):
        """Pre-synthesis cost estimate is printed to stderr."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        captured = capsys.readouterr()
        assert "Synthesis will cost ~$" in captured.err

    def test_cost_estimate_includes_panelist_cost(self, capsys):
        """When panelist_cost is provided, it appears in the estimate."""
        from synth_panel.cost import CostEstimate

        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        pc = CostEstimate(input_cost=0.001, output_cost=0.002)
        synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS, panelist_cost=pc)

        captured = capsys.readouterr()
        assert "panelist cost was $" in captured.err

    def test_cost_estimate_without_panelist_cost(self, capsys):
        """Without panelist_cost, the estimate omits the comparison."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_synthesis_response(_SYNTHESIS_DATA)

        synthesize_panel(mock_client, _PANELIST_RESULTS, _QUESTIONS)

        captured = capsys.readouterr()
        assert "Synthesis will cost ~$" in captured.err
        assert "panelist cost" not in captured.err


class TestFormatPanelistData:
    def test_basic_formatting(self):
        """Formats panelist data with questions and responses."""
        output = _format_panelist_data(_PANELIST_RESULTS, _QUESTIONS)

        assert "## Questions Asked" in output
        assert "What do you think of the product?" in output
        assert "## Panelist Responses" in output
        assert "### Alice" in output
        assert "### Bob" in output
        assert "I love the simplicity." in output
        assert "It's decent but overpriced." in output

    def test_follow_up_marked(self):
        """Follow-up responses are marked differently."""
        results = [
            PanelistResult(
                persona_name="Carol",
                responses=[
                    {"question": "Main Q", "response": "Answer"},
                    {"question": "Follow up", "response": "More detail", "follow_up": True},
                ],
                usage=ZERO_USAGE,
            ),
        ]

        output = _format_panelist_data(results, [{"text": "Main Q"}])

        assert "Follow-up: Follow up" in output

    def test_structured_response_serialized(self):
        """Dict responses are JSON-serialized."""
        results = [
            PanelistResult(
                persona_name="Dave",
                responses=[
                    {"question": "Rate it", "response": {"rating": 5, "reason": "great"}},
                ],
                usage=ZERO_USAGE,
            ),
        ]

        output = _format_panelist_data(results, [{"text": "Rate it"}])

        assert '"rating": 5' in output

    def test_string_questions(self):
        """Plain string questions are handled."""
        output = _format_panelist_data(_PANELIST_RESULTS, ["Simple question?"])

        assert "Simple question?" in output


class TestSynthesisSchema:
    def test_required_fields(self):
        """Schema requires all expected fields."""
        required = _SYNTHESIS_SCHEMA["required"]
        assert "summary" in required
        assert "themes" in required
        assert "agreements" in required
        assert "disagreements" in required
        assert "surprises" in required
        assert "recommendation" in required

    def test_array_fields(self):
        """themes, agreements, disagreements, surprises are arrays."""
        props = _SYNTHESIS_SCHEMA["properties"]
        for field_name in ["themes", "agreements", "disagreements", "surprises"]:
            assert props[field_name]["type"] == "array"

    def test_string_fields(self):
        """summary and recommendation are strings."""
        props = _SYNTHESIS_SCHEMA["properties"]
        assert props["summary"]["type"] == "string"
        assert props["recommendation"]["type"] == "string"
