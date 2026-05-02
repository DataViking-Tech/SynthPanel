"""Tests for structured output engine (SPEC.md §5)."""

from __future__ import annotations

from unittest.mock import MagicMock

from synth_panel.llm.models import (
    CompletionResponse,
    InputMessage,
    TextBlock,
    TokenUsage,
    ToolInvocationBlock,
)
from synth_panel.structured.output import (
    StructuredOutputConfig,
    StructuredOutputEngine,
    _is_cheap_model,
    _missing_required,
)

_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "summary": {"type": "string"},
    },
    "required": ["sentiment", "summary"],
}


def _make_response_with_tool(data: dict, *, usage: TokenUsage | None = None) -> CompletionResponse:
    return CompletionResponse(
        id="r1",
        model="test",
        content=[
            ToolInvocationBlock(id="tc1", name="respond", input=data),
        ],
        usage=usage or TokenUsage(input_tokens=10, output_tokens=20),
    )


def _make_text_response(*, usage: TokenUsage | None = None) -> CompletionResponse:
    return CompletionResponse(
        id="r1",
        model="test",
        content=[TextBlock(text="Just some text, no tool call")],
        usage=usage or TokenUsage(input_tokens=10, output_tokens=20),
    )


def _messages():
    return [InputMessage(role="user", content=[TextBlock(text="Analyze this")])]


class TestStructuredOutputEngine:
    def test_successful_extraction(self):
        """Happy path: LLM responds with valid tool call on first try."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_response_with_tool(
            {
                "sentiment": "positive",
                "summary": "Great product!",
            }
        )

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.data == {"sentiment": "positive", "summary": "Great product!"}
        assert result.retries_used == 0
        assert result.is_fallback is False
        assert mock_client.send.call_count == 1

        # Verify tool_choice was set to specific
        call_request = mock_client.send.call_args[0][0]
        assert call_request.tool_choice is not None
        assert call_request.tool_choice.name == "respond"

    def test_retry_on_no_tool_call(self):
        """If LLM doesn't produce a tool call, engine retries."""
        mock_client = MagicMock()
        mock_client.send.side_effect = [
            _make_text_response(),  # Attempt 1: no tool call
            _make_response_with_tool(
                {  # Attempt 2: valid
                    "sentiment": "neutral",
                    "summary": "OK",
                }
            ),
        ]

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.data == {"sentiment": "neutral", "summary": "OK"}
        assert result.retries_used == 1
        assert mock_client.send.call_count == 2

    def test_fallback_after_retries_exhausted(self):
        """After all retries, returns a fallback result with error."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_text_response()

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=1)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.is_fallback is True
        assert "_error" in result.data
        assert "_fallback" in result.data
        assert result.error is not None
        assert mock_client.send.call_count == 2  # 1 initial + 1 retry

    def test_disabled_skips_tool_forcing(self):
        """When disabled, does a plain completion without tool forcing."""
        mock_client = MagicMock()
        mock_client.send.return_value = _make_text_response()

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, enabled=False)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.data == {}
        call_request = mock_client.send.call_args[0][0]
        assert call_request.tools is None
        assert call_request.tool_choice is None

    def test_custom_tool_name(self):
        """Custom tool name is used in tool_choice."""
        mock_client = MagicMock()
        mock_client.send.return_value = CompletionResponse(
            id="r1",
            model="test",
            content=[ToolInvocationBlock(id="tc1", name="analyze", input={"score": 5})],
            usage=TokenUsage(),
        )

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(
            schema={"type": "object", "properties": {"score": {"type": "integer"}}},
            tool_name="analyze",
        )

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.data == {"score": 5}
        call_request = mock_client.send.call_args[0][0]
        assert call_request.tool_choice.name == "analyze"
        assert call_request.tools[0].name == "analyze"


class TestSchemaConformanceRetry:
    """sp-d1x0: 3-strike retry for schema non-conformance."""

    def test_retry_on_missing_required_field(self):
        """Tool call with missing required field triggers retry."""
        mock_client = MagicMock()
        # First attempt: returns tool call missing 'summary'
        partial = _make_response_with_tool({"sentiment": "positive"})
        # Second attempt: full schema-conformant response
        full = _make_response_with_tool({"sentiment": "positive", "summary": "Good"})
        mock_client.send.side_effect = [partial, full]

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        result = engine.extract(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.data == {"sentiment": "positive", "summary": "Good"}
        assert result.retries_used == 1
        assert result.is_fallback is False
        assert mock_client.send.call_count == 2

    def test_corrective_messages_appended_on_retry(self):
        """Retry attempt uses extended message list with corrective turn."""
        mock_client = MagicMock()
        partial = _make_response_with_tool({"sentiment": "positive"})  # missing 'summary'
        full = _make_response_with_tool({"sentiment": "positive", "summary": "Good"})
        mock_client.send.side_effect = [partial, full]

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        # First call: original messages only
        first_req = mock_client.send.call_args_list[0][0][0]
        assert len(first_req.messages) == 1

        # Second call: original + assistant (failed) + user (correction) = 3 messages
        second_req = mock_client.send.call_args_list[1][0][0]
        assert len(second_req.messages) == 3
        assert second_req.messages[1].role == "assistant"
        assert second_req.messages[2].role == "user"

    def test_escalates_to_sonnet_on_final_attempt_for_cheap_model(self):
        """Final strike uses sonnet when original model is in the cheap tier."""
        mock_client = MagicMock()
        bad = _make_response_with_tool({"sentiment": "positive"})  # missing 'summary'
        mock_client.send.return_value = bad

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        result = engine.extract(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.is_fallback is True
        assert mock_client.send.call_count == 3

        # First two calls use haiku; final call uses sonnet
        models_used = [call[0][0].model for call in mock_client.send.call_args_list]
        assert models_used[0] == "claude-haiku-4-5-20251001"
        assert models_used[1] == "claude-haiku-4-5-20251001"
        assert models_used[2] == "sonnet"

    def test_no_model_escalation_for_non_cheap_model(self):
        """When model is not cheap, all attempts use the same model."""
        mock_client = MagicMock()
        bad = _make_response_with_tool({"sentiment": "positive"})  # missing 'summary'
        mock_client.send.return_value = bad

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        engine.extract(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        models_used = [call[0][0].model for call in mock_client.send.call_args_list]
        assert all(m == "claude-sonnet-4-6" for m in models_used)

    def test_total_usage_accumulates_across_retries(self):
        """total_usage sums token counts from all retry attempts."""
        usage_a = TokenUsage(input_tokens=10, output_tokens=5)
        usage_b = TokenUsage(input_tokens=12, output_tokens=8)

        mock_client = MagicMock()
        mock_client.send.side_effect = [
            _make_response_with_tool({"sentiment": "positive"}, usage=usage_a),  # missing field
            _make_response_with_tool({"sentiment": "positive", "summary": "OK"}, usage=usage_b),
        ]

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.total_usage.input_tokens == 22  # 10 + 12
        assert result.total_usage.output_tokens == 13  # 5 + 8
        assert result.retries_used == 1

    def test_total_usage_accumulates_on_terminal_failure(self):
        """total_usage is cumulative even when all retries fail."""
        usage_per_call = TokenUsage(input_tokens=10, output_tokens=5)
        mock_client = MagicMock()
        mock_client.send.return_value = _make_response_with_tool(
            {"sentiment": "positive"},
            usage=usage_per_call,  # missing 'summary'
        )

        engine = StructuredOutputEngine(mock_client)
        config = StructuredOutputConfig(schema=_SENTIMENT_SCHEMA, retry_limit=2)

        result = engine.extract(
            model="sonnet",
            max_tokens=1024,
            messages=_messages(),
            config=config,
        )

        assert result.is_fallback is True
        assert result.total_usage.input_tokens == 30  # 3 attempts × 10
        assert result.total_usage.output_tokens == 15  # 3 attempts × 5


class TestIsCheapModel:
    def test_haiku_is_cheap(self):
        assert _is_cheap_model("haiku") is True
        assert _is_cheap_model("claude-haiku-4-5-20251001") is True

    def test_flash_is_cheap(self):
        assert _is_cheap_model("gemini-2.5-flash") is True
        assert _is_cheap_model("gemini-2.5-flash-lite") is True

    def test_sonnet_is_not_cheap(self):
        assert _is_cheap_model("sonnet") is False
        assert _is_cheap_model("claude-sonnet-4-6") is False

    def test_opus_is_not_cheap(self):
        assert _is_cheap_model("opus") is False
        assert _is_cheap_model("claude-opus-4-6") is False


class TestMissingRequired:
    def test_all_present(self):
        assert _missing_required({"a": 1, "b": 2}, {"required": ["a", "b"]}) == []

    def test_one_missing(self):
        assert _missing_required({"a": 1}, {"required": ["a", "b"]}) == ["b"]

    def test_no_required_in_schema(self):
        assert _missing_required({"a": 1}, {}) == []
