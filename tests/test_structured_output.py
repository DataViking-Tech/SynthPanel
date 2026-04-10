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
)

_SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "summary": {"type": "string"},
    },
    "required": ["sentiment", "summary"],
}


def _make_response_with_tool(data: dict) -> CompletionResponse:
    return CompletionResponse(
        id="r1",
        model="test",
        content=[
            ToolInvocationBlock(id="tc1", name="respond", input=data),
        ],
        usage=TokenUsage(input_tokens=10, output_tokens=20),
    )


def _make_text_response() -> CompletionResponse:
    return CompletionResponse(
        id="r1",
        model="test",
        content=[TextBlock(text="Just some text, no tool call")],
        usage=TokenUsage(input_tokens=10, output_tokens=20),
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
