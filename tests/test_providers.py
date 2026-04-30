"""Tests for LLM provider implementations and shared OpenAI format helpers.

Covers: _openai_format.py, anthropic.py, gemini.py, openai_compat.py, xai.py
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    InputMessage,
    StopReason,
    StreamEventType,
    TextBlock,
    ToolChoice,
    ToolDefinition,
    ToolInvocationBlock,
    ToolResultBlock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_request(model: str = "test-model") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=100,
        messages=[InputMessage(role="user", content=[TextBlock(text="Hello")])],
    )


def _openai_json_response(text: str = "Hi there") -> dict:
    return {
        "id": "chatcmpl-123",
        "model": "test-model",
        "choices": [
            {
                "message": {"content": text, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _anthropic_json_response(text: str = "Hi there") -> dict:
    return {
        "id": "msg_123",
        "model": "claude-sonnet-4-6-20250414",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _mock_httpx_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


# ---------------------------------------------------------------------------
# Tests: _openai_format.py — build_openai_body
# ---------------------------------------------------------------------------


class TestBuildOpenaiBody:
    """Test OpenAI request body serialization."""

    def test_simple_message(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = _simple_request()
        body = build_openai_body(req)
        assert body["model"] == "test-model"
        assert body["max_tokens"] == 100
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    def test_system_prompt(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            system="You are helpful.",
        )
        body = build_openai_body(req)
        assert body["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert body["messages"][1]["role"] == "user"

    def test_assistant_with_tool_calls(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[
                InputMessage(
                    role="assistant",
                    content=[
                        ToolInvocationBlock(id="tc_1", name="search", input={"q": "test"}),
                    ],
                ),
            ],
        )
        body = build_openai_body(req)
        msg = body["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}

    def test_assistant_tool_calls_with_text(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[
                InputMessage(
                    role="assistant",
                    content=[
                        TextBlock(text="Let me search."),
                        ToolInvocationBlock(id="tc_1", name="search", input={"q": "x"}),
                    ],
                ),
            ],
        )
        body = build_openai_body(req)
        msg = body["messages"][0]
        assert msg["content"] == "Let me search."
        assert len(msg["tool_calls"]) == 1

    def test_tool_result_messages(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[
                InputMessage(
                    role="user",
                    content=[
                        ToolResultBlock(
                            tool_use_id="tc_1",
                            content=[TextBlock(text="Result data")],
                        ),
                    ],
                ),
            ],
        )
        body = build_openai_body(req)
        msg = body["messages"][0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "tc_1"
        assert msg["content"] == "Result data"

    def test_tools_serialized(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            tools=[ToolDefinition(name="calc", input_schema={"type": "object"}, description="A calculator")],
        )
        body = build_openai_body(req)
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "calc"
        assert body["tools"][0]["function"]["description"] == "A calculator"

    def test_tool_choice_auto(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            tool_choice=ToolChoice.auto(),
        )
        body = build_openai_body(req)
        assert body["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            tool_choice=ToolChoice.any(),
        )
        body = build_openai_body(req)
        assert body["tool_choice"] == "required"

    def test_tool_choice_specific(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        req = CompletionRequest(
            model="test",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            tool_choice=ToolChoice.specific("calc"),
        )
        body = build_openai_body(req)
        assert body["tool_choice"]["function"]["name"] == "calc"

    def test_stream_flag(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        body = build_openai_body(_simple_request(), stream=True)
        assert body["stream"] is True

    def test_no_stream_flag_by_default(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        body = build_openai_body(_simple_request())
        assert "stream" not in body


# ---------------------------------------------------------------------------
# Tests: _openai_format.py — parse_openai_response
# ---------------------------------------------------------------------------


class TestParseOpenaiResponse:
    """Test OpenAI response parsing."""

    def test_text_response(self):
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = _openai_json_response("Hello!")
        resp = parse_openai_response(data, "test-model")
        assert resp.id == "chatcmpl-123"
        assert resp.text == "Hello!"
        assert resp.stop_reason == StopReason.END_TURN
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_tool_call_response(self):
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "chatcmpl-456",
            "model": "test-model",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"query": "test"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8},
        }
        resp = parse_openai_response(data, "test-model")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].input == {"query": "test"}
        assert resp.stop_reason == StopReason.TOOL_USE

    def test_malformed_tool_arguments(self):
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "m",
            "choices": [
                {
                    "message": {
                        "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "not-json"}}],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        resp = parse_openai_response(data, "m")
        assert resp.tool_calls[0].input == {"_raw": "not-json"}

    def test_max_tokens_stop(self):
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "m",
            "choices": [{"message": {"content": "trunc"}, "finish_reason": "length"}],
            "usage": {},
        }
        resp = parse_openai_response(data, "m")
        assert resp.stop_reason == StopReason.MAX_TOKENS

    def test_openrouter_cost_captured(self):
        """sp-j3vk: OpenRouter's usage.cost must flow into TokenUsage."""
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "openai/gpt-4o-mini",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost": 0.000123,
            },
        }
        resp = parse_openai_response(data, "openai/gpt-4o-mini")
        assert float(resp.usage.provider_reported_cost) == pytest.approx(0.000123)

    def test_openrouter_cost_details_upstream_inference_cost(self):
        """Prefer cost_details.upstream_inference_cost over absent top-level cost."""
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_details": {"upstream_inference_cost": 0.000456},
            },
        }
        resp = parse_openai_response(data, "m")
        assert float(resp.usage.provider_reported_cost) == pytest.approx(0.000456)

    def test_openrouter_cost_details_prompt_plus_completion(self):
        """Fall back to prompt+completion cost sum if upstream total is missing."""
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cost_details": {
                    "upstream_inference_prompt_cost": 0.0003,
                    "upstream_inference_completions_cost": 0.0004,
                },
            },
        }
        resp = parse_openai_response(data, "m")
        assert float(resp.usage.provider_reported_cost) == pytest.approx(0.0007)

    def test_reasoning_and_cached_tokens_captured(self):
        """sp-loil: reasoning_tokens and cached_tokens subcounts land in TokenUsage."""
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "openai/gpt-5-mini",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 180},
                "completion_tokens_details": {"reasoning_tokens": 30},
            },
        }
        resp = parse_openai_response(data, "openai/gpt-5-mini")
        assert resp.usage.cached_tokens == 180
        assert resp.usage.reasoning_tokens == 30
        # Sub-counts do NOT get promoted into cache_read_tokens, otherwise
        # local cost estimates would double-bill on non-OR providers.
        assert resp.usage.cache_read_tokens == 0

    def test_missing_cost_is_none_not_zero(self):
        """Absent provider cost must be None so resolve_cost falls back to table."""
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data = {
            "id": "x",
            "model": "m",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        resp = parse_openai_response(data, "m")
        assert resp.usage.provider_reported_cost is None


# ---------------------------------------------------------------------------
# Tests: _openai_format.py — parse_openai_sse_stream
# ---------------------------------------------------------------------------


class TestParseOpenaiSseStream:
    """Test OpenAI SSE stream parsing."""

    def test_text_stream(self):
        from synth_panel.llm.providers._openai_format import parse_openai_sse_stream

        lines = iter(
            [
                'data: {"choices":[{"delta":{"content":"Hel"},"index":0}]}',
                "",
                'data: {"choices":[{"delta":{"content":"lo"},"index":0}]}',
                "",
                'data: {"choices":[{"finish_reason":"stop","delta":{}}]}',
                "",
                "data: [DONE]",
            ]
        )
        events = list(parse_openai_sse_stream(lines))
        assert events[0].type == StreamEventType.CONTENT_BLOCK_DELTA
        assert events[0].data["text"] == "Hel"
        assert events[1].type == StreamEventType.CONTENT_BLOCK_DELTA
        assert events[1].data["text"] == "lo"
        assert events[2].type == StreamEventType.MESSAGE_DELTA
        assert events[3].type == StreamEventType.MESSAGE_STOP

    def test_tool_call_stream(self):
        from synth_panel.llm.providers._openai_format import parse_openai_sse_stream

        lines = iter(
            [
                'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1","function":{"name":"f"}}]},"index":0}]}',
                "",
                "data: [DONE]",
            ]
        )
        events = list(parse_openai_sse_stream(lines))
        assert events[0].type == StreamEventType.CONTENT_BLOCK_DELTA
        assert "tool_calls" in events[0].data

    def test_comment_lines_ignored(self):
        from synth_panel.llm.providers._openai_format import parse_openai_sse_stream

        lines = iter(
            [
                ": keepalive",
                'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}',
                "",
                "data: [DONE]",
            ]
        )
        events = list(parse_openai_sse_stream(lines))
        assert events[0].type == StreamEventType.CONTENT_BLOCK_DELTA

    def test_no_choices_emits_message_start(self):
        from synth_panel.llm.providers._openai_format import parse_openai_sse_stream

        lines = iter(
            [
                'data: {"id":"chatcmpl-abc","object":"chat.completion.chunk"}',
                "",
                "data: [DONE]",
            ]
        )
        events = list(parse_openai_sse_stream(lines))
        assert events[0].type == StreamEventType.MESSAGE_START

    def test_malformed_json_skipped(self):
        from synth_panel.llm.providers._openai_format import parse_openai_sse_stream

        lines = iter(
            [
                "data: {not valid json}",
                "",
                'data: {"choices":[{"delta":{"content":"ok"},"index":0}]}',
                "",
                "data: [DONE]",
            ]
        )
        events = list(parse_openai_sse_stream(lines))
        assert len(events) >= 1
        assert events[0].data["text"] == "ok"


# ---------------------------------------------------------------------------
# Tests: Anthropic provider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    """Test the Anthropic provider send/stream and helpers."""

    def test_send_success(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        mock_resp = _mock_httpx_response(_anthropic_json_response("Hello!"))
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("claude-sonnet-4-6-20250414"))
        assert result.text == "Hello!"
        assert result.usage.input_tokens == 10

    def test_send_with_tool_use_response(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        data = {
            "id": "msg_1",
            "model": "claude-sonnet-4-6-20250414",
            "content": [{"type": "tool_use", "id": "tu_1", "name": "calc", "input": {"x": 1}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        mock_resp = _mock_httpx_response(data)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("claude-sonnet-4-6-20250414"))
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calc"
        assert result.stop_reason == StopReason.TOOL_USE

    def test_send_with_thinking_block(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        data = {
            "id": "msg_2",
            "model": "claude-sonnet-4-6-20250414",
            "content": [
                {"type": "thinking", "thinking": "Let me think...", "signature": "sig123"},
                {"type": "text", "text": "Answer"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        mock_resp = _mock_httpx_response(data)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("claude-sonnet-4-6-20250414"))
        assert result.text == "Answer"
        assert len(result.content) == 2

    def test_send_transport_error(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("timeout")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("claude-sonnet-4-6-20250414"))
            assert exc_info.value.category == LLMErrorCategory.TRANSPORT

    def test_send_non_200_status(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        mock_resp = _mock_httpx_response({}, status_code=429)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("claude-sonnet-4-6-20250414"))
            assert exc_info.value.category == LLMErrorCategory.RATE_LIMIT

    def test_send_json_decode_error(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("claude-sonnet-4-6-20250414"))
            assert exc_info.value.category == LLMErrorCategory.DESERIALIZATION

    def test_missing_api_key(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(LLMError) as exc_info:
                AnthropicProvider()
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS

    def test_cache_token_usage(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        data = {
            "id": "msg_3",
            "model": "claude-sonnet-4-6-20250414",
            "content": [{"type": "text", "text": "Cached response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 80,
            },
        }
        mock_resp = _mock_httpx_response(data)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("claude-sonnet-4-6-20250414"))
        assert result.usage.cache_write_tokens == 200
        assert result.usage.cache_read_tokens == 80

    def test_build_body_with_system_and_tools(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
            system="Be helpful.",
            tools=[ToolDefinition(name="calc", input_schema={"type": "object"})],
            tool_choice=ToolChoice.any(),
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        body = provider._build_body(req)
        assert body["system"] == [{"type": "text", "text": "Be helpful.", "cache_control": {"type": "ephemeral"}}]
        assert len(body["tools"]) == 1
        assert body["tool_choice"] == {"type": "any"}

    def test_cache_control_on_system_prompt(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hello")])],
            system="You are a helpful assistant.",
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        body = provider._build_body(req)
        system = body["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["type"] == "text"
        assert system[0]["text"] == "You are a helpful assistant."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_omits_system_key(self):
        from synth_panel.llm.providers.anthropic import AnthropicProvider

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hello")])],
        )
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = AnthropicProvider()
        body = provider._build_body(req)
        assert "system" not in body

    def test_cache_control_on_last_user_message_last_text_block(self):
        from synth_panel.llm.providers.anthropic import _build_messages

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[
                InputMessage(role="user", content=[TextBlock(text="First question")]),
                InputMessage(role="assistant", content=[TextBlock(text="First answer")]),
                InputMessage(role="user", content=[TextBlock(text="Second question")]),
            ],
        )
        messages = _build_messages(req)
        last_user_content = messages[2]["content"]
        assert last_user_content[-1]["cache_control"] == {"type": "ephemeral"}
        # Earlier user message must NOT have cache_control
        first_user_content = messages[0]["content"]
        assert "cache_control" not in first_user_content[0]

    def test_cache_control_only_on_last_user_message(self):
        from synth_panel.llm.providers.anthropic import _build_messages

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[
                InputMessage(role="user", content=[TextBlock(text="Q1")]),
            ],
        )
        messages = _build_messages(req)
        assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_not_on_assistant_messages(self):
        from synth_panel.llm.providers.anthropic import _build_messages

        req = CompletionRequest(
            model="claude-sonnet-4-6-20250414",
            max_tokens=100,
            messages=[
                InputMessage(role="user", content=[TextBlock(text="Q1")]),
                InputMessage(role="assistant", content=[TextBlock(text="A1")]),
            ],
        )
        messages = _build_messages(req)
        assistant_content = messages[1]["content"]
        assert "cache_control" not in assistant_content[0]


# ---------------------------------------------------------------------------
# Tests: Anthropic SSE stream parsing
# ---------------------------------------------------------------------------


class TestAnthropicStream:
    """Test Anthropic SSE stream parsing."""

    def test_parse_sse_stream(self):
        from synth_panel.llm.providers.anthropic import _parse_sse_stream

        lines = iter(
            [
                'data: {"type":"message_start","message":{"id":"msg_1"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"text":"Hi"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        )
        events = list(_parse_sse_stream(lines))
        assert events[0].type == StreamEventType.MESSAGE_START
        assert events[1].type == StreamEventType.CONTENT_BLOCK_DELTA
        assert events[2].type == StreamEventType.MESSAGE_STOP

    def test_ping_events_skipped(self):
        from synth_panel.llm.providers.anthropic import _parse_sse_stream

        lines = iter(
            [
                'data: {"type":"ping"}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        )
        events = list(_parse_sse_stream(lines))
        assert len(events) == 1
        assert events[0].type == StreamEventType.MESSAGE_STOP

    def test_done_sentinel(self):
        from synth_panel.llm.providers.anthropic import _parse_sse_stream

        lines = iter(
            [
                "data: [DONE]",
                "",
            ]
        )
        events = list(_parse_sse_stream(lines))
        assert events == []

    def test_comment_keepalive_ignored(self):
        from synth_panel.llm.providers.anthropic import _parse_sse_stream

        lines = iter(
            [
                ": keepalive",
                'data: {"type":"message_stop"}',
                "",
            ]
        )
        events = list(_parse_sse_stream(lines))
        assert len(events) == 1

    def test_unknown_event_type_skipped(self):
        from synth_panel.llm.providers.anthropic import _sse_payload_to_event

        result = _sse_payload_to_event({"type": "unknown_event_xyz"})
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Anthropic helper functions
# ---------------------------------------------------------------------------


class TestAnthropicHelpers:
    """Test Anthropic serialization helpers."""

    def test_build_tool_choice_auto(self):
        from synth_panel.llm.providers.anthropic import _build_tool_choice

        req = CompletionRequest(
            model="m",
            max_tokens=1,
            messages=[],
            tool_choice=ToolChoice.auto(),
        )
        assert _build_tool_choice(req) == {"type": "auto"}

    def test_build_tool_choice_specific(self):
        from synth_panel.llm.providers.anthropic import _build_tool_choice

        req = CompletionRequest(
            model="m",
            max_tokens=1,
            messages=[],
            tool_choice=ToolChoice.specific("calc"),
        )
        assert _build_tool_choice(req) == {"type": "tool", "name": "calc"}

    def test_build_tool_choice_none(self):
        from synth_panel.llm.providers.anthropic import _build_tool_choice

        req = CompletionRequest(model="m", max_tokens=1, messages=[])
        assert _build_tool_choice(req) is None

    def test_build_content_blocks_with_tool_result(self):
        from synth_panel.llm.providers.anthropic import _build_content_blocks

        blocks = [
            TextBlock(text="Hello"),
            ToolResultBlock(
                tool_use_id="tu_1",
                content=[TextBlock(text="Result")],
                is_error=True,
            ),
        ]
        result = _build_content_blocks(blocks)
        assert result[0] == {"type": "text", "text": "Hello"}
        assert result[1]["type"] == "tool_result"
        assert result[1]["tool_use_id"] == "tu_1"
        assert result[1]["is_error"] is True

    def test_parse_content_block_unknown_type(self):
        from synth_panel.llm.providers.anthropic import _parse_content_block

        block = _parse_content_block({"type": "mystery", "data": 42})
        assert isinstance(block, TextBlock)
        assert "mystery" in block.text

    def test_parse_stop_reason_unknown(self):
        from synth_panel.llm.providers.anthropic import _parse_stop_reason

        assert _parse_stop_reason("something_weird") == StopReason.END_TURN

    def test_parse_stop_reason_none(self):
        from synth_panel.llm.providers.anthropic import _parse_stop_reason

        assert _parse_stop_reason(None) is None


# ---------------------------------------------------------------------------
# Tests: Gemini provider
# ---------------------------------------------------------------------------


class TestGeminiProvider:
    """Test the Gemini provider."""

    def test_send_success(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        mock_resp = _mock_httpx_response(_openai_json_response("Gemini says hi"))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk-test"}, clear=False):
            provider = GeminiProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("gemini-2.5-flash"))
        assert result.text == "Gemini says hi"

    def test_google_api_key_fallback(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        env = {"GOOGLE_API_KEY": "google-key"}
        with patch.dict(os.environ, env, clear=True):
            provider = GeminiProvider()
        assert provider._api_key == "google-key"

    def test_missing_both_keys(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError) as exc_info:
                GeminiProvider()
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS

    def test_send_transport_error(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk-test"}, clear=False):
            provider = GeminiProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("fail")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gemini-2.5-flash"))
            assert exc_info.value.category == LLMErrorCategory.TRANSPORT

    def test_send_non_200(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        mock_resp = _mock_httpx_response({}, status_code=500)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk-test"}, clear=False):
            provider = GeminiProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gemini-2.5-flash"))
            assert exc_info.value.category == LLMErrorCategory.SERVER_ERROR

    def test_send_json_error(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk-test"}, clear=False):
            provider = GeminiProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gemini-2.5-flash"))
            assert exc_info.value.category == LLMErrorCategory.DESERIALIZATION

    def test_headers_use_bearer(self):
        from synth_panel.llm.providers.gemini import GeminiProvider

        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk-test"}, clear=False):
            provider = GeminiProvider()
        headers = provider._headers()
        assert headers["Authorization"] == "Bearer gk-test"


# ---------------------------------------------------------------------------
# Tests: OpenAI-compatible provider
# ---------------------------------------------------------------------------


class TestOpenAICompatProvider:
    """Test the OpenAI-compatible provider."""

    def test_send_success(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        mock_resp = _mock_httpx_response(_openai_json_response("OpenAI says hi"))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai"}, clear=False):
            provider = OpenAICompatibleProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("gpt-4o"))
        assert result.text == "OpenAI says hi"

    def test_missing_api_key(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError) as exc_info:
                OpenAICompatibleProvider()
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS

    def test_send_transport_error(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai"}, clear=False):
            provider = OpenAICompatibleProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("fail")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gpt-4o"))
            assert exc_info.value.category == LLMErrorCategory.TRANSPORT

    def test_send_non_200(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        mock_resp = _mock_httpx_response({}, status_code=401)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai"}, clear=False):
            provider = OpenAICompatibleProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gpt-4o"))
            assert exc_info.value.category == LLMErrorCategory.AUTHENTICATION

    def test_send_json_error(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai"}, clear=False):
            provider = OpenAICompatibleProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("gpt-4o"))
            assert exc_info.value.category == LLMErrorCategory.DESERIALIZATION

    def test_custom_base_url(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        with patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-oai", "OPENAI_BASE_URL": "http://localhost:8080"}, clear=False
        ):
            provider = OpenAICompatibleProvider()
        assert provider._base_url == "http://localhost:8080"


# ---------------------------------------------------------------------------
# Tests: xAI provider
# ---------------------------------------------------------------------------


class TestXAIProvider:
    """Test the xAI / Grok provider."""

    def test_send_success(self):
        from synth_panel.llm.providers.xai import XAIProvider

        mock_resp = _mock_httpx_response(_openai_json_response("Grok says hi"))
        with patch.dict(os.environ, {"XAI_API_KEY": "xk-test"}, clear=False):
            provider = XAIProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("grok-3"))
        assert result.text == "Grok says hi"

    def test_missing_api_key(self):
        from synth_panel.llm.providers.xai import XAIProvider

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError) as exc_info:
                XAIProvider()
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS

    def test_send_transport_error(self):
        from synth_panel.llm.providers.xai import XAIProvider

        with patch.dict(os.environ, {"XAI_API_KEY": "xk-test"}, clear=False):
            provider = XAIProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("fail")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("grok-3"))
            assert exc_info.value.category == LLMErrorCategory.TRANSPORT

    def test_send_non_200(self):
        from synth_panel.llm.providers.xai import XAIProvider

        mock_resp = _mock_httpx_response({}, status_code=500)
        with patch.dict(os.environ, {"XAI_API_KEY": "xk-test"}, clear=False):
            provider = XAIProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("grok-3"))
            assert exc_info.value.category == LLMErrorCategory.SERVER_ERROR

    def test_send_json_error(self):
        from synth_panel.llm.providers.xai import XAIProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch.dict(os.environ, {"XAI_API_KEY": "xk-test"}, clear=False):
            provider = XAIProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("grok-3"))
            assert exc_info.value.category == LLMErrorCategory.DESERIALIZATION

    def test_headers_use_bearer(self):
        from synth_panel.llm.providers.xai import XAIProvider

        with patch.dict(os.environ, {"XAI_API_KEY": "xk-test"}, clear=False):
            provider = XAIProvider()
        headers = provider._headers()
        assert headers["Authorization"] == "Bearer xk-test"


# ---------------------------------------------------------------------------
# Tests: OpenRouter provider
# ---------------------------------------------------------------------------


class TestOpenRouterProvider:
    """Test the OpenRouter provider."""

    def test_send_success(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = _mock_httpx_response(_openai_json_response("OpenRouter says hi"))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/meta-llama/llama-3"))
        assert result.text == "OpenRouter says hi"

    def test_missing_api_key(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LLMError) as exc_info:
                OpenRouterProvider()
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS

    def test_send_transport_error(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("fail")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("openrouter/meta-llama/llama-3"))
            assert exc_info.value.category == LLMErrorCategory.TRANSPORT

    def test_send_non_200(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = _mock_httpx_response({}, status_code=500)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("openrouter/meta-llama/llama-3"))
            assert exc_info.value.category == LLMErrorCategory.SERVER_ERROR

    def test_send_json_error(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_simple_request("openrouter/meta-llama/llama-3"))
            assert exc_info.value.category == LLMErrorCategory.DESERIALIZATION

    def test_default_base_url(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        assert provider._base_url == "https://openrouter.ai/api"

    def test_custom_base_url(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        env = {"OPENROUTER_API_KEY": "or-test", "OPENROUTER_BASE_URL": "http://custom:8080"}
        with patch.dict(os.environ, env, clear=False):
            provider = OpenRouterProvider()
        assert provider._base_url == "http://custom:8080"

    def test_headers_use_bearer(self):
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        headers = provider._headers()
        assert headers["Authorization"] == "Bearer or-test"

    def test_send_requests_detailed_usage(self):
        """sp-2xy: OpenRouter send() must set ``usage.include=true`` so cost data is reliable.

        Without this flag, some upstream providers omit ``usage`` entirely,
        which zeros out panelist_cost / total_cost in the panel JSON output.
        """
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = _mock_httpx_response(_openai_json_response("hi"))
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/meta-llama/llama-3"))
        body = mock_post.call_args.kwargs["json"]
        assert body.get("usage") == {"include": True}

    def test_stream_requests_detailed_usage(self):
        """sp-2xy: streaming requests must also set ``usage.include=true``."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        # Minimal mock of an httpx streaming context manager
        mock_stream_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(["data: [DONE]", ""])
        mock_stream_cm.__enter__.return_value = mock_resp
        mock_stream_cm.__exit__.return_value = False

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.stream", return_value=mock_stream_cm) as mock_stream:
            list(provider.stream(_simple_request("openrouter/meta-llama/llama-3")))
        body = mock_stream.call_args.kwargs["json"]
        assert body.get("usage") == {"include": True}

    def test_usage_captured_from_response(self):
        """sp-2xy: usage returned by OpenRouter must propagate into the CompletionResponse."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        payload = _openai_json_response("hello")
        payload["usage"] = {"prompt_tokens": 1234, "completion_tokens": 56, "total_tokens": 1290}
        mock_resp = _mock_httpx_response(payload)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))
        assert result.usage.input_tokens == 1234
        assert result.usage.output_tokens == 56

    def test_null_usage_does_not_crash(self):
        """sp-2xy: some providers return ``"usage": null`` — parser must tolerate it."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        payload = _openai_json_response("hello")
        payload["usage"] = None  # Defensive: provider returned null instead of a dict
        mock_resp = _mock_httpx_response(payload)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))
        # Zero usage is the correct fallback — no AttributeError, no silent crash.
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.text == "hello"

    def test_missing_usage_block_does_not_crash(self):
        """sp-2xy: some providers omit the ``usage`` block entirely."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        payload = _openai_json_response("hello")
        payload.pop("usage", None)
        mock_resp = _mock_httpx_response(payload)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_send_429_surfaces_downstream_provider_and_type(self):
        """sy-2185: typed error JSON exposes downstream provider + error type."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        body = {
            "error": {
                "code": 429,
                "message": "Rate limit exceeded for organization, please try again in 12 seconds",
                "type": "rate_limit_error",
                "metadata": {"provider_name": "anthropic"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = body
        mock_resp.text = json.dumps(body)
        mock_resp.headers = {"retry-after": "12"}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp), pytest.raises(LLMError) as exc_info:
            provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))

        err = exc_info.value
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert err.status_code == 429
        assert err.retry_after == 12.0
        msg = str(err)
        assert "OpenRouter" in msg
        assert "anthropic" in msg
        assert "rate_limit_error" in msg
        assert "Rate limit exceeded" in msg

    def test_send_429_generic_body_falls_back_gracefully(self):
        """sy-2185: a 429 without typed-error JSON must not crash."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = _mock_httpx_response({}, status_code=429)
        mock_resp.headers = {}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp), pytest.raises(LLMError) as exc_info:
            provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))

        err = exc_info.value
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert "OpenRouter" in str(err)

    def test_send_429_malformed_body_falls_back_gracefully(self):
        """sy-2185: non-JSON body on 429 must not crash error construction."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.side_effect = json.JSONDecodeError("bad", "", 0)
        mock_resp.text = "<html>Service unavailable</html>"
        mock_resp.headers = {}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp), pytest.raises(LLMError) as exc_info:
            provider.send(_simple_request("openrouter/anthropic/claude-haiku-4-5"))

        err = exc_info.value
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert "OpenRouter" in str(err)

    def test_send_500_typed_error_keeps_server_error_category(self):
        """sy-2185: enrichment must not change the HTTP-status-derived category."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        body = {
            "error": {
                "code": 500,
                "message": "Upstream model crashed",
                "type": "internal_error",
                "metadata": {"provider_name": "groq"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = body
        mock_resp.text = json.dumps(body)
        mock_resp.headers = {}

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.post", return_value=mock_resp), pytest.raises(LLMError) as exc_info:
            provider.send(_simple_request("openrouter/groq/llama-3"))

        err = exc_info.value
        assert err.category == LLMErrorCategory.SERVER_ERROR
        msg = str(err)
        assert "groq" in msg
        assert "internal_error" in msg

    def test_stream_429_surfaces_downstream_provider(self):
        """sy-2185: streaming path enriches errors the same as send()."""
        from synth_panel.llm.providers.openrouter import OpenRouterProvider

        body = {
            "error": {
                "code": 429,
                "message": "Slow down",
                "type": "rate_limit_error",
                "metadata": {"provider_name": "openai"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = body
        mock_resp.text = json.dumps(body)
        mock_resp.headers = {"retry-after": "5"}
        mock_resp.read.return_value = None

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__.return_value = mock_resp
        mock_stream_cm.__exit__.return_value = False

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
            provider = OpenRouterProvider()
        with patch("httpx.stream", return_value=mock_stream_cm), pytest.raises(LLMError) as exc_info:
            list(provider.stream(_simple_request("openrouter/openai/gpt-4")))

        err = exc_info.value
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert err.retry_after == 5.0
        msg = str(err)
        assert "openai" in msg
        assert "rate_limit_error" in msg


# ---------------------------------------------------------------------------
# Tests: OpenAI-compatible provider — constructor overrides (for local models)
# ---------------------------------------------------------------------------


class TestOpenAICompatOverrides:
    """Test OpenAICompatibleProvider with explicit base_url/api_key overrides."""

    def test_explicit_base_url(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(base_url="http://localhost:11434", api_key="no-key-required")
        assert provider._base_url == "http://localhost:11434"
        assert provider._api_key == "no-key-required"

    def test_send_with_overrides(self):
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(base_url="http://localhost:11434", api_key="no-key-required")
        mock_resp = _mock_httpx_response(_openai_json_response("Local model says hi"))
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = provider.send(_simple_request("llama3"))
        assert result.text == "Local model says hi"
        called_url = mock_post.call_args[0][0]
        assert called_url == "http://localhost:11434/v1/chat/completions"
