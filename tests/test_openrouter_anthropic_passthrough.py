"""Regression coverage for hq-olrk: ``openrouter/anthropic/*`` routing.

The bug (hq-m333): synthpanel's OpenRouter provider sent every request,
including ``openrouter/anthropic/claude-sonnet-4.5``, through OR's
``/v1/chat/completions`` (OpenAI-shape) endpoint. For Anthropic-upstream
models, OR's downstream conversion from OpenAI ``image_url`` to Anthropic
image blocks is **lossy** — multimodal images silently drop, the model
sees text-only context, and the run looks successful (200 OK).

The fix routes ``openrouter/anthropic/*`` traffic through OR's
Anthropic-native ``/v1/messages`` passthrough with an Anthropic-shape
body. These tests lock in:

* the routing branch (URL + body shape)
* preservation of multimodal blocks through the passthrough
* the ``anthropic-version`` header (so cache_control etc. line up)
* non-Anthropic OR traffic is **unchanged** (no regression on
  ``openrouter/openai/*``, ``openrouter/google/*``, etc.)
* Anthropic-shape response parsing (text + native usage)
* SSE streaming through the passthrough emits Anthropic-native events

A regression here means image attachments silently drop again.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.llm.errors import LLMError
from synth_panel.llm.models import (
    CompletionRequest,
    ImageBlock,
    InlineSource,
    InputMessage,
    StreamEventType,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_request(model: str) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=100,
        messages=[InputMessage(role="user", content=[TextBlock(text="Hello")])],
    )


def _multimodal_request(model: str) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=512,
        messages=[
            InputMessage(
                role="user",
                content=[
                    ImageBlock(
                        source=InlineSource(data="ZmFrZWltYWdl"),
                        media_type="image/png",
                    ),
                    TextBlock(text="Describe the image."),
                ],
            )
        ],
    )


def _anthropic_json_response(text: str = "I see a screenshot") -> dict:
    return {
        "id": "msg_or_123",
        "model": "anthropic/claude-sonnet-4.5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 12},
    }


def _mock_httpx_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


def _provider():
    from synth_panel.llm.providers.openrouter import OpenRouterProvider

    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}, clear=False):
        return OpenRouterProvider()


# ---------------------------------------------------------------------------
# Routing branch: anthropic vs non-anthropic upstream
# ---------------------------------------------------------------------------


class TestAnthropicRoutingBranch:
    def test_anthropic_model_targets_messages_endpoint(self):
        """``openrouter/anthropic/*`` must POST to ``/v1/messages``."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        url = mock_post.call_args.args[0]
        assert url.endswith("/v1/messages"), (
            f"Anthropic-routed traffic hit {url!r} — must use /v1/messages "
            "passthrough or images will silently drop (hq-m333)"
        )

    def test_non_anthropic_model_keeps_chat_completions(self):
        """``openrouter/openai/*`` etc. must keep using chat-completions."""
        provider = _provider()
        # Use OpenAI-shape response so the OpenAI parser succeeds.
        openai_payload = {
            "id": "chatcmpl-abc",
            "model": "openai/gpt-4o",
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_resp = _mock_httpx_response(openai_payload)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/openai/gpt-4o"))
        url = mock_post.call_args.args[0]
        assert url.endswith("/v1/chat/completions"), (
            f"Non-Anthropic OR traffic must stay on chat-completions; got {url!r}"
        )

    def test_google_model_keeps_chat_completions(self):
        """Regression guard: google/gemini-* must stay on chat-completions."""
        provider = _provider()
        openai_payload = {
            "id": "chatcmpl-xyz",
            "model": "google/gemini-2.0-flash",
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_resp = _mock_httpx_response(openai_payload)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/google/gemini-2.0-flash"))
        url = mock_post.call_args.args[0]
        assert url.endswith("/v1/chat/completions")


# ---------------------------------------------------------------------------
# Body shape: Anthropic-shape on the passthrough, OpenAI-shape elsewhere
# ---------------------------------------------------------------------------


class TestAnthropicPassthroughBodyShape:
    def test_body_uses_anthropic_messages_field(self):
        """Body must have ``messages`` array of Anthropic-shape entries."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        body = mock_post.call_args.kwargs["json"]
        assert "messages" in body
        # Anthropic shape: {role, content: [{type, text}]} — never the OpenAI
        # ``content: "string"`` shortcut and never ``image_url`` parts.
        msg = body["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"

    def test_body_strips_openrouter_prefix_from_model(self):
        """OR's passthrough sees the upstream-namespaced model id."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        body = mock_post.call_args.kwargs["json"]
        # The ``openrouter/`` routing prefix has been stripped; OR sees the
        # canonical ``anthropic/<model>`` id (matches OR's routing convention).
        assert body["model"] == "anthropic/claude-sonnet-4.5"

    def test_body_preserves_image_blocks_natively(self):
        """The whole point: image blocks must survive into the body.

        On the chat-completions path, ImageBlock would lower to OpenAI's
        ``image_url`` shape and OR's downstream conversion drops it. On
        the messages passthrough, the body carries Anthropic-native
        ``{"type": "image", "source": {...}}`` and OR forwards it intact.
        """
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_multimodal_request("openrouter/anthropic/claude-sonnet-4.5"))
        body = mock_post.call_args.kwargs["json"]
        content = body["messages"][0]["content"]
        types = [block["type"] for block in content]
        assert "image" in types, f"image block dropped from body (got {types!r})"
        image = next(b for b in content if b["type"] == "image")
        assert image["source"]["type"] == "base64"
        assert image["source"]["media_type"] == "image/png"
        assert image["source"]["data"] == "ZmFrZWltYWdl"
        # Sanity: this must NEVER be the OpenAI ``image_url`` shape.
        assert "image_url" not in [b.get("type") for b in content]


# ---------------------------------------------------------------------------
# Headers: Bearer auth + anthropic-version on the passthrough
# ---------------------------------------------------------------------------


class TestAnthropicPassthroughHeaders:
    def test_anthropic_passthrough_sends_anthropic_version(self):
        """OR's /v1/messages emits Anthropic-native shape; we must send
        ``anthropic-version`` so cache_control etc. line up with the
        version the direct Anthropic provider targets."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        headers = mock_post.call_args.kwargs["headers"]
        assert headers.get("anthropic-version"), "Anthropic passthrough must send anthropic-version header"

    def test_anthropic_passthrough_uses_bearer_auth(self):
        """Auth is OR's bearer token, not Anthropic's x-api-key."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        headers = mock_post.call_args.kwargs["headers"]
        assert headers.get("Authorization") == "Bearer or-test"
        # Critical: must NOT use Anthropic's x-api-key — that would leak
        # the OR key into the wrong header and may fail auth.
        assert "x-api-key" not in headers

    def test_chat_completions_does_not_send_anthropic_version(self):
        """Non-Anthropic traffic must not send the anthropic-version header."""
        provider = _provider()
        openai_payload = {
            "id": "chatcmpl-1",
            "model": "openai/gpt-4o",
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_resp = _mock_httpx_response(openai_payload)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/openai/gpt-4o"))
        headers = mock_post.call_args.kwargs["headers"]
        assert "anthropic-version" not in headers


# ---------------------------------------------------------------------------
# Response parsing: Anthropic-shape on the passthrough
# ---------------------------------------------------------------------------


class TestAnthropicPassthroughResponseParsing:
    def test_text_response_parsed(self):
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response("a sunset photo"))
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        assert result.text == "a sunset photo"

    def test_native_anthropic_usage_captured(self):
        """``input_tokens`` / ``output_tokens`` must flow through.

        OR's passthrough may not populate ``usage.cost`` the way
        chat-completions does; the Anthropic-native token counts are the
        canonical signal and what synthpanel's cost tables key off.
        """
        provider = _provider()
        payload = _anthropic_json_response()
        payload["usage"] = {
            "input_tokens": 1234,
            "output_tokens": 56,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50,
        }
        mock_resp = _mock_httpx_response(payload)
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        assert result.usage.input_tokens == 1234
        assert result.usage.output_tokens == 56
        assert result.usage.cache_write_tokens == 100
        assert result.usage.cache_read_tokens == 50

    def test_null_usage_does_not_crash_passthrough(self):
        """OR passthrough may omit usage; tolerate it without zeroing the response."""
        provider = _provider()
        payload = _anthropic_json_response("ok")
        payload["usage"] = None
        mock_resp = _mock_httpx_response(payload)
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.text == "ok"

    def test_response_model_matches_existing_chat_completions_form(self):
        """Both transports must agree on response.model conventions.

        The existing chat-completions path returns the stripped upstream
        id (``anthropic/claude-...``), not the synthpanel-prefixed form
        — see ``_send_openai`` in openrouter.py and ``test_send_success``
        in test_providers.py. Locking that parity in keeps cost tables
        and persistence working uniformly across transports.
        """
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp):
            result = provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        # OR echoes back the stripped form; both paths must produce the
        # same shape (no surprising ``openrouter/`` re-prefixing on one
        # path but not the other).
        assert result.model == "anthropic/claude-sonnet-4.5"


# ---------------------------------------------------------------------------
# Streaming: SSE through the passthrough emits Anthropic-native events
# ---------------------------------------------------------------------------


class TestAnthropicPassthroughStream:
    def test_stream_targets_messages_endpoint(self):
        provider = _provider()
        mock_stream_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            [
                'data: {"type": "message_start", "message": {"id": "x"}}',
                "",
                "data: [DONE]",
                "",
            ]
        )
        mock_stream_cm.__enter__.return_value = mock_resp
        mock_stream_cm.__exit__.return_value = False

        with patch("httpx.stream", return_value=mock_stream_cm) as mock_stream:
            list(provider.stream(_simple_request("openrouter/anthropic/claude-sonnet-4.5")))
        url = mock_stream.call_args.args[1]
        assert url.endswith("/v1/messages")
        # Body must have stream=True on the Anthropic shape.
        body = mock_stream.call_args.kwargs["json"]
        assert body.get("stream") is True

    def test_stream_emits_anthropic_native_events(self):
        """The passthrough emits ``message_start`` etc., not OpenAI deltas."""
        provider = _provider()
        mock_stream_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(
            [
                'data: {"type": "message_start", "message": {"id": "x"}}',
                "",
                'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
                "",
                'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}',
                "",
                'data: {"type": "message_stop"}',
                "",
            ]
        )
        mock_stream_cm.__enter__.return_value = mock_resp
        mock_stream_cm.__exit__.return_value = False

        with patch("httpx.stream", return_value=mock_stream_cm):
            events = list(provider.stream(_simple_request("openrouter/anthropic/claude-sonnet-4.5")))

        types = [e.type for e in events]
        assert StreamEventType.MESSAGE_START in types
        assert StreamEventType.CONTENT_BLOCK_DELTA in types

    def test_non_anthropic_stream_keeps_chat_completions(self):
        provider = _provider()
        mock_stream_cm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = iter(["data: [DONE]", ""])
        mock_stream_cm.__enter__.return_value = mock_resp
        mock_stream_cm.__exit__.return_value = False

        with patch("httpx.stream", return_value=mock_stream_cm) as mock_stream:
            list(provider.stream(_simple_request("openrouter/openai/gpt-4o")))
        url = mock_stream.call_args.args[1]
        assert url.endswith("/v1/chat/completions")


# ---------------------------------------------------------------------------
# Error path: typed-error JSON still surfaces upstream provider/type
# ---------------------------------------------------------------------------


class TestAnthropicPassthroughErrors:
    def test_429_on_passthrough_surfaces_downstream_provider(self):
        """OR's typed-error JSON enrichment must work on the messages route."""
        provider = _provider()
        body = {
            "error": {
                "code": 429,
                "message": "Anthropic rate limit",
                "type": "rate_limit_error",
                "metadata": {"provider_name": "anthropic"},
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = body
        mock_resp.text = json.dumps(body)
        mock_resp.headers = {"retry-after": "5"}
        with patch("httpx.post", return_value=mock_resp), pytest.raises(LLMError) as exc_info:
            provider.send(_simple_request("openrouter/anthropic/claude-sonnet-4.5"))
        msg = str(exc_info.value)
        assert "OpenRouter" in msg
        assert "anthropic" in msg
        assert "rate_limit_error" in msg


# ---------------------------------------------------------------------------
# Edge: model alias + non-prefixed ``anthropic/``-only id (defense in depth)
# ---------------------------------------------------------------------------


class TestRoutingEdgeCases:
    def test_haiku_anthropic_model_also_routes_through_messages(self):
        """All ``openrouter/anthropic/*`` models — not just sonnet — route through messages."""
        provider = _provider()
        mock_resp = _mock_httpx_response(_anthropic_json_response())
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/anthropic/claude-haiku-4.5"))
        assert mock_post.call_args.args[0].endswith("/v1/messages")

    def test_meta_llama_routes_through_chat_completions(self):
        """Sanity: a different OR upstream stays on chat-completions."""
        provider = _provider()
        openai_payload = {
            "id": "chatcmpl-1",
            "model": "meta-llama/llama-3",
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        mock_resp = _mock_httpx_response(openai_payload)
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            provider.send(_simple_request("openrouter/meta-llama/llama-3"))
        assert mock_post.call_args.args[0].endswith("/v1/chat/completions")
