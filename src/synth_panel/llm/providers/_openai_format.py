"""Shared serialization helpers for OpenAI-compatible chat completions APIs.

Used by both the xAI provider and the generic OpenAI-compatible provider.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    TokenUsage,
    ToolChoiceKind,
    ToolInvocationBlock,
)

# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _content_to_openai(blocks: list[ContentBlock]) -> str | list[dict[str, Any]]:
    """Serialize content blocks to OpenAI message content format."""
    # Simple case: single text block → plain string
    if len(blocks) == 1 and isinstance(blocks[0], TextBlock):
        return blocks[0].text

    parts: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolInvocationBlock):
            # Tool calls are handled separately in OpenAI format
            pass
    return parts if parts else ""


def build_openai_body(
    request: CompletionRequest,
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Build an OpenAI chat-completions request body."""
    messages: list[dict[str, Any]] = []

    if request.system:
        messages.append({"role": "system", "content": request.system})

    for msg in request.messages:
        entry: dict[str, Any] = {"role": msg.role}

        # Check for tool calls in assistant messages
        tool_calls = [b for b in msg.content if isinstance(b, ToolInvocationBlock)]
        text_blocks = [b for b in msg.content if isinstance(b, TextBlock)]

        if msg.role == "assistant" and tool_calls:
            if text_blocks:
                entry["content"] = text_blocks[0].text
            else:
                entry["content"] = None
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in tool_calls
            ]
        else:
            # Check for tool result blocks
            tool_results = [b for b in msg.content if hasattr(b, "tool_use_id")]
            if tool_results:
                # OpenAI uses role=tool for tool results
                for tr in tool_results:
                    content_text = " ".join(c.text for c in tr.content)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.tool_use_id,
                            "content": content_text,
                        }
                    )
                continue
            else:
                entry["content"] = _content_to_openai(msg.content)

        messages.append(entry)

    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": messages,
    }

    if request.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in request.tools
        ]

    if request.tool_choice is not None:
        tc = request.tool_choice
        if tc.kind == ToolChoiceKind.AUTO:
            body["tool_choice"] = "auto"
        elif tc.kind == ToolChoiceKind.ANY:
            body["tool_choice"] = "required"
        elif tc.kind == ToolChoiceKind.SPECIFIC and tc.name:
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.name},
            }

    if stream:
        body["stream"] = True

    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.seed is not None:
        body["seed"] = request.seed

    return body


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_usage(usage_raw: dict[str, Any]) -> TokenUsage:
    """Map an OpenAI-compatible ``usage`` block to ``TokenUsage``.

    Beyond the base ``prompt_tokens`` / ``completion_tokens``, this
    captures OpenRouter's authoritative cost fields and OpenAI-style
    sub-counts that the upstream API only reports when asked for detail:

    - ``usage.cost`` (float, USD): native cost as billed by OpenRouter,
      reflecting BYOK discounts and actual upstream pricing. Preferred
      over the local pricing table when present. ``cost_details`` may
      expose ``upstream_inference_cost`` (preferred), otherwise the
      sum of ``upstream_inference_prompt_cost`` and
      ``upstream_inference_completions_cost``.
    - ``prompt_tokens_details.cached_tokens``: the subset of
      ``prompt_tokens`` that was served from cache.
    - ``completion_tokens_details.reasoning_tokens``: the subset of
      ``completion_tokens`` spent on reasoning (e.g. GPT-5).

    ``reasoning_tokens`` and ``cached_tokens`` remain sub-counts —
    callers should not subtract them from the base counts.
    """

    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    provider_cost = _coerce_float(usage_raw.get("cost"))
    cost_details = usage_raw.get("cost_details")
    if provider_cost is None and isinstance(cost_details, dict):
        upstream = _coerce_float(cost_details.get("upstream_inference_cost"))
        if upstream is not None:
            provider_cost = upstream
        else:
            prompt_cost = _coerce_float(cost_details.get("upstream_inference_prompt_cost"))
            completion_cost = _coerce_float(cost_details.get("upstream_inference_completions_cost"))
            if prompt_cost is not None or completion_cost is not None:
                provider_cost = (prompt_cost or 0.0) + (completion_cost or 0.0)

    prompt_details = usage_raw.get("prompt_tokens_details")
    cached_tokens = 0
    if isinstance(prompt_details, dict):
        cached_tokens = int(prompt_details.get("cached_tokens") or 0)

    completion_details = usage_raw.get("completion_tokens_details")
    reasoning_tokens = 0
    if isinstance(completion_details, dict):
        reasoning_tokens = int(completion_details.get("reasoning_tokens") or 0)

    # NOTE: ``cached_tokens`` is NOT propagated into ``cache_read_tokens``
    # because OpenAI-style ``prompt_tokens`` already includes the cached
    # subset. Adding it again would double-count under local pricing (which
    # bills ``input_tokens`` + ``cache_read_tokens`` separately). The sub-
    # count is preserved in the dedicated ``cached_tokens`` field for
    # reporting. When provider_reported_cost is present (OpenRouter), the
    # local estimate is overridden anyway.
    return TokenUsage(
        input_tokens=usage_raw.get("prompt_tokens") or 0,
        output_tokens=usage_raw.get("completion_tokens") or 0,
        provider_reported_cost=provider_cost,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
    )


def _parse_openai_stop(reason: str | None) -> StopReason | None:
    if reason is None:
        return None
    mapping = {
        "stop": StopReason.END_TURN,
        "tool_calls": StopReason.TOOL_USE,
        "length": StopReason.MAX_TOKENS,
    }
    return mapping.get(reason, StopReason.END_TURN)


def parse_openai_response(
    data: dict[str, Any],
    request_model: str,
) -> CompletionResponse:
    """Parse an OpenAI chat-completions response into our internal format."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    content: list[ContentBlock] = []
    if message.get("content"):
        content.append(TextBlock(text=message["content"]))

    for tc in message.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        content.append(
            ToolInvocationBlock(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=args,
            )
        )

    # Some OpenAI-compatible providers (notably OpenRouter) occasionally
    # return ``"usage": null`` or omit the block entirely. Treat any
    # non-dict value as an absent usage block so the caller still gets
    # a well-formed TokenUsage (all zeros) instead of an AttributeError.
    usage_raw = data.get("usage") or {}
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    usage = _extract_usage(usage_raw)

    return CompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", request_model),
        content=content,
        stop_reason=_parse_openai_stop(choice.get("finish_reason")),
        usage=usage,
    )


# ---------------------------------------------------------------------------
# SSE stream parsing
# ---------------------------------------------------------------------------


def parse_openai_sse_stream(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Parse an OpenAI-compatible SSE stream into StreamEvents."""
    data_buf: list[str] = []
    for line in lines:
        if line.startswith("data: "):
            payload_str = line[6:]
            if payload_str.strip() == "[DONE]":
                yield StreamEvent(type=StreamEventType.MESSAGE_STOP, data={})
                return
            data_buf.append(payload_str)
        elif line == "" and data_buf:
            raw = "\n".join(data_buf)
            data_buf.clear()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            event = _openai_chunk_to_event(payload)
            if event is not None:
                yield event
        elif line.startswith(":"):
            continue


def _openai_chunk_to_event(payload: dict[str, Any]) -> StreamEvent | None:
    """Convert a single OpenAI streaming chunk to a StreamEvent."""
    choices = payload.get("choices", [])
    if not choices:
        return StreamEvent(type=StreamEventType.MESSAGE_START, data=payload)

    choice = choices[0]
    delta = choice.get("delta", {})
    finish = choice.get("finish_reason")

    if finish:
        return StreamEvent(
            type=StreamEventType.MESSAGE_DELTA,
            data={"stop_reason": finish, **payload},
        )

    if delta.get("content"):
        return StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=choice.get("index", 0),
            data={"text": delta["content"]},
        )

    if "tool_calls" in delta:
        return StreamEvent(
            type=StreamEventType.CONTENT_BLOCK_DELTA,
            index=choice.get("index", 0),
            data={"tool_calls": delta["tool_calls"]},
        )

    return None
