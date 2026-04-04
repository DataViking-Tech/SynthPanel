"""Shared serialization helpers for OpenAI-compatible chat completions APIs.

Used by both the xAI provider and the generic OpenAI-compatible provider.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

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
            tool_results = [
                b for b in msg.content if hasattr(b, "tool_use_id")
            ]
            if tool_results:
                # OpenAI uses role=tool for tool results
                for tr in tool_results:
                    content_text = " ".join(c.text for c in tr.content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": content_text,
                    })
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

    return body


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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
        content.append(ToolInvocationBlock(
            id=tc.get("id", ""),
            name=fn.get("name", ""),
            input=args,
        ))

    usage_raw = data.get("usage", {})
    usage = TokenUsage(
        input_tokens=usage_raw.get("prompt_tokens", 0),
        output_tokens=usage_raw.get("completion_tokens", 0),
    )

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

    if "content" in delta and delta["content"]:
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
