"""Shared serialization helpers for Anthropic Messages API shape.

Used by both the direct Anthropic provider and the OpenRouter provider when
routing ``openrouter/anthropic/*`` traffic through OpenRouter's Anthropic-native
``/v1/messages`` passthrough endpoint (hq-olrk).

Centralizing these helpers means OpenAI-shape multimodal blocks never enter the
Anthropic-routed path, which is what caused hq-m333's silent image-drop on
the chat-completions route.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from althing.llm.models import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    DocumentBlock,
    FileRefSource,
    HTMLBlock,
    ImageBlock,
    InlineSource,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolChoiceKind,
    ToolInvocationBlock,
    URLBlock,
    URLSource,
)

ANTHROPIC_API_VERSION = "2023-06-01"


def build_tool_choice(request: CompletionRequest) -> dict[str, Any] | None:
    if request.tool_choice is None:
        return None
    tc = request.tool_choice
    if tc.kind == ToolChoiceKind.AUTO:
        return {"type": "auto"}
    if tc.kind == ToolChoiceKind.ANY:
        return {"type": "any"}
    return {"type": "tool", "name": tc.name}


def build_source(src: Any, *, media_type: str) -> dict[str, Any]:
    """Serialize an attachment source (tagged-union) to Anthropic API shape."""
    if isinstance(src, InlineSource):
        return {"type": "base64", "media_type": media_type, "data": src.data}
    if isinstance(src, URLSource):
        return {"type": "url", "url": src.url}
    if isinstance(src, FileRefSource):
        return {"type": "file", "file_id": src.file_id}
    raise TypeError(f"Unrecognized attachment source: {type(src).__name__}")


def build_content_blocks(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Serialize content blocks to Anthropic API format.

    Image and document blocks lower to the native ``{"type": "image"|"document",
    "source": {...}}`` shape. ``HTMLBlock`` lowers to a TextBlock at the wire.
    ``URLBlock`` is a pre-fetch stub owned by the URL fetcher (hq-hqlp); it
    must be lowered to a concrete block before reaching this function, so we
    raise rather than silently dropping it.
    """
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append({"type": "text", "text": b.text})
        elif isinstance(b, ToolInvocationBlock):
            out.append(
                {
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                }
            )
        elif isinstance(b, ImageBlock):
            entry: dict[str, Any] = {
                "type": "image",
                "source": build_source(b.source, media_type=b.media_type),
            }
            if b.cache_control is not None:
                entry["cache_control"] = {"type": b.cache_control}
            out.append(entry)
        elif isinstance(b, DocumentBlock):
            entry = {
                "type": "document",
                "source": build_source(b.source, media_type=b.media_type),
            }
            if b.cache_control is not None:
                entry["cache_control"] = {"type": b.cache_control}
            out.append(entry)
        elif isinstance(b, HTMLBlock):
            entry = {"type": "text", "text": b.text}
            if b.cache_control is not None:
                entry["cache_control"] = {"type": b.cache_control}
            out.append(entry)
        elif isinstance(b, URLBlock):
            raise ValueError(
                f"URLBlock(url={b.url!r}) reached the wire serializer; "
                "URL blocks must be lowered by the fetcher (hq-hqlp) before send"
            )
        elif hasattr(b, "tool_use_id"):  # ToolResultBlock
            content = [{"type": "text", "text": c.text} for c in b.content]
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": b.tool_use_id,
                    "content": content,
                    "is_error": b.is_error,
                }
            )
    return out


def build_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    """Convert InputMessages to Anthropic API format.

    When ``request.cache_enabled`` is True (default), automatically marks
    the trailing text block of the last user message with
    ``cache_control: ephemeral`` so single-turn callers get prefix
    caching for free. With ``cache_enabled`` False (hq-0pbp: P=1 panels
    or sub-minimum prefixes), no auto marker is added — explicit
    per-block ``cache_control`` set by the caller is preserved either way.
    """
    last_user_idx = -1
    for i, msg in enumerate(request.messages):
        if msg.role == "user":
            last_user_idx = i

    result = []
    for i, msg in enumerate(request.messages):
        content = build_content_blocks(msg.content)
        if request.cache_enabled and i == last_user_idx and content:
            already_marked = any("cache_control" in b for b in content)
            if not already_marked:
                for j in range(len(content) - 1, -1, -1):
                    if content[j].get("type") == "text":
                        content[j] = {**content[j], "cache_control": {"type": "ephemeral"}}
                        break
        result.append({"role": msg.role, "content": content})
    return result


def build_tools(request: CompletionRequest) -> list[dict[str, Any]] | None:
    if not request.tools:
        return None
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.input_schema,
        }
        for t in request.tools
    ]


def build_anthropic_body(request: CompletionRequest) -> dict[str, Any]:
    """Build a request body in Anthropic Messages API shape.

    Used by the direct Anthropic provider and by OpenRouter's
    ``openrouter/anthropic/*`` route through ``/v1/messages`` (hq-olrk).
    """
    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": request.max_tokens,
        "messages": build_messages(request),
    }
    if request.system:
        sys_block: dict[str, Any] = {"type": "text", "text": request.system}
        if request.cache_enabled:
            sys_block["cache_control"] = {"type": "ephemeral"}
        body["system"] = [sys_block]
    tools = build_tools(request)
    if tools is not None:
        body["tools"] = tools
    tc = build_tool_choice(request)
    if tc is not None:
        body["tool_choice"] = tc
    if request.stream:
        body["stream"] = True
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    return body


def parse_content_block(raw: dict[str, Any]) -> ContentBlock:
    btype = raw.get("type")
    if btype == "text":
        return TextBlock(text=raw["text"])
    if btype == "tool_use":
        return ToolInvocationBlock(
            id=raw["id"],
            name=raw["name"],
            input=raw.get("input", {}),
        )
    if btype == "thinking":
        return ThinkingBlock(
            thinking=raw.get("thinking", ""),
            signature=raw.get("signature"),
        )
    return TextBlock(text=json.dumps(raw))


def parse_usage(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        cache_write_tokens=raw.get("cache_creation_input_tokens", 0),
        cache_read_tokens=raw.get("cache_read_input_tokens", 0),
    )


def parse_stop_reason(raw: str | None) -> StopReason | None:
    if raw is None:
        return None
    try:
        return StopReason(raw)
    except ValueError:
        return StopReason.END_TURN


def parse_anthropic_response(data: dict[str, Any], fallback_model: str) -> CompletionResponse:
    """Parse an Anthropic Messages API JSON response into CompletionResponse."""
    content = [parse_content_block(b) for b in data.get("content", [])]
    return CompletionResponse(
        id=data.get("id", ""),
        model=data.get("model", fallback_model),
        content=content,
        stop_reason=parse_stop_reason(data.get("stop_reason")),
        usage=parse_usage(data.get("usage") or {}),
    )


def parse_sse_stream(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Parse Anthropic-shape SSE stream into StreamEvents."""
    data_buf: list[str] = []
    for line in lines:
        if line.startswith("data: "):
            data_buf.append(line[6:])
        elif line == "" and data_buf:
            raw_data = "\n".join(data_buf)
            data_buf.clear()
            if raw_data.strip() == "[DONE]":
                return
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            event = sse_payload_to_event(payload)
            if event is not None:
                yield event
        elif line.startswith(":"):
            # Comment / keepalive — discard
            continue


def sse_payload_to_event(payload: dict[str, Any]) -> StreamEvent | None:
    etype = payload.get("type", "")
    try:
        event_type = StreamEventType(etype)
    except ValueError:
        return None
    if event_type == StreamEventType.PING:
        return None
    return StreamEvent(
        type=event_type,
        index=payload.get("index"),
        data=payload,
    )
