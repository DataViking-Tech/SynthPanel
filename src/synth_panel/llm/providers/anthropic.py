"""Anthropic provider implementation (SPEC.md §2)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from synth_panel.llm.errors import LLMError, LLMErrorCategory, llm_error_from_response
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    ThinkingBlock,
    TokenUsage,
    ToolChoiceKind,
    ToolInvocationBlock,
)
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig

_ANTHROPIC_API_VERSION = "2023-06-01"

ANTHROPIC_CONFIG = ProviderConfig(
    api_key_env="ANTHROPIC_API_KEY",
    base_url_env="ANTHROPIC_BASE_URL",
    default_base_url="https://api.anthropic.com",
    model_prefixes=("claude-",),
    name="Anthropic",
)


def _build_tool_choice(request: CompletionRequest) -> dict[str, Any] | None:
    if request.tool_choice is None:
        return None
    tc = request.tool_choice
    if tc.kind == ToolChoiceKind.AUTO:
        return {"type": "auto"}
    if tc.kind == ToolChoiceKind.ANY:
        return {"type": "any"}
    return {"type": "tool", "name": tc.name}


def _build_content_blocks(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    """Serialize content blocks to Anthropic API format."""
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


def _build_messages(request: CompletionRequest) -> list[dict[str, Any]]:
    """Convert InputMessages to Anthropic API format."""
    last_user_idx = -1
    for i, msg in enumerate(request.messages):
        if msg.role == "user":
            last_user_idx = i

    result = []
    for i, msg in enumerate(request.messages):
        content = _build_content_blocks(msg.content)
        if i == last_user_idx and content:
            for j in range(len(content) - 1, -1, -1):
                if content[j].get("type") == "text":
                    content[j] = {**content[j], "cache_control": {"type": "ephemeral"}}
                    break
        result.append({"role": msg.role, "content": content})
    return result


def _build_tools(request: CompletionRequest) -> list[dict[str, Any]] | None:
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


def _parse_content_block(raw: dict[str, Any]) -> ContentBlock:
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


def _parse_usage(raw: dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        cache_write_tokens=raw.get("cache_creation_input_tokens", 0),
        cache_read_tokens=raw.get("cache_read_input_tokens", 0),
    )


def _parse_stop_reason(raw: str | None) -> StopReason | None:
    if raw is None:
        return None
    try:
        return StopReason(raw)
    except ValueError:
        return StopReason.END_TURN


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider."""

    config = ANTHROPIC_CONFIG

    def __init__(self) -> None:
        self._api_key = self.config.get_api_key()
        self._base_url = self.config.get_base_url()

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

    def _build_body(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": _build_messages(request),
        }
        if request.system:
            body["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
        tools = _build_tools(request)
        if tools is not None:
            body["tools"] = tools
        tc = _build_tool_choice(request)
        if tc is not None:
            body["tool_choice"] = tc
        if request.stream:
            body["stream"] = True
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        return body

    def send(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/v1/messages"
        body = self._build_body(request)
        try:
            resp = httpx.post(
                url,
                headers=self._headers(),
                json=body,
                timeout=120.0,
            )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc

        if resp.status_code != 200:
            raise llm_error_from_response(resp, "Anthropic")

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                "Failed to parse Anthropic response",
                LLMErrorCategory.DESERIALIZATION,
                cause=exc,
            ) from exc

        content = [_parse_content_block(b) for b in data.get("content", [])]
        return CompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", request.model),
            content=content,
            stop_reason=_parse_stop_reason(data.get("stop_reason")),
            usage=_parse_usage(data.get("usage", {})),
        )

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        request_copy = CompletionRequest(
            model=request.model,
            max_tokens=request.max_tokens,
            messages=request.messages,
            system=request.system,
            tools=request.tools,
            tool_choice=request.tool_choice,
            stream=True,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        url = f"{self._base_url}/v1/messages"
        body = self._build_body(request_copy)

        try:
            with httpx.stream(
                "POST",
                url,
                headers=self._headers(),
                json=body,
                timeout=120.0,
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    raise llm_error_from_response(resp, "Anthropic")
                yield from _parse_sse_stream(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error during stream: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc


def _parse_sse_stream(lines: Iterator[str]) -> Iterator[StreamEvent]:
    """Parse Anthropic SSE stream into StreamEvents."""
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
            event = _sse_payload_to_event(payload)
            if event is not None:
                yield event
        elif line.startswith(":"):
            # Comment / keepalive — discard
            continue


def _sse_payload_to_event(payload: dict[str, Any]) -> StreamEvent | None:
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
