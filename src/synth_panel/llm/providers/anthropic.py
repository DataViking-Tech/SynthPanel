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
    StreamEvent,
)
from synth_panel.llm.providers._anthropic_format import (
    ANTHROPIC_API_VERSION as _ANTHROPIC_API_VERSION,
)
from synth_panel.llm.providers._anthropic_format import (
    build_anthropic_body as _build_anthropic_body,
)
from synth_panel.llm.providers._anthropic_format import (
    build_content_blocks as _build_content_blocks,
)
from synth_panel.llm.providers._anthropic_format import (
    build_messages as _build_messages,
)
from synth_panel.llm.providers._anthropic_format import (
    build_source as _build_source,
)
from synth_panel.llm.providers._anthropic_format import (
    build_tool_choice as _build_tool_choice,
)
from synth_panel.llm.providers._anthropic_format import (
    build_tools as _build_tools,
)
from synth_panel.llm.providers._anthropic_format import (
    parse_anthropic_response as _parse_anthropic_response,
)
from synth_panel.llm.providers._anthropic_format import (
    parse_content_block as _parse_content_block,
)
from synth_panel.llm.providers._anthropic_format import (
    parse_sse_stream as _parse_sse_stream,
)
from synth_panel.llm.providers._anthropic_format import (
    parse_stop_reason as _parse_stop_reason,
)
from synth_panel.llm.providers._anthropic_format import (
    parse_usage as _parse_usage,
)
from synth_panel.llm.providers._anthropic_format import (
    sse_payload_to_event as _sse_payload_to_event,
)
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig

ANTHROPIC_CONFIG = ProviderConfig(
    api_key_env="ANTHROPIC_API_KEY",
    base_url_env="ANTHROPIC_BASE_URL",
    default_base_url="https://api.anthropic.com",
    model_prefixes=("claude-",),
    name="Anthropic",
)


__all__ = [
    "ANTHROPIC_CONFIG",
    "_ANTHROPIC_API_VERSION",
    "AnthropicProvider",
    "_build_anthropic_body",
    "_build_content_blocks",
    "_build_messages",
    "_build_source",
    "_build_tool_choice",
    "_build_tools",
    "_parse_anthropic_response",
    "_parse_content_block",
    "_parse_sse_stream",
    "_parse_stop_reason",
    "_parse_usage",
    "_sse_payload_to_event",
]


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
        return _build_anthropic_body(request)

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

        return _parse_anthropic_response(data, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        # Anthropic's API has no ``seed`` parameter (sy-cxp); the
        # LLMClient warns once per provider when seed is set on a
        # non-supporting provider, so we just drop it here.
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
            cache_enabled=request.cache_enabled,
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
