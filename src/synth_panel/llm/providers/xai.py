"""xAI (Grok) provider implementation (SPEC.md §2).

xAI exposes an OpenAI-compatible chat completions endpoint, so this provider
translates our internal request/response models to the OpenAI chat format.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

import httpx

from synth_panel.llm.errors import LLMError, LLMErrorCategory, classify_http_status
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
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig
from synth_panel.llm.providers._openai_format import (
    build_openai_body,
    parse_openai_response,
    parse_openai_sse_stream,
)

XAI_CONFIG = ProviderConfig(
    api_key_env="XAI_API_KEY",
    base_url_env="XAI_BASE_URL",
    default_base_url="https://api.x.ai",
    model_prefixes=("grok-",),
)


class XAIProvider(LLMProvider):
    """xAI / Grok provider (OpenAI-compatible chat completions)."""

    config = XAI_CONFIG

    def __init__(self) -> None:
        self._api_key = self.config.get_api_key()
        self._base_url = self.config.get_base_url()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def send(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/v1/chat/completions"
        body = build_openai_body(request)
        try:
            resp = httpx.post(url, headers=self._headers(), json=body, timeout=120.0)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc

        if resp.status_code != 200:
            cat = classify_http_status(resp.status_code)
            raise LLMError(
                f"xAI API error {resp.status_code}: {resp.text[:500]}",
                cat,
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                "Failed to parse xAI response",
                LLMErrorCategory.DESERIALIZATION,
                cause=exc,
            ) from exc

        return parse_openai_response(data, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        url = f"{self._base_url}/v1/chat/completions"
        body = build_openai_body(request, stream=True)
        try:
            with httpx.stream(
                "POST", url, headers=self._headers(), json=body, timeout=120.0,
            ) as resp:
                if resp.status_code != 200:
                    resp.read()
                    cat = classify_http_status(resp.status_code)
                    raise LLMError(
                        f"xAI API error {resp.status_code}: {resp.text[:500]}",
                        cat,
                        status_code=resp.status_code,
                    )
                yield from parse_openai_sse_stream(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error during stream: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc
