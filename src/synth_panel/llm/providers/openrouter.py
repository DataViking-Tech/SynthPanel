"""OpenRouter provider (OpenAI-compatible).

Routes requests to any model via OpenRouter's unified API.
Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from synth_panel.llm.errors import LLMError, LLMErrorCategory, classify_http_status
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)
from synth_panel.llm.providers._openai_format import (
    build_openai_body,
    parse_openai_response,
    parse_openai_sse_stream,
)
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig

OPENROUTER_CONFIG = ProviderConfig(
    api_key_env="OPENROUTER_API_KEY",
    base_url_env="OPENROUTER_BASE_URL",
    default_base_url="https://openrouter.ai/api",
    model_prefixes=("openrouter/",),
)


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider (OpenAI-compatible chat completions)."""

    config = OPENROUTER_CONFIG

    def __init__(self) -> None:
        self._api_key = self.config.get_api_key()
        self._base_url = self.config.get_base_url()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _strip_prefix(model: str) -> str:
        """Strip the ``openrouter/`` routing prefix so the API sees the
        upstream model ID (e.g. ``anthropic/claude-3.5-haiku-20241022``)."""
        for prefix in OPENROUTER_CONFIG.model_prefixes:
            if model.startswith(prefix):
                return model[len(prefix) :]
        return model

    def send(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/v1/chat/completions"
        # Strip routing prefix before building the request body
        if request.model != self._strip_prefix(request.model):
            from synth_panel.llm.models import CompletionRequest as CR

            request = CR(
                model=self._strip_prefix(request.model),
                max_tokens=request.max_tokens,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream=request.stream,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        body = build_openai_body(request)
        # Ask OpenRouter to always return the detailed usage block (including
        # token counts and native cost). Without this flag, some upstream
        # providers omit ``usage`` entirely, causing synthpanel to compute
        # $0 for the whole panel. See sp-2xy.
        body["usage"] = {"include": True}
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
                f"OpenRouter API error {resp.status_code}: {resp.text[:500]}",
                cat,
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                "Failed to parse OpenRouter response",
                LLMErrorCategory.DESERIALIZATION,
                cause=exc,
            ) from exc

        return parse_openai_response(data, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        url = f"{self._base_url}/v1/chat/completions"
        # Strip routing prefix before building the request body
        if request.model != self._strip_prefix(request.model):
            from synth_panel.llm.models import CompletionRequest as CR

            request = CR(
                model=self._strip_prefix(request.model),
                max_tokens=request.max_tokens,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream=True,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        body = build_openai_body(request, stream=True)
        # Mirror send(): ensure OpenRouter emits usage details in the final
        # stream chunk so synthpanel can track cost accurately.
        body["usage"] = {"include": True}
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
                    cat = classify_http_status(resp.status_code)
                    raise LLMError(
                        f"OpenRouter API error {resp.status_code}: {resp.text[:500]}",
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
