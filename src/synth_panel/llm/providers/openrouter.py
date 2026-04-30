"""OpenRouter provider (OpenAI-compatible).

Routes requests to any model via OpenRouter's unified API.
Requires OPENROUTER_API_KEY.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from synth_panel.llm.errors import LLMError, LLMErrorCategory, llm_error_from_response
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


def _openrouter_error_from_response(resp: httpx.Response) -> LLMError:
    """Build an LLMError from an OpenRouter non-2xx response, surfacing
    upstream provider details when present.

    OpenRouter forwards typed errors from the downstream provider in the
    body:

        {"error": {"code": 429, "message": "Rate limit exceeded ...",
                   "type": "rate_limit_error",
                   "metadata": {"provider_name": "anthropic", ...}}}

    When ``error.type`` and/or ``error.metadata.provider_name`` are
    present, include them in the message so callers can tell which
    downstream provider rejected the request and pick a recovery
    strategy (wait vs. switch model). Falls back to the generic
    ``llm_error_from_response`` message when the body is missing or
    unparseable.
    """
    base = llm_error_from_response(resp, "OpenRouter")

    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        return base
    if not isinstance(body, dict):
        return base

    err = body.get("error")
    if not isinstance(err, dict):
        return base

    upstream: str | None = None
    metadata = err.get("metadata")
    if isinstance(metadata, dict):
        provider_name = metadata.get("provider_name")
        if isinstance(provider_name, str) and provider_name:
            upstream = provider_name

    error_type = err.get("type") if isinstance(err.get("type"), str) else None
    upstream_msg = err.get("message") if isinstance(err.get("message"), str) else None

    if upstream is None and error_type is None and upstream_msg is None:
        return base

    label = f"OpenRouter (downstream: {upstream})" if upstream else "OpenRouter"
    head = f"{label} API error {resp.status_code}"
    if error_type:
        head += f" [{error_type}]"
    message = f"{head}: {upstream_msg}" if upstream_msg else head

    return LLMError(
        message,
        base.category,
        status_code=base.status_code,
        retry_after=base.retry_after,
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
            raise _openrouter_error_from_response(resp)

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
                    raise _openrouter_error_from_response(resp)
                yield from parse_openai_sse_stream(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error during stream: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc
