"""Google Gemini provider implementation (SPEC.md §2).

Google's Gemini API exposes an OpenAI-compatible chat completions endpoint,
so this provider translates our internal request/response models to that format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from synth_panel.credentials import get_credential, missing_api_key_message
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

GEMINI_CONFIG = ProviderConfig(
    api_key_env="GEMINI_API_KEY",
    base_url_env="GEMINI_BASE_URL",
    default_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model_prefixes=("gemini-",),
    name="Gemini",
    supports_seed=True,
)


class GeminiProvider(LLMProvider):
    """Google Gemini provider (OpenAI-compatible chat completions)."""

    config = GEMINI_CONFIG

    def __init__(self) -> None:
        # Support both GEMINI_API_KEY and GOOGLE_API_KEY, resolved via env
        # or the on-disk credential store.
        key = get_credential("GEMINI_API_KEY") or get_credential("GOOGLE_API_KEY")
        if not key:
            raise LLMError(
                missing_api_key_message("GEMINI_API_KEY", alt_env_vars=("GOOGLE_API_KEY",)),
                LLMErrorCategory.MISSING_CREDENTIALS,
            )
        self._api_key = key
        self._base_url = self.config.get_base_url()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def send(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/chat/completions"
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
            raise llm_error_from_response(resp, "Gemini")

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                "Failed to parse Gemini response",
                LLMErrorCategory.DESERIALIZATION,
                cause=exc,
            ) from exc

        return parse_openai_response(data, request.model)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        url = f"{self._base_url}/chat/completions"
        body = build_openai_body(request, stream=True)
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
                    raise llm_error_from_response(resp, "Gemini")
                yield from parse_openai_sse_stream(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error during stream: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc
