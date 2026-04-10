"""LLM Client with provider resolution and retry (SPEC.md §2)."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator

from synth_panel.llm.aliases import resolve_alias
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionRequest, CompletionResponse, StreamEvent
from synth_panel.llm.providers.anthropic import ANTHROPIC_CONFIG, AnthropicProvider
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig
from synth_panel.llm.providers.gemini import GEMINI_CONFIG, GeminiProvider
from synth_panel.llm.providers.openai_compat import OPENAI_COMPAT_CONFIG, OpenAICompatibleProvider
from synth_panel.llm.providers.xai import XAI_CONFIG, XAIProvider

# Provider detection order (SPEC.md §2 — Provider Resolution).
_PROVIDER_REGISTRY: list[tuple[ProviderConfig, type[LLMProvider]]] = [
    (ANTHROPIC_CONFIG, AnthropicProvider),
    (GEMINI_CONFIG, GeminiProvider),
    (XAI_CONFIG, XAIProvider),
    (OPENAI_COMPAT_CONFIG, OpenAICompatibleProvider),
]

# Default retry policy (SPEC.md §2 — Retry policy).
_DEFAULT_INITIAL_BACKOFF = 0.2  # 200ms
_DEFAULT_MAX_BACKOFF = 2.0  # 2s
_DEFAULT_MAX_RETRIES = 2


class LLMClient:
    """Provider-agnostic LLM client with automatic provider resolution and retry.

    Usage::

        client = LLMClient()
        response = client.send(CompletionRequest(
            model="sonnet",
            max_tokens=1024,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
        ))
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_backoff: float = _DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = _DEFAULT_MAX_BACKOFF,
    ) -> None:
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._provider_cache: dict[str, LLMProvider] = {}

    def _resolve_provider(self, model: str) -> LLMProvider:
        """Resolve a provider for the given canonical model name."""
        if model in self._provider_cache:
            return self._provider_cache[model]

        # 1. Match by prefix
        for config, cls in _PROVIDER_REGISTRY:
            for prefix in config.model_prefixes:
                if model.startswith(prefix):
                    provider = cls()
                    self._provider_cache[model] = provider
                    return provider

        # 2. Fallback: first provider with credentials available
        for config, cls in _PROVIDER_REGISTRY:
            if config.has_credentials():
                provider = cls()
                self._provider_cache[model] = provider
                return provider

        raise LLMError(
            "No LLM provider credentials found. Set ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, or OPENAI_API_KEY.",
            LLMErrorCategory.MISSING_CREDENTIALS,
        )

    def _prepare(self, request: CompletionRequest) -> CompletionRequest:
        """Resolve aliases on the request model."""
        canonical = resolve_alias(request.model)
        if canonical != request.model:
            return CompletionRequest(
                model=canonical,
                max_tokens=request.max_tokens,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream=request.stream,
            )
        return request

    def send(self, request: CompletionRequest) -> CompletionResponse:
        """Send a blocking completion request with automatic retry."""
        request = self._prepare(request)
        provider = self._resolve_provider(request.model)
        return self._with_retry(provider.send, request)

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Send a streaming request. Retry is NOT applied to streams."""
        request = self._prepare(request)
        provider = self._resolve_provider(request.model)
        return provider.stream(request)

    def _with_retry(
        self,
        fn: Callable[[CompletionRequest], CompletionResponse],
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Execute *fn* with exponential backoff + jitter on retryable errors."""
        last_error: LLMError | None = None
        backoff = self._initial_backoff

        for attempt in range(1 + self._max_retries):
            try:
                return fn(request)
            except LLMError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self._max_retries:
                    break
                # Exponential backoff with full jitter
                jitter = random.uniform(0, backoff)
                time.sleep(jitter)
                backoff = min(backoff * 2, self._max_backoff)

        assert last_error is not None
        if last_error.retryable:
            raise LLMError(
                f"Retries exhausted after {self._max_retries + 1} attempts: {last_error}",
                LLMErrorCategory.RETRIES_EXHAUSTED,
                cause=last_error,
            )
        raise last_error
