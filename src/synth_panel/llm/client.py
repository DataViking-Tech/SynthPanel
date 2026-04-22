"""LLM Client with provider resolution and retry (SPEC.md §2)."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator

from synth_panel.llm.aliases import get_base_url_override, resolve_alias
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionRequest, CompletionResponse, StreamEvent
from synth_panel.llm.providers.anthropic import ANTHROPIC_CONFIG, AnthropicProvider
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig
from synth_panel.llm.providers.gemini import GEMINI_CONFIG, GeminiProvider
from synth_panel.llm.providers.openai_compat import OPENAI_COMPAT_CONFIG, OpenAICompatibleProvider
from synth_panel.llm.providers.openrouter import OPENROUTER_CONFIG, OpenRouterProvider
from synth_panel.llm.providers.xai import XAI_CONFIG, XAIProvider

logger = logging.getLogger(__name__)

# Provider detection order (SPEC.md §2 — Provider Resolution).
_PROVIDER_REGISTRY: list[tuple[ProviderConfig, type[LLMProvider]]] = [
    (ANTHROPIC_CONFIG, AnthropicProvider),
    (GEMINI_CONFIG, GeminiProvider),
    (XAI_CONFIG, XAIProvider),
    (OPENROUTER_CONFIG, OpenRouterProvider),
    (OPENAI_COMPAT_CONFIG, OpenAICompatibleProvider),
]

# Default retry policy (SPEC.md §2 — Retry policy).
_DEFAULT_INITIAL_BACKOFF = 0.2  # 200ms
_DEFAULT_MAX_BACKOFF = 2.0  # 2s
_DEFAULT_MAX_RETRIES = 2
# Rate-limit backoff has a higher ceiling because providers often hand out
# Retry-After values in the 1-30s range. Without this, we'd clamp useful
# server-supplied waits down to max_backoff and burn the retry budget.
_DEFAULT_MAX_BACKOFF_RATE_LIMIT = 60.0
_DEFAULT_MAX_RETRIES_RATE_LIMIT = 5


class _TokenBucket:
    """Simple thread-safe token bucket for RPS throttling.

    Capacity defaults to the rate, giving a burst of up to one second of
    calls; refill is continuous.
    """

    def __init__(self, rate_per_second: float, capacity: float | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self._rate = rate_per_second
        self._capacity = capacity if capacity is not None else rate_per_second
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until *tokens* are available, then subtract them."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                wait = needed / self._rate
            time.sleep(wait)


class LLMClient:
    """Provider-agnostic LLM client with automatic provider resolution and retry.

    Usage::

        client = LLMClient()
        response = client.send(CompletionRequest(
            model="sonnet",
            max_tokens=1024,
            messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
        ))

    For scaled panel runs, pass ``max_concurrent`` to cap in-flight calls
    and ``rate_limit_rps`` to smooth the request rate per second.
    """

    def __init__(
        self,
        *,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        initial_backoff: float = _DEFAULT_INITIAL_BACKOFF,
        max_backoff: float = _DEFAULT_MAX_BACKOFF,
        max_retries_rate_limit: int = _DEFAULT_MAX_RETRIES_RATE_LIMIT,
        max_backoff_rate_limit: float = _DEFAULT_MAX_BACKOFF_RATE_LIMIT,
        max_concurrent: int | None = None,
        rate_limit_rps: float | None = None,
    ) -> None:
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._max_retries_rate_limit = max_retries_rate_limit
        self._max_backoff_rate_limit = max_backoff_rate_limit
        self._provider_cache: dict[str, LLMProvider] = {}
        self._cache_lock = threading.Lock()
        self._concurrency: threading.Semaphore | None = (
            threading.Semaphore(max_concurrent) if max_concurrent and max_concurrent > 0 else None
        )
        self._bucket: _TokenBucket | None = (
            _TokenBucket(rate_limit_rps) if rate_limit_rps and rate_limit_rps > 0 else None
        )

    def _resolve_provider(self, model: str) -> LLMProvider:
        """Resolve a provider for the given canonical model name."""
        with self._cache_lock:
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
            "No LLM provider credentials found. Run `synthpanel login` or set "
            "ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY, "
            "or OPENAI_API_KEY.",
            LLMErrorCategory.MISSING_CREDENTIALS,
        )

    def _prepare(self, request: CompletionRequest) -> CompletionRequest:
        """Resolve aliases and local model prefixes on the request model."""
        original = request.model
        base_url = get_base_url_override(original)
        canonical = resolve_alias(original)

        # For local models (ollama:*, local:*), cache a provider with the
        # correct base URL so _resolve_provider finds it.
        if base_url is not None:
            with self._cache_lock:
                if canonical not in self._provider_cache:
                    self._provider_cache[canonical] = OpenAICompatibleProvider(
                        base_url=base_url, api_key="no-key-required"
                    )

        if canonical != original:
            return CompletionRequest(
                model=canonical,
                max_tokens=request.max_tokens,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                stream=request.stream,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        return request

    def send(self, request: CompletionRequest) -> CompletionResponse:
        """Send a blocking completion request with automatic retry."""
        request = self._prepare(request)
        provider = self._resolve_provider(request.model)
        logger.debug("send model=%s max_tokens=%d", request.model, request.max_tokens)
        if self._bucket is not None:
            self._bucket.acquire()
        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            t0 = time.monotonic()
            response = self._with_retry(provider.send, request)
            elapsed = time.monotonic() - t0
        finally:
            if self._concurrency is not None:
                self._concurrency.release()
        logger.debug(
            "response model=%s latency=%.2fs tokens_in=%d tokens_out=%d",
            response.model,
            elapsed,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return response

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Send a streaming request. Retry is NOT applied to streams."""
        request = self._prepare(request)
        provider = self._resolve_provider(request.model)
        return provider.stream(request)

    def _sleep_for_retry(self, exc: LLMError, backoff: float) -> float:
        """Compute how long to sleep before the next retry.

        Honors ``retry_after`` from the error (server-supplied wait, e.g.
        via ``Retry-After`` header) when present; otherwise uses the
        current exponential backoff with full jitter.
        """
        if exc.retry_after is not None:
            # Clamp to the rate-limit ceiling so a pathological
            # Retry-After (e.g. 3600s) doesn't stall the whole run.
            return min(exc.retry_after, self._max_backoff_rate_limit)
        return random.uniform(0, backoff)

    def _with_retry(
        self,
        fn: Callable[[CompletionRequest], CompletionResponse],
        request: CompletionRequest,
    ) -> CompletionResponse:
        """Execute *fn* with exponential backoff + jitter on retryable errors.

        Rate-limit errors get a larger retry budget and a higher backoff
        ceiling than other retryable categories, and respect a server
        ``Retry-After`` hint when present.
        """
        last_error: LLMError | None = None
        backoff = self._initial_backoff

        attempt = 0
        while True:
            try:
                return fn(request)
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    break
                if exc.category == LLMErrorCategory.RATE_LIMIT:
                    budget = self._max_retries_rate_limit
                    ceiling = self._max_backoff_rate_limit
                else:
                    budget = self._max_retries
                    ceiling = self._max_backoff
                if attempt >= budget:
                    break
                sleep_for = self._sleep_for_retry(exc, backoff)
                logger.warning(
                    "retryable error (attempt %d/%d, category=%s, sleep=%.2fs): %s",
                    attempt + 1,
                    budget + 1,
                    exc.category.value,
                    sleep_for,
                    exc,
                )
                time.sleep(sleep_for)
                backoff = min(backoff * 2, ceiling)
                attempt += 1

        assert last_error is not None
        if last_error.retryable:
            raise LLMError(
                f"Retries exhausted after {attempt + 1} attempts: {last_error}",
                LLMErrorCategory.RETRIES_EXHAUSTED,
                cause=last_error,
            )
        raise last_error
