"""Shared retry/backoff policy for LLM provider calls (GH#340).

Every provider goes through ``LLMClient`` which delegates to
``RetryPolicy.run`` — so all providers share the same retry budget,
backoff curve, and Retry-After handling. Retries log at INFO level
with ``provider``, ``attempt``, and ``reason`` so operators can see
where backoff is happening without having to crank up DEBUG.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from synth_panel.llm.errors import LLMError, LLMErrorCategory

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Default retry policy (SPEC.md §2 — Retry policy).
DEFAULT_INITIAL_BACKOFF = 0.2  # 200ms
DEFAULT_MAX_BACKOFF = 2.0  # 2s
DEFAULT_MAX_RETRIES = 2
# Rate-limit backoff has a higher ceiling because providers often hand out
# Retry-After values in the 1-30s range. Without this, we'd clamp useful
# server-supplied waits down to max_backoff and burn the retry budget.
DEFAULT_MAX_BACKOFF_RATE_LIMIT = 60.0
DEFAULT_MAX_RETRIES_RATE_LIMIT = 5


@dataclass
class RetryPolicy:
    """Shared retry/backoff policy for LLM provider calls.

    Encapsulates:
      - Exponential backoff with full jitter for transient/server errors
      - Server-supplied ``Retry-After`` precedence on rate-limit errors
      - Distinct budgets and ceilings for rate-limit vs. other retryables
      - INFO-level retry logging with provider, attempt, and reason
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF
    max_backoff: float = DEFAULT_MAX_BACKOFF
    max_retries_rate_limit: int = DEFAULT_MAX_RETRIES_RATE_LIMIT
    max_backoff_rate_limit: float = DEFAULT_MAX_BACKOFF_RATE_LIMIT

    def sleep_for(self, exc: LLMError, backoff: float) -> float:
        """Compute how long to sleep before the next retry.

        Honors ``retry_after`` from the error (server-supplied wait,
        e.g. via ``Retry-After`` header) when present; otherwise uses
        the current exponential backoff with full jitter.
        """
        if exc.retry_after is not None:
            # Clamp pathological Retry-After (e.g. 3600s) so it doesn't
            # stall the whole run.
            return min(exc.retry_after, self.max_backoff_rate_limit)
        return random.uniform(0, backoff)

    def run(self, fn: Callable[[], T], *, provider_name: str) -> T:
        """Execute ``fn`` with retry/backoff.

        Raises the underlying ``LLMError`` if it's non-retryable, or a
        ``RETRIES_EXHAUSTED`` ``LLMError`` if the retry budget is spent.
        ``provider_name`` is used in retry logs so operators can see
        which provider is backing off.
        """
        last_error: LLMError | None = None
        backoff = self.initial_backoff
        attempt = 0
        while True:
            try:
                return fn()
            except LLMError as exc:
                last_error = exc
                if not exc.retryable:
                    break
                if exc.category == LLMErrorCategory.RATE_LIMIT:
                    budget = self.max_retries_rate_limit
                    ceiling = self.max_backoff_rate_limit
                else:
                    budget = self.max_retries
                    ceiling = self.max_backoff
                if attempt >= budget:
                    break
                sleep_for = self.sleep_for(exc, backoff)
                logger.info(
                    "llm retry: provider=%s attempt=%d/%d reason=%s sleep=%.2fs",
                    provider_name,
                    attempt + 1,
                    budget + 1,
                    exc.category.value,
                    sleep_for,
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
