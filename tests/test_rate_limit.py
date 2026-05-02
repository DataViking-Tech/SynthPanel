"""Rate-limit-aware LLM client tests (sp-i2ub slice a).

Covers:
  - Retry-After header parsing (seconds, HTTP-date, garbage)
  - LLMError.retry_after propagation
  - llm_error_from_response helper extracts retry-after on 429
  - LLMClient honors retry_after on rate-limit retries
  - LLMClient rate-limit retry budget exceeds the default budget
  - LLMClient.send Semaphore caps in-flight concurrency
  - LLMClient.send token bucket caps requests-per-second
  - Panel completes with a mock provider that returns 429 on every Nth call
"""

from __future__ import annotations

import os
import threading
import time
from email.utils import formatdate
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.llm.client import LLMClient, _TokenBucket
from synth_panel.llm.errors import (
    LLMError,
    LLMErrorCategory,
    llm_error_from_response,
    parse_retry_after,
    retry_after_from_headers,
)
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    TextBlock,
    TokenUsage,
)


def _req(model: str = "claude-sonnet-4-6-20250414") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=64,
        messages=[InputMessage(role="user", content=[TextBlock(text="hi")])],
    )


def _resp(text: str = "ok") -> CompletionResponse:
    return CompletionResponse(
        id="r",
        model="test",
        content=[TextBlock(text=text)],
        usage=TokenUsage(input_tokens=1, output_tokens=1),
    )


# ---------------------------------------------------------------------------
# Retry-After parsing
# ---------------------------------------------------------------------------


class TestParseRetryAfter:
    def test_integer_seconds(self):
        assert parse_retry_after("5") == 5.0

    def test_float_seconds(self):
        assert parse_retry_after("2.5") == 2.5

    def test_negative_clamped(self):
        assert parse_retry_after("-3") == 0.0

    def test_none(self):
        assert parse_retry_after(None) is None

    def test_empty(self):
        assert parse_retry_after("") is None
        assert parse_retry_after("   ") is None

    def test_garbage(self):
        assert parse_retry_after("tomorrow") is None

    def test_http_date_future(self):
        # A date 30s in the future should yield ~30s.
        future = time.time() + 30
        header = formatdate(future, usegmt=True)
        got = parse_retry_after(header)
        assert got is not None
        assert 25 <= got <= 35

    def test_http_date_past_is_zero(self):
        past = time.time() - 60
        header = formatdate(past, usegmt=True)
        got = parse_retry_after(header)
        assert got == 0.0


class TestRetryAfterFromHeaders:
    def test_retry_after(self):
        headers = {"retry-after": "7"}
        assert retry_after_from_headers(headers) == 7.0

    def test_anthropic_ratelimit_requests_reset(self):
        headers = {"anthropic-ratelimit-requests-reset": "12"}
        assert retry_after_from_headers(headers) == 12.0

    def test_x_ratelimit_reset(self):
        headers = {"x-ratelimit-reset": "3"}
        assert retry_after_from_headers(headers) == 3.0

    def test_none_headers(self):
        assert retry_after_from_headers(None) is None

    def test_missing(self):
        assert retry_after_from_headers({}) is None


# ---------------------------------------------------------------------------
# LLMError + llm_error_from_response
# ---------------------------------------------------------------------------


class TestLLMError:
    def test_default_retry_after_none(self):
        err = LLMError("boom", LLMErrorCategory.RATE_LIMIT)
        assert err.retry_after is None

    def test_retry_after_stored(self):
        err = LLMError("boom", LLMErrorCategory.RATE_LIMIT, retry_after=4.0)
        assert err.retry_after == 4.0


class TestLLMErrorFromResponse:
    def test_429_with_retry_after(self):
        resp = MagicMock()
        resp.status_code = 429
        resp.text = "too many"
        resp.headers = {"retry-after": "15"}
        err = llm_error_from_response(resp, "Anthropic")
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert err.status_code == 429
        assert err.retry_after == 15.0
        assert "Anthropic" in str(err)

    def test_429_without_retry_after(self):
        resp = MagicMock()
        resp.status_code = 429
        resp.text = "slow down"
        resp.headers = {}
        err = llm_error_from_response(resp, "OpenAI")
        assert err.category == LLMErrorCategory.RATE_LIMIT
        assert err.retry_after is None

    def test_500_ignores_retry_after_header(self):
        # Only rate-limit responses populate retry_after; a 500 with a
        # random Retry-After would mislead the client.
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "oops"
        resp.headers = {"retry-after": "30"}
        err = llm_error_from_response(resp, "xAI")
        assert err.category == LLMErrorCategory.SERVER_ERROR
        assert err.retry_after is None


# ---------------------------------------------------------------------------
# LLMClient rate-limit behavior
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    def test_honors_retry_after_over_backoff(self):
        """When a 429 carries retry_after, sleep that long, not the backoff."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries=0,
                max_retries_rate_limit=1,
                initial_backoff=10.0,  # big, to prove we ignore it
                max_backoff=10.0,
                max_backoff_rate_limit=60.0,
            )
            provider = MagicMock()
            provider.send.side_effect = [
                LLMError(
                    "rate limit",
                    LLMErrorCategory.RATE_LIMIT,
                    status_code=429,
                    retry_after=0.05,
                ),
                _resp("recovered"),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider
            sleep_calls: list[float] = []

            def fake_sleep(s: float) -> None:
                sleep_calls.append(s)

            with patch("synth_panel.llm.retry.time.sleep", side_effect=fake_sleep):
                result = client.send(_req())
            assert result.text == "recovered"
            assert len(sleep_calls) == 1
            # Server-supplied retry_after should dominate the exponential backoff.
            assert sleep_calls[0] == pytest.approx(0.05, abs=1e-6)

    def test_retry_after_clamped_to_ceiling(self):
        """An absurd Retry-After is clamped to max_backoff_rate_limit."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries=0,
                max_retries_rate_limit=1,
                max_backoff_rate_limit=2.0,
            )
            provider = MagicMock()
            provider.send.side_effect = [
                LLMError(
                    "rate limit",
                    LLMErrorCategory.RATE_LIMIT,
                    retry_after=3600.0,
                ),
                _resp(),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider
            sleep_calls: list[float] = []
            with patch(
                "synth_panel.llm.retry.time.sleep",
                side_effect=lambda s: sleep_calls.append(s),
            ):
                client.send(_req())
            assert sleep_calls == [2.0]

    def test_rate_limit_budget_larger_than_default(self):
        """Rate-limit retries use their own budget, not max_retries."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries=0,  # transport/server have 0 retries
                max_retries_rate_limit=3,  # but rate-limit has 3
                initial_backoff=0.001,
                max_backoff=0.001,
                max_backoff_rate_limit=0.001,
            )
            provider = MagicMock()
            # 3 rate-limits, then success — must succeed via RL budget.
            provider.send.side_effect = [
                LLMError("rl", LLMErrorCategory.RATE_LIMIT, retry_after=0.001),
                LLMError("rl", LLMErrorCategory.RATE_LIMIT, retry_after=0.001),
                LLMError("rl", LLMErrorCategory.RATE_LIMIT, retry_after=0.001),
                _resp("ok"),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider
            result = client.send(_req())
            assert result.text == "ok"
            assert provider.send.call_count == 4

    def test_rate_limit_exhaustion_raises(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries_rate_limit=2,
                max_backoff_rate_limit=0.001,
            )
            provider = MagicMock()
            provider.send.side_effect = LLMError("rl", LLMErrorCategory.RATE_LIMIT, retry_after=0.001)
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider
            with pytest.raises(LLMError) as exc_info:
                client.send(_req())
            assert exc_info.value.category == LLMErrorCategory.RETRIES_EXHAUSTED
            assert provider.send.call_count == 3  # 1 initial + 2 retries


# ---------------------------------------------------------------------------
# Concurrency cap
# ---------------------------------------------------------------------------


class TestMaxConcurrent:
    def test_semaphore_caps_in_flight(self):
        """With max_concurrent=2, only 2 sends may be in flight at once."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(max_concurrent=2)
            in_flight = 0
            peak = 0
            lock = threading.Lock()

            def fake_send(_req: CompletionRequest) -> CompletionResponse:
                nonlocal in_flight, peak
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                time.sleep(0.02)
                with lock:
                    in_flight -= 1
                return _resp()

            provider = MagicMock()
            provider.send.side_effect = fake_send
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            threads = [threading.Thread(target=lambda: client.send(_req())) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert peak <= 2
            assert provider.send.call_count == 8

    def test_no_semaphore_when_unset(self):
        """Without max_concurrent, calls proceed unthrottled."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient()
            assert client._concurrency is None


# ---------------------------------------------------------------------------
# RPS throttle
# ---------------------------------------------------------------------------


class TestRateLimitRps:
    def test_token_bucket_rejects_nonpositive_rate(self):
        with pytest.raises(ValueError):
            _TokenBucket(0)

    def test_token_bucket_throttles(self):
        """A 20-rps bucket should take ~0.25s for 5 tokens past the initial burst."""
        bucket = _TokenBucket(20.0, capacity=1)
        t0 = time.monotonic()
        for _ in range(5):
            bucket.acquire()
        elapsed = time.monotonic() - t0
        # 1 initial + 4 refilled at 20/s = 4/20 = 0.2s minimum.
        assert elapsed >= 0.15

    def test_client_uses_rps(self):
        """LLMClient wires rate_limit_rps to a token bucket."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(rate_limit_rps=10.0)
            assert client._bucket is not None


# ---------------------------------------------------------------------------
# Panel-style integration: mock 429 every Nth call
# ---------------------------------------------------------------------------


class TestPanelSurvives429s:
    def test_every_third_call_429(self):
        """A panel-like sequence where every 3rd call 429s still completes."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries=0,
                max_retries_rate_limit=3,
                max_backoff_rate_limit=0.001,
            )

            state = {"calls": 0}

            def flaky(_req: CompletionRequest) -> CompletionResponse:
                state["calls"] += 1
                # 1st, 2nd pass; 3rd 429; 4th pass; 5th pass; 6th 429; ...
                if state["calls"] % 3 == 0:
                    raise LLMError(
                        "slow down",
                        LLMErrorCategory.RATE_LIMIT,
                        status_code=429,
                        retry_after=0.001,
                    )
                return _resp("ok")

            provider = MagicMock()
            provider.send.side_effect = flaky
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            # Simulate 10 panelists doing one call each.
            for _ in range(10):
                result = client.send(_req())
                assert result.text == "ok"
            # 10 logical sends + some retries for the 429s.
            assert provider.send.call_count >= 10
