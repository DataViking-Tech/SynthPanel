"""Tests for the unified RetryPolicy (GH#340).

Covers the required scenarios from the bug report:
  - mock 429 → 200 (recovers after one retry)
  - mock 429 + Retry-After (sleep honors the header)
  - mock 401 (authentication; must NOT retry)

Plus:
  - Retry log line includes provider name + attempt + reason at INFO
  - Provider name on RetryPolicy.run is plumbed through to the log
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.llm.client import LLMClient, _provider_name
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    TextBlock,
    TokenUsage,
)
from synth_panel.llm.providers.anthropic import AnthropicProvider
from synth_panel.llm.providers.gemini import GeminiProvider
from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider
from synth_panel.llm.providers.openrouter import OpenRouterProvider
from synth_panel.llm.providers.xai import XAIProvider
from synth_panel.llm.retry import RetryPolicy


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
# Required scenarios from GH#340
# ---------------------------------------------------------------------------


class TestRequiredScenarios:
    """The three scenarios called out explicitly in the bug report."""

    def test_429_then_200_recovers(self):
        """A rate-limit response followed by 200 succeeds via retry."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries_rate_limit=2,
                initial_backoff=0.001,
                max_backoff=0.001,
                max_backoff_rate_limit=0.001,
            )
            provider = MagicMock()
            provider.send.side_effect = [
                LLMError(
                    "rate limited",
                    LLMErrorCategory.RATE_LIMIT,
                    status_code=429,
                ),
                _resp("recovered"),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            result = client.send(_req())
            assert result.text == "recovered"
            assert provider.send.call_count == 2

    def test_429_with_retry_after_sleeps_that_long(self):
        """A 429 carrying ``retry_after`` makes the client sleep that duration."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries_rate_limit=1,
                initial_backoff=10.0,  # large; would dominate without retry_after
                max_backoff=10.0,
                max_backoff_rate_limit=60.0,
            )
            provider = MagicMock()
            provider.send.side_effect = [
                LLMError(
                    "slow down",
                    LLMErrorCategory.RATE_LIMIT,
                    status_code=429,
                    retry_after=0.07,
                ),
                _resp(),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            sleeps: list[float] = []
            with patch(
                "synth_panel.llm.retry.time.sleep",
                side_effect=sleeps.append,
            ):
                client.send(_req())
            assert sleeps == [pytest.approx(0.07, abs=1e-6)]
            assert provider.send.call_count == 2

    def test_401_is_not_retried(self):
        """Authentication errors short-circuit without retry."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(
                max_retries=5,
                max_retries_rate_limit=5,
                initial_backoff=0.001,
                max_backoff=0.001,
            )
            provider = MagicMock()
            provider.send.side_effect = LLMError(
                "invalid api key",
                LLMErrorCategory.AUTHENTICATION,
                status_code=401,
            )
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            with pytest.raises(LLMError) as exc_info:
                client.send(_req())
            assert exc_info.value.category == LLMErrorCategory.AUTHENTICATION
            assert exc_info.value.status_code == 401
            # Crucial: no retries on auth errors.
            assert provider.send.call_count == 1


# ---------------------------------------------------------------------------
# Retry log line
# ---------------------------------------------------------------------------


class TestRetryLogLine:
    def test_log_includes_provider_attempt_reason(self, caplog: pytest.LogCaptureFixture):
        """Retry attempts log INFO with provider, attempt, and reason."""
        policy = RetryPolicy(
            max_retries=1,
            max_retries_rate_limit=1,
            initial_backoff=0.001,
            max_backoff=0.001,
            max_backoff_rate_limit=0.001,
        )
        calls = {"n": 0}

        def fn() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise LLMError("transient", LLMErrorCategory.SERVER_ERROR, status_code=503)
            return "ok"

        with (
            caplog.at_level(logging.INFO, logger="synth_panel.llm.retry"),
            patch("synth_panel.llm.retry.time.sleep"),
        ):
            got = policy.run(fn, provider_name="Anthropic")

        assert got == "ok"
        records = [r for r in caplog.records if r.name == "synth_panel.llm.retry"]
        assert len(records) == 1
        msg = records[0].getMessage()
        assert "provider=Anthropic" in msg
        assert "attempt=1/" in msg
        assert "reason=server_error" in msg
        assert records[0].levelno == logging.INFO


# ---------------------------------------------------------------------------
# Provider name plumbing
# ---------------------------------------------------------------------------


class TestProviderName:
    """Each shipped provider exposes a stable ``name`` for retry logs."""

    @pytest.mark.parametrize(
        "env,model,provider_cls,expected_name",
        [
            ({"ANTHROPIC_API_KEY": "sk"}, "claude-sonnet-4-6-20250414", AnthropicProvider, "Anthropic"),
            ({"GEMINI_API_KEY": "g"}, "gemini-2.5-flash", GeminiProvider, "Gemini"),
            ({"XAI_API_KEY": "x"}, "grok-3", XAIProvider, "xAI"),
            (
                {"OPENROUTER_API_KEY": "or"},
                "openrouter/anthropic/claude-3.5-haiku-20241022",
                OpenRouterProvider,
                "OpenRouter",
            ),
        ],
    )
    def test_provider_name_for_each_provider(
        self,
        env: dict[str, str],
        model: str,
        provider_cls: type,
        expected_name: str,
    ):
        with patch.dict(os.environ, env, clear=False):
            client = LLMClient()
            provider = client._resolve_provider(model)
            assert isinstance(provider, provider_cls)
            assert _provider_name(provider) == expected_name

    def test_openai_compat_name(self):
        env = {
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "XAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "sk-oai",
        }
        with patch.dict(os.environ, env, clear=False):
            client = LLMClient()
            provider = client._resolve_provider("some-unknown-model")
            assert isinstance(provider, OpenAICompatibleProvider)
            assert _provider_name(provider) == "OpenAI"

    def test_provider_name_falls_back_to_class_name(self):
        """If a provider is missing a config or name, fall back gracefully."""

        class _NoConfigProvider:
            pass

        assert _provider_name(_NoConfigProvider()) == "_NoConfigProvider"


# ---------------------------------------------------------------------------
# Custom RetryPolicy injection
# ---------------------------------------------------------------------------


class TestCustomPolicy:
    def test_caller_can_inject_a_retry_policy(self):
        """A caller-supplied RetryPolicy overrides the per-kwarg defaults."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            policy = RetryPolicy(
                max_retries=0,
                max_retries_rate_limit=0,
                initial_backoff=0.001,
                max_backoff=0.001,
                max_backoff_rate_limit=0.001,
            )
            client = LLMClient(retry_policy=policy)

            provider = MagicMock()
            provider.send.side_effect = LLMError(
                "transient",
                LLMErrorCategory.SERVER_ERROR,
                status_code=502,
            )
            client._provider_cache["claude-sonnet-4-6-20250414"] = provider

            with pytest.raises(LLMError) as exc_info:
                client.send(_req())
            # max_retries=0 → no retries attempted.
            assert provider.send.call_count == 1
            assert exc_info.value.category == LLMErrorCategory.RETRIES_EXHAUSTED
