"""Tests for LLMClient — provider resolution and retry logic."""

import os
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.llm.client import LLMClient
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    TextBlock,
    TokenUsage,
)


def _simple_request(model: str = "claude-sonnet-4-6-20250414") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=100,
        messages=[InputMessage(role="user", content=[TextBlock(text="Hi")])],
    )


def _simple_response() -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="test",
        content=[TextBlock(text="Hello!")],
        usage=TokenUsage(input_tokens=5, output_tokens=10),
    )


class TestProviderResolution:
    def test_anthropic_prefix(self):
        """claude-* models resolve to AnthropicProvider."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient()
            provider = client._resolve_provider("claude-sonnet-4-6-20250414")
            from synth_panel.llm.providers.anthropic import AnthropicProvider
            assert isinstance(provider, AnthropicProvider)

    def test_xai_prefix(self):
        """grok-* models resolve to XAIProvider."""
        with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}):
            client = LLMClient()
            provider = client._resolve_provider("grok-3")
            from synth_panel.llm.providers.xai import XAIProvider
            assert isinstance(provider, XAIProvider)

    def test_fallback_to_available_credentials(self):
        """Unknown models fall back to first provider with credentials."""
        env = {"ANTHROPIC_API_KEY": "", "XAI_API_KEY": "", "OPENAI_API_KEY": "sk-oai"}
        with patch.dict(os.environ, env, clear=False):
            client = LLMClient()
            provider = client._resolve_provider("some-unknown-model")
            from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_no_credentials_raises(self):
        """Missing all credentials raises MISSING_CREDENTIALS."""
        env = {"ANTHROPIC_API_KEY": "", "XAI_API_KEY": "", "OPENAI_API_KEY": ""}
        with patch.dict(os.environ, env, clear=False):
            client = LLMClient()
            with pytest.raises(LLMError) as exc_info:
                client._resolve_provider("some-model")
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS


class TestAliasResolution:
    def test_alias_is_resolved_in_send(self):
        """Short aliases like 'sonnet' are resolved before provider lookup."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient()

            mock_provider = MagicMock()
            mock_provider.send.return_value = _simple_response()
            client._provider_cache["claude-sonnet-4-6-20250414"] = mock_provider

            client.send(_simple_request(model="sonnet"))
            call_args = mock_provider.send.call_args[0][0]
            assert call_args.model == "claude-sonnet-4-6-20250414"


class TestRetry:
    def test_retries_on_transport_error(self):
        """Transport errors trigger retry up to max_retries."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(max_retries=2, initial_backoff=0.001, max_backoff=0.01)

            mock_provider = MagicMock()
            # Fail twice, succeed on third
            mock_provider.send.side_effect = [
                LLMError("timeout", LLMErrorCategory.TRANSPORT),
                LLMError("timeout", LLMErrorCategory.TRANSPORT),
                _simple_response(),
            ]
            client._provider_cache["claude-sonnet-4-6-20250414"] = mock_provider

            result = client.send(_simple_request())
            assert result.text == "Hello!"
            assert mock_provider.send.call_count == 3

    def test_no_retry_on_auth_error(self):
        """Authentication errors are not retried."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(max_retries=2)

            mock_provider = MagicMock()
            mock_provider.send.side_effect = LLMError(
                "invalid key", LLMErrorCategory.AUTHENTICATION
            )
            client._provider_cache["claude-sonnet-4-6-20250414"] = mock_provider

            with pytest.raises(LLMError) as exc_info:
                client.send(_simple_request())
            assert exc_info.value.category == LLMErrorCategory.AUTHENTICATION
            assert mock_provider.send.call_count == 1

    def test_retries_exhausted_raises(self):
        """After max retries, raises RETRIES_EXHAUSTED."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient(max_retries=1, initial_backoff=0.001, max_backoff=0.01)

            mock_provider = MagicMock()
            mock_provider.send.side_effect = LLMError(
                "server error", LLMErrorCategory.SERVER_ERROR, status_code=500
            )
            client._provider_cache["claude-sonnet-4-6-20250414"] = mock_provider

            with pytest.raises(LLMError) as exc_info:
                client.send(_simple_request())
            assert exc_info.value.category == LLMErrorCategory.RETRIES_EXHAUSTED
            assert mock_provider.send.call_count == 2  # 1 initial + 1 retry
