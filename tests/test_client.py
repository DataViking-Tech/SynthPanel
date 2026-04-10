"""Tests for LLMClient — provider resolution and retry logic."""

from __future__ import annotations

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

    def test_openrouter_prefix(self):
        """openrouter/* models resolve to OpenRouterProvider."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}):
            client = LLMClient()
            provider = client._resolve_provider("openrouter/meta-llama/llama-3")
            from synth_panel.llm.providers.openrouter import OpenRouterProvider

            assert isinstance(provider, OpenRouterProvider)

    def test_fallback_to_available_credentials(self):
        """Unknown models fall back to first provider with credentials."""
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
            from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

            assert isinstance(provider, OpenAICompatibleProvider)

    def test_no_credentials_raises(self):
        """Missing all credentials raises MISSING_CREDENTIALS."""
        env = {
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "",
            "XAI_API_KEY": "",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "",
        }
        with patch.dict(os.environ, env, clear=False):
            client = LLMClient()
            with pytest.raises(LLMError) as exc_info:
                client._resolve_provider("some-model")
            assert exc_info.value.category == LLMErrorCategory.MISSING_CREDENTIALS


class TestAliasResolution:
    def test_alias_is_resolved_in_send(self):
        """Short aliases like 'sonnet' are resolved before provider lookup."""
        from synth_panel.llm.aliases import resolve_alias

        canonical = resolve_alias("sonnet")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            client = LLMClient()

            mock_provider = MagicMock()
            mock_provider.send.return_value = _simple_response()

            with patch.object(client, "_resolve_provider", return_value=mock_provider) as mock_resolve:
                client.send(_simple_request(model="sonnet"))
                # _resolve_provider must receive the canonical name, not the alias
                mock_resolve.assert_called_once_with(canonical)
                # Provider receives request with the resolved model name
                call_args = mock_provider.send.call_args[0][0]
                assert call_args.model == canonical


class TestLocalModelResolution:
    def test_ollama_prefix_creates_local_provider(self):
        """ollama:llama3 creates an OpenAI-compat provider with Ollama base URL."""
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        client = LLMClient()
        mock_provider = MagicMock()
        mock_provider.send.return_value = _simple_response()

        # _prepare should cache a local provider
        prepared = client._prepare(_simple_request(model="ollama:llama3"))
        assert prepared.model == "llama3"
        assert "llama3" in client._provider_cache
        provider = client._provider_cache["llama3"]
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider._base_url == "http://localhost:11434"

    def test_local_prefix_creates_lmstudio_provider(self):
        """local:phi3 creates an OpenAI-compat provider with LM Studio base URL."""
        from synth_panel.llm.providers.openai_compat import OpenAICompatibleProvider

        client = LLMClient()
        prepared = client._prepare(_simple_request(model="local:phi3"))
        assert prepared.model == "phi3"
        provider = client._provider_cache["phi3"]
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider._base_url == "http://localhost:1234"

    def test_ollama_send_uses_correct_url(self):
        """End-to-end: ollama:llama3 sends to localhost:11434."""
        client = LLMClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "x",
            "model": "llama3",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = client.send(_simple_request(model="ollama:llama3"))
        assert result.text == "hi"
        called_url = mock_post.call_args[0][0]
        assert called_url == "http://localhost:11434/v1/chat/completions"


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
            mock_provider.send.side_effect = LLMError("invalid key", LLMErrorCategory.AUTHENTICATION)
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
            mock_provider.send.side_effect = LLMError("server error", LLMErrorCategory.SERVER_ERROR, status_code=500)
            client._provider_cache["claude-sonnet-4-6-20250414"] = mock_provider

            with pytest.raises(LLMError) as exc_info:
                client.send(_simple_request())
            assert exc_info.value.category == LLMErrorCategory.RETRIES_EXHAUSTED
            assert mock_provider.send.call_count == 2  # 1 initial + 1 retry
