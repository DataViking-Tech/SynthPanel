"""Abstract base for LLM providers (SPEC.md §2)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from synth_panel.credentials import (
    get_credential,
    has_credential,
    missing_api_key_message,
)
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionRequest, CompletionResponse, StreamEvent


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a provider: env var names and defaults."""

    api_key_env: str
    base_url_env: str
    default_base_url: str
    model_prefixes: tuple[str, ...]
    name: str = ""
    supports_seed: bool = False

    def get_api_key(self) -> str:
        """Return the API key from env or the on-disk credential store."""
        key = get_credential(self.api_key_env)
        if not key:
            raise LLMError(
                missing_api_key_message(self.api_key_env),
                LLMErrorCategory.MISSING_CREDENTIALS,
            )
        return key

    def get_base_url(self) -> str:
        """Read the base URL from env, falling back to the default."""
        return os.environ.get(self.base_url_env, "").strip() or self.default_base_url

    def has_credentials(self) -> bool:
        """Return True if the API key is set in env or on disk."""
        return has_credential(self.api_key_env)


class LLMProvider(ABC):
    """Interface every provider must implement."""

    config: ProviderConfig

    @abstractmethod
    def send(self, request: CompletionRequest) -> CompletionResponse:
        """Send a blocking completion request."""

    @abstractmethod
    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Send a streaming completion request, yielding events."""
