"""Tests for the --seed flag (sy-cxp).

Covers:
- ``CompletionRequest.seed`` propagates through ``build_openai_body``.
- Anthropic body never carries ``seed`` even when the request has one.
- ``LLMClient`` warns once per unsupported provider.
- CLI parser exposes ``--seed`` on ``panel run`` and forwards it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from synth_panel.llm.client import LLMClient
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    InputMessage,
    StreamEvent,
    TextBlock,
    TokenUsage,
)
from synth_panel.llm.providers.anthropic import ANTHROPIC_CONFIG, AnthropicProvider
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig
from synth_panel.llm.providers.openai_compat import OPENAI_COMPAT_CONFIG


def _request(model: str = "gpt-4o-mini", seed: int | None = 42) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=10,
        messages=[InputMessage(role="user", content=[TextBlock(text="hi")])],
        seed=seed,
    )


class TestProviderConfigSeed:
    def test_anthropic_does_not_support_seed(self):
        assert ANTHROPIC_CONFIG.supports_seed is False

    def test_openai_supports_seed(self):
        assert OPENAI_COMPAT_CONFIG.supports_seed is True


class TestBuildOpenaiBodySeed:
    def test_seed_in_body(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        body = build_openai_body(_request(seed=42))
        assert body["seed"] == 42

    def test_no_seed_when_none(self):
        from synth_panel.llm.providers._openai_format import build_openai_body

        body = build_openai_body(_request(seed=None))
        assert "seed" not in body


class TestAnthropicBodyOmitsSeed:
    def test_anthropic_body_never_includes_seed(self):
        provider = AnthropicProvider.__new__(AnthropicProvider)
        provider._api_key = "sk-test"  # type: ignore[attr-defined]
        provider._base_url = "https://api.anthropic.com"  # type: ignore[attr-defined]
        body = provider._build_body(_request(model="claude-sonnet-4-6", seed=42))
        assert "seed" not in body


class _FakeProvider(LLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.last_request: CompletionRequest | None = None

    def send(self, request: CompletionRequest) -> CompletionResponse:
        self.last_request = request
        return CompletionResponse(id="x", model=request.model, usage=TokenUsage())

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        self.last_request = request
        return iter([])


class TestSeedWarning:
    def test_warns_once_for_unsupported_provider(self, caplog):
        client = LLMClient()
        fake = _FakeProvider(ANTHROPIC_CONFIG)
        client._provider_cache["claude-sonnet-4-6"] = fake

        req = _request(model="claude-sonnet-4-6", seed=7)
        with caplog.at_level(logging.WARNING, logger="synth_panel.llm.client"):
            client.send(req)
            client.send(req)
            client.send(req)

        warnings = [r for r in caplog.records if "--seed" in r.getMessage()]
        assert len(warnings) == 1
        assert "Anthropic" in warnings[0].getMessage()
        assert "7" in warnings[0].getMessage()

    def test_no_warning_for_supporting_provider(self, caplog):
        client = LLMClient()
        fake = _FakeProvider(OPENAI_COMPAT_CONFIG)
        client._provider_cache["gpt-4o-mini"] = fake

        with caplog.at_level(logging.WARNING, logger="synth_panel.llm.client"):
            client.send(_request(model="gpt-4o-mini", seed=42))

        warnings = [r for r in caplog.records if "--seed" in r.getMessage()]
        assert warnings == []

    def test_no_warning_when_seed_is_none(self, caplog):
        client = LLMClient()
        fake = _FakeProvider(ANTHROPIC_CONFIG)
        client._provider_cache["claude-sonnet-4-6"] = fake

        with caplog.at_level(logging.WARNING, logger="synth_panel.llm.client"):
            client.send(_request(model="claude-sonnet-4-6", seed=None))

        warnings = [r for r in caplog.records if "--seed" in r.getMessage()]
        assert warnings == []

    def test_warning_is_per_provider(self, caplog):
        client = LLMClient()
        anth = _FakeProvider(ANTHROPIC_CONFIG)
        unknown_cfg = ProviderConfig(
            api_key_env="X",
            base_url_env="Y",
            default_base_url="z",
            model_prefixes=(),
            name="LocalLlama",
            supports_seed=False,
        )
        local = _FakeProvider(unknown_cfg)
        client._provider_cache["claude-sonnet-4-6"] = anth
        client._provider_cache["llama3"] = local

        with caplog.at_level(logging.WARNING, logger="synth_panel.llm.client"):
            client.send(_request(model="claude-sonnet-4-6", seed=1))
            client.send(_request(model="llama3", seed=1))

        warnings = [r for r in caplog.records if "--seed" in r.getMessage()]
        assert len(warnings) == 2
        provider_names = sorted("Anthropic" if "Anthropic" in m.getMessage() else "LocalLlama" for m in warnings)
        assert provider_names == ["Anthropic", "LocalLlama"]


class TestCliParser:
    def test_seed_flag_accepted(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--seed", "42", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert args.seed == 42

    def test_seed_defaults_to_none(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert args.seed is None


class TestRequestPropagation:
    """End-to-end: seed on the public request reaches the provider."""

    def test_seed_reaches_provider(self):
        client = LLMClient()
        fake = _FakeProvider(OPENAI_COMPAT_CONFIG)
        client._provider_cache["gpt-4o-mini"] = fake

        client.send(_request(model="gpt-4o-mini", seed=99))
        assert fake.last_request is not None
        assert fake.last_request.seed == 99
