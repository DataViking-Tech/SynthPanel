# Writing a synthpanel LLM Adapter

This guide walks through adding support for a new LLM provider to synthpanel by writing an **adapter** — a thin translator between synthpanel's internal request/response model and a provider's HTTP API. It is the highest-leverage way to contribute: one adapter makes every synthpanel feature (panels, MCP tools, ensemble blending, budgeting, branching) instantly available against a new backend.

The canonical behavioral contract is [SPEC.md §2 — LLM Client Abstraction](../SPEC.md). This guide is the hands-on companion: what files to touch, what `LLMProvider` looks like in code, and a complete worked example (a hypothetical Mistral adapter).

> This is documentation, not a code change request. Do not copy the Mistral example into the repo — implement a real adapter only against a real, contributor-tested provider.

## 1. Overview

### What is an adapter?

An adapter is a Python module in `src/synth_panel/llm/providers/` that:

- Subclasses `LLMProvider` (from `base.py`)
- Implements two methods: `send()` (blocking) and `stream()` (iterator of SSE events)
- Declares a `ProviderConfig` describing its env vars, base URL, and model prefixes
- Gets registered in the provider resolver (`client.py`) so model strings like `"mistral-large-latest"` route to it automatically

The rest of synthpanel — orchestrator, cost tracker, MCP server, CLI — never talks to the provider directly. It only holds an `LLMClient`, which picks the right adapter by inspecting the model identifier and the environment.

### When would you write one?

Write a new adapter when:

- The provider exposes its own API shape (Anthropic-style, Gemini-style) that isn't OpenAI-compatible. Adapters like `anthropic.py` and `gemini.py` translate native formats into synthpanel's internal model.
- The provider is OpenAI-compatible but warrants a first-class entry — a dedicated env var, prefix-based routing, a specific default base URL. OpenRouter and xAI both follow this pattern: they reuse `_openai_format.py` helpers but live as their own module for discoverability and config isolation.
- The provider needs adapter-specific behavior (auth refresh, response translation quirks, custom streaming framing) that shouldn't pollute the generic `openai_compat` provider.

If the provider is plain-vanilla OpenAI-compatible and doesn't need a dedicated env var, users can already reach it via `OPENAI_BASE_URL` + `OPENAI_API_KEY`. No new adapter is required.

## 2. The LLMProvider base class

Source: [`src/synth_panel/llm/providers/base.py`](../src/synth_panel/llm/providers/base.py)

Every adapter implements this interface:

```python
from collections.abc import Iterator
from synth_panel.llm.models import (
    CompletionRequest,
    CompletionResponse,
    StreamEvent,
)
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig


class MyProvider(LLMProvider):
    config: ProviderConfig  # class attribute

    def send(self, request: CompletionRequest) -> CompletionResponse:
        """Blocking completion. Build an HTTP request, parse the response,
        return a CompletionResponse. Raise LLMError on failure."""

    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        """Streaming completion. Yield StreamEvents in the order defined by
        SPEC.md §2 — message_start, content_block_start/delta/stop, message_delta,
        message_stop. Raise LLMError on failure."""
```

### `ProviderConfig`

```python
@dataclass(frozen=True)
class ProviderConfig:
    api_key_env: str         # e.g. "MISTRAL_API_KEY"
    base_url_env: str        # e.g. "MISTRAL_BASE_URL"
    default_base_url: str    # e.g. "https://api.mistral.ai"
    model_prefixes: tuple[str, ...]  # e.g. ("mistral-", "open-mistral-", "open-mixtral-")
```

Three helpers come for free:

- `config.get_api_key()` — reads `api_key_env`, raises `LLMError(MISSING_CREDENTIALS)` if unset.
- `config.get_base_url()` — reads `base_url_env`, falls back to `default_base_url`.
- `config.has_credentials()` — returns `True` if the key is set (used by the fallback path in provider resolution).

### Data models

`CompletionRequest`, `CompletionResponse`, `StreamEvent`, `TokenUsage`, and content block types live in [`src/synth_panel/llm/models.py`](../src/synth_panel/llm/models.py). They are provider-agnostic — your job is to translate to and from the provider's wire format.

The four token buckets (`input_tokens`, `output_tokens`, `cache_write_tokens`, `cache_read_tokens`) feed the cost tracker. If the provider doesn't expose cache counters, leave them at `0`.

## 3. Step-by-step: Writing a Mistral adapter

A hypothetical worked example. Mistral exposes an OpenAI-compatible chat completions endpoint, so we can lean on `_openai_format.py` the same way `xai.py` does.

### Step 1 — Create the module

`src/synth_panel/llm/providers/mistral.py`:

```python
"""Mistral provider (OpenAI-compatible chat completions).

Routes requests to Mistral's native API. Requires MISTRAL_API_KEY.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from synth_panel.llm.errors import LLMError, LLMErrorCategory, classify_http_status
from synth_panel.llm.models import CompletionRequest, CompletionResponse, StreamEvent
from synth_panel.llm.providers._openai_format import (
    build_openai_body,
    parse_openai_response,
    parse_openai_sse_stream,
)
from synth_panel.llm.providers.base import LLMProvider, ProviderConfig

MISTRAL_CONFIG = ProviderConfig(
    api_key_env="MISTRAL_API_KEY",
    base_url_env="MISTRAL_BASE_URL",
    default_base_url="https://api.mistral.ai",
    model_prefixes=("mistral-", "open-mistral-", "open-mixtral-"),
)
```

### Step 2 — Implement `send()`

Pattern: build the OpenAI-shaped body, `POST` it, classify errors, parse the JSON response.

```python
class MistralProvider(LLMProvider):
    config = MISTRAL_CONFIG

    def __init__(self) -> None:
        self._api_key = self.config.get_api_key()
        self._base_url = self.config.get_base_url()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def send(self, request: CompletionRequest) -> CompletionResponse:
        url = f"{self._base_url}/v1/chat/completions"
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
            cat = classify_http_status(resp.status_code)
            raise LLMError(
                f"Mistral API error {resp.status_code}: {resp.text[:500]}",
                cat,
                status_code=resp.status_code,
            )

        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(
                "Failed to parse Mistral response",
                LLMErrorCategory.DESERIALIZATION,
                cause=exc,
            ) from exc

        return parse_openai_response(data, request.model)
```

Key points:

- **Timeout**: 120 seconds matches other providers. Do not remove it.
- **Error classification**: `classify_http_status()` maps 429/5xx → retryable, 4xx → non-retryable. Keep response body truncated (`[:500]`) to avoid leaking secrets into logs.
- **Deserialization failures** are non-retryable — a garbled response will not un-garble on retry.
- **No retry logic here.** `LLMClient._with_retry()` handles that around `send()` in the client layer. Adapters just raise typed errors.

### Step 3 — Implement `stream()`

Pattern: open a streaming HTTP request, yield parsed SSE events.

```python
    def stream(self, request: CompletionRequest) -> Iterator[StreamEvent]:
        url = f"{self._base_url}/v1/chat/completions"
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
                    cat = classify_http_status(resp.status_code)
                    raise LLMError(
                        f"Mistral API error {resp.status_code}: {resp.text[:500]}",
                        cat,
                        status_code=resp.status_code,
                    )
                yield from parse_openai_sse_stream(resp.iter_lines())
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Transport error during stream: {exc}",
                LLMErrorCategory.TRANSPORT,
                cause=exc,
            ) from exc
```

`parse_openai_sse_stream()` handles chunked delivery, ping/keepalive frames, end-of-stream sentinels, and multi-line data fields — all the framing quirks SPEC.md §2 requires.

If the provider uses Anthropic-style events instead of OpenAI SSE, you'll need a bespoke parser. See `anthropic.py` for the reference implementation.

### Step 4 — Register in the provider resolver

`src/synth_panel/llm/client.py`:

```python
from synth_panel.llm.providers.mistral import MISTRAL_CONFIG, MistralProvider

_PROVIDER_REGISTRY: list[tuple[ProviderConfig, type[LLMProvider]]] = [
    (ANTHROPIC_CONFIG, AnthropicProvider),
    (GEMINI_CONFIG, GeminiProvider),
    (XAI_CONFIG, XAIProvider),
    (OPENROUTER_CONFIG, OpenRouterProvider),
    (MISTRAL_CONFIG, MistralProvider),     # ← add here
    (OPENAI_COMPAT_CONFIG, OpenAICompatibleProvider),
]
```

**Order matters.** `OPENAI_COMPAT_CONFIG` is a catch-all fallback and should stay last. Put prefix-matched providers before it.

Also update the "no credentials" error message at the bottom of `_resolve_provider()` so `MISTRAL_API_KEY` appears in the hint.

### Step 5 — Env var convention

synthpanel follows a two-variable convention per provider:

| Variable | Purpose |
|----------|---------|
| `<PROVIDER>_API_KEY` | Required. The auth token. |
| `<PROVIDER>_BASE_URL` | Optional. Overrides `default_base_url`. Useful for proxies, VPC endpoints, and mock servers in tests. |

For Mistral: `MISTRAL_API_KEY` and `MISTRAL_BASE_URL`. Keep the naming consistent — users, CI configs, and the MCP server env block all rely on the `<NAME>_API_KEY` pattern.

If your provider supports an alias (e.g. `SYNTHPANEL_MODEL_ALIASES='{"mistral-small": "mistral-small-latest"}'`), document it but do not hardcode new aliases in `aliases.py` unless the short name is already well-known across the ecosystem (as `haiku`, `sonnet`, `gemini` are).

### Step 6 — Unit tests

`tests/test_mistral_provider.py`:

```python
"""Unit tests for the Mistral provider.

All network calls are mocked via httpx. The network guard in conftest.py
will fail the test if a real connection is attempted.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import CompletionRequest, InputMessage, TextBlock


def _request(model: str = "mistral-large-latest") -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=100,
        messages=[InputMessage(role="user", content=[TextBlock(text="Hello")])],
    )


def _ok_response(text: str = "Hi there") -> dict:
    return {
        "id": "chatcmpl-mistral-1",
        "model": "mistral-large-latest",
        "choices": [
            {
                "message": {"content": text, "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def _mock_http(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.text = json.dumps(data)
    return resp


class TestMistralProviderSend:
    def test_happy_path(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        from synth_panel.llm.providers.mistral import MistralProvider

        provider = MistralProvider()
        with patch("httpx.post", return_value=_mock_http(_ok_response())):
            resp = provider.send(_request())

        assert resp.text == "Hi there"
        assert resp.usage.input_tokens == 10
        assert resp.usage.output_tokens == 5

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        from synth_panel.llm.providers.mistral import MistralProvider

        with pytest.raises(LLMError) as exc_info:
            MistralProvider()
        assert exc_info.value.category is LLMErrorCategory.MISSING_CREDENTIALS

    def test_rate_limit_is_retryable(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        from synth_panel.llm.providers.mistral import MistralProvider

        provider = MistralProvider()
        with patch("httpx.post", return_value=_mock_http({}, status_code=429)):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_request())
        assert exc_info.value.retryable is True

    def test_bad_request_is_not_retryable(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        from synth_panel.llm.providers.mistral import MistralProvider

        provider = MistralProvider()
        with patch("httpx.post", return_value=_mock_http({}, status_code=400)):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_request())
        assert exc_info.value.retryable is False

    def test_transport_error_raises_transport_category(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        from synth_panel.llm.providers.mistral import MistralProvider

        provider = MistralProvider()
        with patch("httpx.post", side_effect=httpx.ConnectError("boom")):
            with pytest.raises(LLMError) as exc_info:
                provider.send(_request())
        assert exc_info.value.category is LLMErrorCategory.TRANSPORT

    def test_base_url_override(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        monkeypatch.setenv("MISTRAL_BASE_URL", "https://proxy.example.com")
        from synth_panel.llm.providers.mistral import MistralProvider

        provider = MistralProvider()
        assert provider._base_url == "https://proxy.example.com"
```

Model the file on [`tests/test_providers.py`](../tests/test_providers.py). The helpers there (`_simple_request`, `_openai_json_response`, `_mock_httpx_response`) can be reused for consistency.

### Step 7 — Acceptance test

Add a test to `tests/test_acceptance.py` gated by `@pytest.mark.acceptance`. These run only when a real API key is set, and are skipped by default CI.

```python
@acceptance
class TestMistralLive:
    """Hits the real Mistral API. Requires MISTRAL_API_KEY."""

    def test_send_hello(self):
        if not os.environ.get("MISTRAL_API_KEY"):
            pytest.skip("MISTRAL_API_KEY not set")
        client = LLMClient()
        response = client.send(CompletionRequest(
            model="mistral-small-latest",
            max_tokens=128,
            messages=[InputMessage(role="user", content=[TextBlock(text="Say hello.")])],
        ))
        assert response.text, "Expected non-empty text response"
        assert "mistral" in response.model.lower()
```

The acceptance marker is load-bearing: [`tests/conftest.py`](../tests/conftest.py) installs a network guard that blocks every non-acceptance test from opening a socket. Forget the marker and the test fails with `RuntimeError: Network access blocked in tests`.

## 4. Provider resolution

Source: `LLMClient._resolve_provider()` in [`src/synth_panel/llm/client.py`](../src/synth_panel/llm/client.py).

The flow on `client.send(request)`:

1. **Alias resolution.** `resolve_alias("sonnet")` → `"claude-sonnet-4-6"`. Local prefixes like `ollama:llama3` are stripped here, and a base-URL override is attached for the OpenAI-compatible provider.
2. **Prefix match.** The registry is scanned in order. The first `ProviderConfig` whose `model_prefixes` contains a prefix of the canonical model wins. For example, `"mistral-large-latest"` matches `("mistral-",)` → `MistralProvider`.
3. **Credential fallback.** If no prefix matches, the first registered provider with credentials in the environment is selected. This is why registry order matters and why `OPENAI_COMPAT_CONFIG` stays last.
4. **Hard fail.** If nothing matches and no credentials exist, `LLMError(MISSING_CREDENTIALS)` is raised with a message listing every `<PROVIDER>_API_KEY` variable. Add yours to that message.

Providers are cached per-canonical-model-string inside the client, so `_resolve_provider()` only instantiates each adapter once. Do not store request-scoped state on `self` — instances are shared across threads by the orchestrator.

## 5. Env var conventions

| Pattern | Example | Notes |
|---------|---------|-------|
| `<NAME>_API_KEY` | `MISTRAL_API_KEY` | Required. Missing → `LLMError(MISSING_CREDENTIALS)`. |
| `<NAME>_BASE_URL` | `MISTRAL_BASE_URL` | Optional. Overrides `default_base_url`. Users set this for proxies, VPC endpoints, or local mocks. |
| `SYNTHPANEL_MODEL_ALIASES` | `'{"m-sm":"mistral-small-latest"}'` | Global. Users add their own aliases without touching code. |

If your provider needs additional config (a region, a project ID, an organization header), use the same `<NAME>_*` prefix. Do not introduce unrelated variables — the MCP `env` block in editor configs is a flat dict, and convention is the only thing keeping it legible.

## 6. Testing your adapter

Two test tiers, both required:

**Unit tests** (`tests/test_<name>_provider.py`):

- All network calls mocked with `unittest.mock.patch("httpx.post", …)` or `httpx.stream`.
- Cover: happy path, missing API key, rate limit (429 → retryable), bad request (400 → non-retryable), transport error, base URL override, streaming happy path, deserialization error on malformed JSON.
- Run under the `conftest.py` network guard — any attempt to open a real socket fails loudly.

```bash
pytest tests/test_mistral_provider.py -v
```

**Acceptance tests** (`tests/test_acceptance.py`, `@pytest.mark.acceptance`):

- Hit the real API.
- `pytest.skip()` if the API key is missing — do not fail.
- Default CI runs with `-m "not acceptance"`, so these are skipped unless a maintainer provisions a key.

```bash
MISTRAL_API_KEY=… pytest tests/test_acceptance.py -m acceptance -v
```

### Determinism — the conftest network guard

[`tests/conftest.py`](../tests/conftest.py) auto-applies a fixture that monkeypatches `socket.socket.connect` for every non-acceptance test. Connections to `localhost` / `127.0.0.1` / `::1` are allowed (for local mock servers); everything else raises `RuntimeError`.

Concretely: if you forget to mock `httpx.post` in a unit test, the test fails with a clear error instead of silently burning API credits. Do not disable this fixture.

## 7. Before submitting a PR

Work through this checklist before opening a PR:

- [ ] `send()` and `stream()` both implemented on `LLMProvider`
- [ ] `ProviderConfig` declared with `api_key_env`, `base_url_env`, `default_base_url`, `model_prefixes`
- [ ] Provider registered in `_PROVIDER_REGISTRY` in `client.py` (before `OPENAI_COMPAT_CONFIG`)
- [ ] "No credentials" error message in `_resolve_provider()` lists your `<NAME>_API_KEY`
- [ ] Unit tests cover: happy path, missing key, 429, 400, transport error, base URL override, streaming, deserialization failure
- [ ] Acceptance test present (skipped when API key is missing)
- [ ] `ruff check src/ tests/` clean
- [ ] `mypy src/synth_panel/` clean
- [ ] `CHANGELOG.md` `[Unreleased] → Added` entry
- [ ] `README.md` provider table row added (alphabetized inside the family, or placed beside the closest cousin)
- [ ] **SynthBench submission** (strongly recommended for adapter PRs) — run your adapter against SynthBench benchmarks and link the results in the PR description. See <https://dataviking-tech.github.io/synthbench/submit/>.

## 8. Benchmark your adapter with SynthBench

> Want proof your adapter produces quality synthetic respondents? Benchmark it against [SynthBench](https://dataviking-tech.github.io/synthbench/), an open benchmark for synthetic survey quality.
>
> Heavy adapter PRs should include SynthBench results so reviewers can evaluate behavior, not just code. An adapter that compiles and type-checks is not the same as an adapter that produces responses statistically comparable to real humans — SynthBench measures the second.
>
> - **Before submitting**, point your adapter at the SynthBench benchmark instruments and capture the Synthetic Parity Score (SPS).
> - **Include the run**: leaderboard link or raw numbers, plus the configuration (model, temperature, ensemble weights if any).
> - **Compare against baselines** on the [leaderboard](https://dataviking-tech.github.io/synthbench/leaderboard/). If your provider + model combo beats the current best single-model score, say so in the PR description; reviewers will prioritize empirical evidence over code review.

Adapters that only route bits — no behavioral change — don't need benchmark runs. Adapters that change defaults, tokenization, or sampling do. When in doubt, run it.

---

Questions or stuck? Open a draft PR early and tag a maintainer; adapter PRs are among the easiest to review because the surface area is bounded and the contract is already specified.
