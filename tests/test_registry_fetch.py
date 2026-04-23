"""Tests for the raw-HTTP registry fetcher (``synth_panel.registry.fetch``)."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from synth_panel.registry.fetch import (
    DEFAULT_REGISTRY_URL,
    REGISTRY_URL_ENV,
    RegistryFetchError,
    fetch_registry_http,
    registry_url,
)

SAMPLE_REGISTRY = {
    "schema_version": 1,
    "generated_at": "2026-04-22T00:00:00Z",
    "packs": [
        {
            "id": "icp-traitprint-cloud",
            "kind": "persona",
            "name": "Traitprint Cloud ICPs",
            "description": "Buyer archetypes for Traitprint Cloud.",
            "repo": "example/traitprint",
            "path": "synthpanel-pack.yaml",
            "ref": "v0.2.0",
            "author": {"github": "example"},
            "tags": ["b2b", "saas"],
            "added_at": "2026-04-22",
            "calibration": None,
        }
    ],
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------- registry_url / env override ----------


def test_registry_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(REGISTRY_URL_ENV, raising=False)
    assert registry_url() == DEFAULT_REGISTRY_URL


def test_registry_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    override = "https://example.internal/registry.json"
    monkeypatch.setenv(REGISTRY_URL_ENV, override)
    assert registry_url() == override


# ---------- happy path ----------


def test_fetch_200_returns_parsed_dict_and_etag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "If-None-Match" not in request.headers
        return httpx.Response(
            200,
            json=SAMPLE_REGISTRY,
            headers={"ETag": '"abc123"'},
        )

    with _client(handler) as client:
        result = fetch_registry_http("https://example.test/default.json", client=client)

    assert result.not_modified is False
    assert result.data == SAMPLE_REGISTRY
    assert result.etag == '"abc123"'


def test_fetch_uses_env_override_when_url_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(REGISTRY_URL_ENV, "https://env.example/registry.json")

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=SAMPLE_REGISTRY)

    with _client(handler) as client:
        fetch_registry_http(client=client)

    assert seen == ["https://env.example/registry.json"]


def test_fetch_200_without_etag_header_returns_none_etag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SAMPLE_REGISTRY)

    with _client(handler) as client:
        result = fetch_registry_http("https://example.test/default.json", client=client)

    assert result.etag is None
    assert result.data == SAMPLE_REGISTRY


# ---------- conditional GET (ETag) ----------


def test_fetch_sends_if_none_match_when_etag_supplied() -> None:
    seen_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.update(dict(request.headers))
        return httpx.Response(200, json=SAMPLE_REGISTRY)

    with _client(handler) as client:
        fetch_registry_http(
            "https://example.test/default.json",
            etag='"abc123"',
            client=client,
        )

    assert seen_headers.get("if-none-match") == '"abc123"'


def test_fetch_304_returns_not_modified_and_echoes_etag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == '"abc123"'
        return httpx.Response(304)

    with _client(handler) as client:
        result = fetch_registry_http(
            "https://example.test/default.json",
            etag='"abc123"',
            client=client,
        )

    assert result.not_modified is True
    assert result.data is None
    assert result.etag == '"abc123"'


# ---------- error surfaces ----------


def test_fetch_404_raises_registry_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="HTTP 404"):
        fetch_registry_http("https://example.test/default.json", client=client)


def test_fetch_500_raises_registry_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="HTTP 500"):
        fetch_registry_http("https://example.test/default.json", client=client)


def test_fetch_malformed_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{this is not json",
            headers={"content-type": "application/json"},
        )

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="malformed JSON"):
        fetch_registry_http("https://example.test/default.json", client=client)


def test_fetch_non_object_json_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="expected JSON object"):
        fetch_registry_http("https://example.test/default.json", client=client)


def test_fetch_timeout_raises_registry_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="network error"):
        fetch_registry_http("https://example.test/default.json", client=client)


def test_fetch_connect_error_raises_registry_fetch_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with _client(handler) as client, pytest.raises(RegistryFetchError, match="network error"):
        fetch_registry_http("https://example.test/default.json", client=client)


# ---------- response body integrity ----------


def test_fetch_preserves_nested_pack_entries() -> None:
    registry = {
        "schema_version": 1,
        "packs": [
            {"id": "a", "repo": "x/a", "path": "p.yaml", "tags": ["t1", "t2"]},
            {"id": "b", "repo": "x/b", "path": "p.yaml", "author": {"github": "me"}},
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=json.dumps(registry).encode("utf-8"), headers={"content-type": "application/json"}
        )

    with _client(handler) as client:
        result = fetch_registry_http("https://example.test/default.json", client=client)

    assert result.data == registry
