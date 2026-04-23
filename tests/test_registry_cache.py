"""Tests for the registry cache layer (``synth_panel.registry.cache``).

Covers TTL, ETag conditional refresh, offline mode, stale fallback with
warning, and the empty-registry graceful-degrade path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from synth_panel.registry import (
    CACHE_FILENAME,
    CACHE_TTL,
    EMPTY_REGISTRY,
    OFFLINE_ENV,
    REFRESH_ENV,
    cache_path,
    fetch_registry,
    read_cache,
    resolve_pack,
    write_cache,
)
from synth_panel.registry.cache import load_registry
from synth_panel.registry.fetch import REGISTRY_URL_ENV

SAMPLE = {
    "schema_version": 1,
    "packs": [
        {
            "id": "icp-traitprint-cloud",
            "kind": "persona",
            "name": "Traitprint",
            "description": "d",
            "repo": "example/traitprint",
            "path": "synthpanel-pack.yaml",
            "ref": "v0.1",
            "author": {"github": "example"},
            "tags": ["b2b"],
            "added_at": "2026-04-22",
        }
    ],
}

URL = "https://example.test/default.json"


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin SYNTH_PANEL_DATA_DIR at a temp dir per test; clear env flags."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(REGISTRY_URL_ENV, URL)
    monkeypatch.delenv(OFFLINE_ENV, raising=False)
    monkeypatch.delenv(REFRESH_ENV, raising=False)


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _explode(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected network call: {request.method} {request.url}")


# ---------- cache path + round-trip ----------


def test_cache_path_honors_data_dir_env(tmp_path: Path) -> None:
    assert cache_path() == tmp_path / CACHE_FILENAME


def test_cache_round_trip_preserves_registry_and_etag() -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    cached = read_cache()
    assert cached is not None
    assert cached.registry == SAMPLE
    assert cached.source_url == URL
    assert cached.etag == '"abc"'


def test_read_cache_missing_returns_none() -> None:
    assert read_cache() is None


def test_read_cache_corrupt_returns_none(tmp_path: Path) -> None:
    cache_path().write_text("{not json", encoding="utf-8")
    assert read_cache() is None


def test_read_cache_wrong_shape_returns_none() -> None:
    cache_path().write_text(json.dumps({"nope": True}), encoding="utf-8")
    assert read_cache() is None


# ---------- fresh cache → zero network ----------


def test_fresh_cache_hit_skips_network() -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    with _client(_explode) as client:
        data = load_registry(client=client)
    assert data == SAMPLE


def test_fresh_cache_is_returned_by_fetch_registry_public_api() -> None:
    write_cache(SAMPLE, source_url=URL, etag=None)
    with _client(_explode) as client:
        data = fetch_registry(client=client)
    assert data["packs"][0]["id"] == "icp-traitprint-cloud"


# ---------- stale cache → conditional refresh ----------


def test_stale_cache_304_keeps_cached_registry_and_bumps_fetched_at() -> None:
    stale = datetime.now(timezone.utc) - CACHE_TTL - timedelta(hours=1)
    write_cache(SAMPLE, source_url=URL, etag='"abc"', fetched_at=stale)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("if-none-match") == '"abc"'
        return httpx.Response(304)

    with _client(handler) as client:
        data = load_registry(client=client)

    assert data == SAMPLE
    cached = read_cache()
    assert cached is not None
    # fetched_at was bumped to ~now (within the last minute)
    assert (datetime.now(timezone.utc) - cached.fetched_at) < timedelta(minutes=1)


def test_stale_cache_200_overwrites_with_new_payload() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    write_cache(SAMPLE, source_url=URL, etag='"old"', fetched_at=stale)

    updated = {
        "schema_version": 1,
        "packs": [{"id": "new-pack", "repo": "x/new", "path": "p.yaml"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=updated, headers={"ETag": '"new"'})

    with _client(handler) as client:
        data = load_registry(client=client)

    assert data == updated
    cached = read_cache()
    assert cached is not None
    assert cached.etag == '"new"'
    assert cached.registry == updated


def test_stale_cache_network_fail_returns_stale_with_warning() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=48)
    write_cache(SAMPLE, source_url=URL, etag='"abc"', fetched_at=stale)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    warnings: list[str] = []
    with _client(handler) as client:
        data = load_registry(client=client, warn=warnings.append)

    assert data == SAMPLE
    assert any("stale cache" in w for w in warnings)


# ---------- cache miss + fetch fail → empty registry ----------


def test_no_cache_and_fetch_fail_returns_empty_registry_gracefully() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    warnings: list[str] = []
    with _client(handler) as client:
        data = load_registry(client=client, warn=warnings.append)

    assert data == EMPTY_REGISTRY
    assert data["packs"] == []
    assert any("registry unavailable" in w for w in warnings)
    # Mutating returned dict must not mutate the module constant.
    data["packs"].append({"id": "leak"})
    assert EMPTY_REGISTRY["packs"] == []


def test_no_cache_and_http_404_returns_empty_registry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    warnings: list[str] = []
    with _client(handler) as client:
        data = load_registry(client=client, warn=warnings.append)

    assert data == EMPTY_REGISTRY


# ---------- SYNTHPANEL_REGISTRY_OFFLINE ----------


def test_offline_env_var_prevents_any_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"', fetched_at=datetime.now(timezone.utc) - timedelta(hours=48))
    monkeypatch.setenv(OFFLINE_ENV, "1")

    with _client(_explode) as client:
        data = load_registry(client=client)

    assert data == SAMPLE


def test_offline_flag_argument_prevents_network() -> None:
    write_cache(SAMPLE, source_url=URL, etag=None, fetched_at=datetime.now(timezone.utc) - timedelta(hours=48))
    with _client(_explode) as client:
        data = load_registry(client=client, offline=True)
    assert data == SAMPLE


def test_offline_with_no_cache_returns_empty_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(OFFLINE_ENV, "1")
    with _client(_explode) as client:
        data = load_registry(client=client)
    assert data == EMPTY_REGISTRY


def test_offline_env_var_false_values_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "0"/"false" must NOT trigger offline mode.
    monkeypatch.setenv(OFFLINE_ENV, "0")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SAMPLE)

    with _client(handler) as client:
        data = load_registry(client=client)
    assert data == SAMPLE


# ---------- SYNTHPANEL_REGISTRY_REFRESH ----------


def test_refresh_env_bypasses_fresh_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')
    monkeypatch.setenv(REFRESH_ENV, "1")

    new_payload = {"schema_version": 1, "packs": []}
    seen_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(dict(request.headers))
        return httpx.Response(200, json=new_payload)

    with _client(handler) as client:
        data = load_registry(client=client)

    assert data == new_payload
    # REFRESH must skip the conditional ETag so we don't accept a 304.
    assert "if-none-match" not in seen_headers[0]


def test_refresh_flag_argument_bypasses_fresh_cache() -> None:
    write_cache(SAMPLE, source_url=URL, etag='"abc"')

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"schema_version": 1, "packs": []})

    with _client(handler) as client:
        data = load_registry(client=client, refresh=True)
    assert data["packs"] == []


# ---------- SYNTHPANEL_REGISTRY_URL override on cache ----------


def test_url_change_invalidates_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    write_cache(SAMPLE, source_url="https://old.example/r.json", etag='"abc"')
    other_url = "https://new.example/r.json"
    monkeypatch.setenv(REGISTRY_URL_ENV, other_url)

    replacement = {"schema_version": 1, "packs": []}
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        assert "if-none-match" not in request.headers
        return httpx.Response(200, json=replacement)

    with _client(handler) as client:
        data = load_registry(client=client)

    assert seen_urls == [other_url]
    assert data == replacement


# ---------- resolve_pack ----------


def test_resolve_pack_hit_returns_entry() -> None:
    write_cache(SAMPLE, source_url=URL, etag=None)
    with _client(_explode) as client:
        entry = resolve_pack("icp-traitprint-cloud", client=client)

    assert entry is not None
    assert entry.id == "icp-traitprint-cloud"
    assert entry.repo == "example/traitprint"
    assert entry.ref == "v0.1"
    assert entry.tags == ("b2b",)
    assert entry.author == {"github": "example"}


def test_resolve_pack_miss_returns_none() -> None:
    write_cache(SAMPLE, source_url=URL, etag=None)
    with _client(_explode) as client:
        entry = resolve_pack("nonexistent", client=client)
    assert entry is None


def test_resolve_pack_empty_registry_returns_none(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with _client(handler) as client:
        entry = resolve_pack("anything", client=client)
    assert entry is None
    # Warning goes to stderr via the default warn hook.
    assert "registry unavailable" in capsys.readouterr().err


def test_resolve_pack_missing_ref_defaults_to_main() -> None:
    registry = {
        "schema_version": 1,
        "packs": [
            {
                "id": "no-ref",
                "kind": "persona",
                "name": "No Ref",
                "description": "",
                "repo": "example/no-ref",
                "path": "synthpanel-pack.yaml",
                "author": {"github": "example"},
            }
        ],
    }
    write_cache(registry, source_url=URL, etag=None)
    with _client(_explode) as client:
        entry = resolve_pack("no-ref", client=client)
    assert entry is not None
    assert entry.ref == "main"


# ---------- cache write atomicity ----------


def test_write_cache_does_not_leave_tmp_file_behind() -> None:
    write_cache(SAMPLE, source_url=URL, etag=None)
    tmp_files = list(cache_path().parent.glob(f"{CACHE_FILENAME}.tmp*"))
    assert tmp_files == []


def test_write_cache_creates_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nested = tmp_path / "nested" / "data"
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(nested))
    write_cache(SAMPLE, source_url=URL, etag=None)
    assert (nested / CACHE_FILENAME).exists()
