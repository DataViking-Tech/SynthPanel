"""Registry cache with 24h TTL, offline fallback, and conditional refresh.

Cache file lives at ``$SYNTH_PANEL_DATA_DIR/registry-cache.json`` (default
``~/.synthpanel/registry-cache.json``) and has the shape described in
structure.md §5::

    {
      "fetched_at": "2026-04-22T14:03:00Z",
      "source_url": "https://.../default.json",
      "etag": "\"abc123\"" | null,
      "registry": { "schema_version": 1, "packs": [...] }
    }

The top-level :func:`load_registry` is the entry point most callers want:
it consults the cache, conditionally refreshes against the network, and
gracefully degrades to a stale or empty registry when the network is
unreachable.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from synth_panel.registry.fetch import (
    RegistryFetchError,
    fetch_registry_http,
    registry_url,
)

CACHE_FILENAME = "registry-cache.json"
CACHE_TTL = timedelta(hours=24)
DATA_DIR_ENV = "SYNTH_PANEL_DATA_DIR"
OFFLINE_ENV = "SYNTHPANEL_REGISTRY_OFFLINE"
REFRESH_ENV = "SYNTHPANEL_REGISTRY_REFRESH"

EMPTY_REGISTRY: dict[str, Any] = {"schema_version": 1, "packs": []}

WarnFn = Callable[[str], None]


def _data_dir() -> Path:
    d = Path(os.environ.get(DATA_DIR_ENV, "~/.synthpanel")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path() -> Path:
    """Return the absolute path to the registry cache file."""
    return _data_dir() / CACHE_FILENAME


@dataclass(frozen=True)
class CachedRegistry:
    """In-memory view of a parsed cache file."""

    fetched_at: datetime
    source_url: str
    etag: str | None
    registry: dict[str, Any]


def _parse_iso(value: str) -> datetime | None:
    # datetime.fromisoformat on 3.10 does not accept a trailing "Z"; normalize.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def read_cache(path: Path | None = None) -> CachedRegistry | None:
    """Read and parse the cache file. Returns ``None`` if missing/corrupt."""
    target = path or cache_path()
    if not target.exists():
        return None
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None

    fetched = _parse_iso(str(raw.get("fetched_at", "")))
    registry = raw.get("registry")
    source = raw.get("source_url")
    if fetched is None or not isinstance(registry, dict) or not isinstance(source, str):
        return None

    etag = raw.get("etag")
    if not isinstance(etag, str) and etag is not None:
        etag = None

    return CachedRegistry(
        fetched_at=fetched,
        source_url=source,
        etag=etag,
        registry=registry,
    )


def write_cache(
    registry: dict[str, Any],
    *,
    source_url: str,
    etag: str | None = None,
    fetched_at: datetime | None = None,
    path: Path | None = None,
) -> None:
    """Atomically write the cache file.

    Writes to ``<path>.tmp`` then ``os.replace``s into place so a crash
    mid-write cannot leave the cache in a half-written state.
    """
    target = path or cache_path()
    stamp = fetched_at or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "fetched_at": stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_url": source_url,
        "etag": etag,
        "registry": registry,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)


def _is_fresh(cached: CachedRegistry, now: datetime | None = None) -> bool:
    moment = now or datetime.now(timezone.utc)
    return (moment - cached.fetched_at) < CACHE_TTL


def _env_flag(name: str) -> bool:
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def _default_warn(message: str) -> None:
    print(message, file=sys.stderr)


def load_registry(
    *,
    client: httpx.Client | None = None,
    url: str | None = None,
    refresh: bool | None = None,
    offline: bool | None = None,
    warn: WarnFn | None = None,
) -> dict[str, Any]:
    """Return the registry dict, going through the cache layer.

    Semantics (structure.md §5):

    - ``offline`` (flag or ``SYNTHPANEL_REGISTRY_OFFLINE=1``) — never
      touch the network. Returns the cached registry if present, else
      an empty ``{"schema_version": 1, "packs": []}``.
    - ``refresh`` (flag or ``SYNTHPANEL_REGISTRY_REFRESH=1``) — bypass
      the TTL and force a fetch (ignoring any cached ETag).
    - Otherwise, if the cache is fresh (<24h) and matches the active
      URL, return it with zero network calls.
    - If the cache is stale, attempt a conditional GET with the cached
      ETag. 304 → bump ``fetched_at`` and return cached; 200 → overwrite
      cache and return new; network error → stderr warning + stale
      cache (or empty registry if no cache existed).
    """
    target_url = url or registry_url()
    emit = warn or _default_warn

    offline_flag = offline if offline is not None else _env_flag(OFFLINE_ENV)
    refresh_flag = refresh if refresh is not None else _env_flag(REFRESH_ENV)

    cached = read_cache()

    if offline_flag:
        if cached is not None:
            return cached.registry
        return _empty()

    url_matches = cached is not None and cached.source_url == target_url
    if cached is not None and url_matches and not refresh_flag and _is_fresh(cached):
        return cached.registry

    conditional_etag: str | None = None
    if cached is not None and url_matches and not refresh_flag:
        conditional_etag = cached.etag

    try:
        result = fetch_registry_http(target_url, etag=conditional_etag, client=client)
    except RegistryFetchError as exc:
        if cached is not None:
            emit(f"synthpanel: registry fetch failed ({exc}); using stale cache from {cached.fetched_at.isoformat()}")
            return cached.registry
        emit(f"synthpanel: registry unavailable ({exc}); returning empty registry")
        return _empty()

    if result.not_modified and cached is not None:
        write_cache(cached.registry, source_url=target_url, etag=cached.etag)
        return cached.registry

    # result.data is guaranteed non-None for a 200 in fetch_registry_http.
    assert result.data is not None
    write_cache(result.data, source_url=target_url, etag=result.etag)
    return result.data


def _empty() -> dict[str, Any]:
    # Return a fresh copy so callers cannot mutate the module constant.
    return {"schema_version": 1, "packs": []}
