"""Pack registry support.

Three layers, each usable on its own:

1. :mod:`.github` — parses ``gh:user/repo[@ref][:path]`` and
   ``github.com/blob/`` URLs into raw-content URLs.
2. :mod:`.fetch` — HTTP GET of the registry ``default.json``
   (conditional on ETag), returning a parsed dict.
3. :mod:`.cache` — 24h TTL cache at
   ``$SYNTH_PANEL_DATA_DIR/registry-cache.json`` with offline and
   stale-fallback behavior.

Most callers want :func:`fetch_registry` (cache-aware) and
:func:`resolve_pack` (id → :class:`RegistryEntry`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from synth_panel.registry.cache import (
    CACHE_FILENAME,
    CACHE_TTL,
    DATA_DIR_ENV,
    EMPTY_REGISTRY,
    OFFLINE_ENV,
    REFRESH_ENV,
    CachedRegistry,
    cache_path,
    load_registry,
    read_cache,
    write_cache,
)
from synth_panel.registry.fetch import (
    DEFAULT_REGISTRY_URL,
    FETCH_TIMEOUT,
    REGISTRY_URL_ENV,
    FetchResult,
    RegistryFetchError,
    fetch_registry_http,
    registry_url,
)
from synth_panel.registry.github import (
    DEFAULT_PATH,
    DEFAULT_REF,
    GitHubSource,
    parse_gh_source,
    resolve_source,
)


@dataclass(frozen=True)
class RegistryEntry:
    """One entry from ``default.json`` — identifies a registered pack.

    Mirrors the schema documented in structure.md §2. ``ref`` defaults
    to ``"main"`` when the JSON entry omits it; ``calibration`` is
    reserved (always ``None`` in v1).
    """

    id: str
    kind: str
    name: str
    description: str
    repo: str
    path: str
    author: dict[str, Any]
    ref: str = "main"
    version: str | None = None
    tags: tuple[str, ...] = ()
    added_at: str | None = None
    calibration: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegistryEntry:
        """Hydrate a :class:`RegistryEntry` from a ``default.json`` entry.

        Raises ``KeyError`` if ``id``/``repo``/``path`` are missing
        (the three fields we require for any useful resolution).
        """
        raw_tags = data.get("tags") or ()
        if isinstance(raw_tags, (list, tuple)):
            tags = tuple(t for t in raw_tags if isinstance(t, str))
        else:
            tags = ()

        author = data.get("author") or {}
        if not isinstance(author, dict):
            author = {}

        version = data.get("version")
        if version is not None and not isinstance(version, str):
            version = str(version)

        added_at = data.get("added_at")
        if added_at is not None and not isinstance(added_at, str):
            added_at = str(added_at)

        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind", "persona")),
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            repo=str(data["repo"]),
            path=str(data["path"]),
            author=dict(author),
            ref=str(data.get("ref") or "main"),
            version=version,
            tags=tags,
            added_at=added_at,
            calibration=data.get("calibration"),
        )


def fetch_registry(
    *,
    client: httpx.Client | None = None,
    url: str | None = None,
    refresh: bool | None = None,
    offline: bool | None = None,
) -> dict[str, Any]:
    """Return the parsed registry dict, using the cache where possible.

    See :func:`synth_panel.registry.cache.load_registry` for the full
    fresh/stale/offline decision tree. This is the thin public wrapper.
    """
    return load_registry(client=client, url=url, refresh=refresh, offline=offline)


def resolve_pack(
    pack_id: str,
    *,
    client: httpx.Client | None = None,
    url: str | None = None,
    refresh: bool | None = None,
    offline: bool | None = None,
) -> RegistryEntry | None:
    """Look up a pack by id; returns ``None`` if it isn't in the registry."""
    registry = fetch_registry(client=client, url=url, refresh=refresh, offline=offline)
    for entry in registry.get("packs", []) or []:
        if isinstance(entry, dict) and entry.get("id") == pack_id:
            try:
                return RegistryEntry.from_dict(entry)
            except (KeyError, TypeError, ValueError):
                return None
    return None


__all__ = [
    "CACHE_FILENAME",
    "CACHE_TTL",
    "DATA_DIR_ENV",
    "DEFAULT_PATH",
    "DEFAULT_REF",
    "DEFAULT_REGISTRY_URL",
    "EMPTY_REGISTRY",
    "FETCH_TIMEOUT",
    "OFFLINE_ENV",
    "REFRESH_ENV",
    "REGISTRY_URL_ENV",
    "CachedRegistry",
    "FetchResult",
    "GitHubSource",
    "RegistryEntry",
    "RegistryFetchError",
    "cache_path",
    "fetch_registry",
    "fetch_registry_http",
    "parse_gh_source",
    "read_cache",
    "registry_url",
    "resolve_pack",
    "resolve_source",
    "write_cache",
]
