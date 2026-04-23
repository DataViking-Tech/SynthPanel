"""HTTP fetch of the synthpanel pack registry ``default.json``.

This module is the raw network layer. It performs a single GET (optionally
conditional on a cached ETag) and returns the parsed JSON body plus any
``ETag`` header the server sent back. Caching, TTL, and offline behavior
live in :mod:`synth_panel.registry.cache`.

The registry URL defaults to
``https://raw.githubusercontent.com/DataViking-Tech/synthpanel-registry/main/default.json``
and can be overridden at runtime via ``SYNTHPANEL_REGISTRY_URL`` — useful
for tests, forks, and air-gapped environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/DataViking-Tech/synthpanel-registry/main/default.json"
REGISTRY_URL_ENV = "SYNTHPANEL_REGISTRY_URL"
FETCH_TIMEOUT = 10.0


def registry_url() -> str:
    """Return the active registry URL, honoring ``SYNTHPANEL_REGISTRY_URL``."""
    return os.environ.get(REGISTRY_URL_ENV, DEFAULT_REGISTRY_URL)


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a single HTTP fetch against the registry.

    On a 200 response, ``data`` holds the parsed JSON body and ``etag``
    is whatever the server returned (possibly ``None``). On a 304
    (conditional-GET hit against ``If-None-Match``), ``data`` is ``None``
    and ``not_modified`` is ``True``; the caller is expected to keep
    using its cached payload.
    """

    data: dict[str, Any] | None
    etag: str | None
    not_modified: bool


class RegistryFetchError(Exception):
    """Network, HTTP-status, or JSON-parse failure during registry fetch."""


def fetch_registry_http(
    url: str | None = None,
    *,
    etag: str | None = None,
    client: httpx.Client | None = None,
    timeout: float = FETCH_TIMEOUT,
) -> FetchResult:
    """GET the registry JSON, optionally conditional on ``etag``.

    Returns a :class:`FetchResult`:

    - 200 → ``data`` is the parsed JSON dict, ``etag`` is the new
      server-sent value (or ``None``), ``not_modified`` is ``False``.
    - 304 → ``data`` is ``None``, ``etag`` echoes the conditional value
      the caller supplied, ``not_modified`` is ``True``.

    Any other status, any ``httpx.HTTPError``, malformed JSON, or a
    non-object top-level value raises :class:`RegistryFetchError`.

    ``client`` is an optional pre-configured :class:`httpx.Client`
    (tests pass a ``MockTransport``-backed client here). When omitted,
    a fresh client with a 10-second timeout is used and closed on exit.
    """
    target = url or registry_url()
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag

    owns_client = client is None
    active = client or httpx.Client(timeout=timeout, follow_redirects=True)
    try:
        try:
            resp = active.get(target, headers=headers)
        except httpx.HTTPError as exc:
            raise RegistryFetchError(f"network error fetching {target}: {exc}") from exc

        if resp.status_code == 304:
            return FetchResult(data=None, etag=etag, not_modified=True)
        if resp.status_code != 200:
            raise RegistryFetchError(f"unexpected HTTP {resp.status_code} from {target}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise RegistryFetchError(f"malformed JSON from {target}: {exc}") from exc
        if not isinstance(body, dict):
            raise RegistryFetchError(f"expected JSON object from {target}, got {type(body).__name__}")

        new_etag = resp.headers.get("ETag") or resp.headers.get("etag")
        return FetchResult(data=body, etag=new_etag, not_modified=False)
    finally:
        if owns_client:
            active.close()
