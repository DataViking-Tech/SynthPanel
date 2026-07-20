"""Content-addressable on-disk cache for fetched URL attachments.

Implements the caching policy from hq-hqlp §4.

Layout (default root: ``~/.althing/cache/url/``)::

    pointers/<sha256(url)>.json   ->  {url, content_sha256, fetched_at,
                                       content_type, mode, pinned}
    blobs/<content_sha256>        ->  raw bytes (markdown / extracted text /
                                       PNG / PDF / etc.)

Two layers sit in front of the disk store:

- ``CacheL1`` — per-run in-memory dict keyed by URL. Avoids disk thrash
  when a panel run reuses the same attachment URL across personas.
- ``UrlCache`` — disk-backed pointer + blob store. TTL defaults to 15
  minutes; pinned pointers never expire. LRU eviction caps the blob
  store at 2 GiB by default.

The cache is intentionally agnostic about *what* the bytes are; the
ladder writes already-extracted markdown / screenshot bytes alongside a
``mode`` tag so a later read can decide whether the cached bytes match
the caller's current ``attachment_intent``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TTL_SECONDS = 15 * 60  # 15 minutes (hq-hqlp §4)
DEFAULT_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB blob-store cap
DEFAULT_CACHE_ROOT = Path.home() / ".althing" / "cache" / "url"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass
class CacheEntry:
    """Pointer record describing a cached URL fetch."""

    url: str
    content_sha256: str
    fetched_at: float  # unix seconds
    content_type: str
    mode: str  # markdown | html-extracted | screenshot | raw | etc.
    pinned: bool = False
    stale: bool = False  # set on read when blob was returned past TTL

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "content_sha256": self.content_sha256,
            "fetched_at": self.fetched_at,
            "content_type": self.content_type,
            "mode": self.mode,
            "pinned": self.pinned,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(
            url=data["url"],
            content_sha256=data["content_sha256"],
            fetched_at=float(data["fetched_at"]),
            content_type=data.get("content_type", ""),
            mode=data.get("mode", "raw"),
            pinned=bool(data.get("pinned", False)),
        )


@dataclass
class CacheHit:
    """Successful cache lookup."""

    entry: CacheEntry
    body: bytes


@dataclass
class CacheL1:
    """Per-panel-run in-memory cache.

    Keys are ``(url, mode)`` tuples so that a question requesting
    ``screenshot`` doesn't accidentally read the markdown blob a
    sibling question stored for the same URL.
    """

    _store: dict[tuple[str, str], CacheHit] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, url: str, mode: str) -> CacheHit | None:
        with self._lock:
            return self._store.get((url, mode))

    def put(self, hit: CacheHit) -> None:
        with self._lock:
            self._store[(hit.entry.url, hit.entry.mode)] = hit

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class UrlCache:
    """Disk-backed content-addressable URL cache.

    Construction is cheap; directories are created lazily on first
    write. The class is safe to share across threads in a single
    process — disk operations are guarded by a single lock to keep
    pointer writes atomic.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self.root = Path(root) if root is not None else DEFAULT_CACHE_ROOT
        self.ttl_seconds = ttl_seconds
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    # -- paths ----------------------------------------------------------

    @property
    def _pointers_dir(self) -> Path:
        return self.root / "pointers"

    @property
    def _blobs_dir(self) -> Path:
        return self.root / "blobs"

    def _pointer_path(self, url: str) -> Path:
        return self._pointers_dir / f"{_sha256_text(url)}.json"

    def _blob_path(self, content_sha: str) -> Path:
        return self._blobs_dir / content_sha

    def _ensure_dirs(self) -> None:
        self._pointers_dir.mkdir(parents=True, exist_ok=True)
        self._blobs_dir.mkdir(parents=True, exist_ok=True)

    # -- read -----------------------------------------------------------

    def lookup(self, url: str, mode: str | None = None, *, now: float | None = None) -> CacheHit | None:
        """Return a cache hit for ``url`` if one is fresh (or pinned).

        ``mode`` filters by the recorded extraction mode — passing
        ``None`` matches any mode (caller knows what it asked for).
        Stale-but-pinned hits are returned with ``entry.stale=True`` so
        callers can tag the result envelope.
        """
        now = now if now is not None else time.time()
        path = self._pointer_path(url)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        try:
            entry = CacheEntry.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None

        if mode is not None and entry.mode != mode:
            return None

        blob_path = self._blob_path(entry.content_sha256)
        if not blob_path.exists():
            return None

        age = now - entry.fetched_at
        is_fresh = age <= self.ttl_seconds
        if not is_fresh and not entry.pinned:
            return None

        try:
            body = blob_path.read_bytes()
        except OSError:
            return None

        # Touch atime so LRU eviction sees this as recently used.
        with contextlib.suppress(OSError):
            os.utime(blob_path, None)

        entry.stale = not is_fresh
        return CacheHit(entry=entry, body=body)

    # -- write ----------------------------------------------------------

    def store(
        self,
        url: str,
        body: bytes,
        *,
        content_type: str,
        mode: str = "raw",
        pinned: bool = False,
        now: float | None = None,
    ) -> CacheEntry:
        """Persist ``body`` and write a pointer for ``url``.

        If the same content already exists on disk (deduplication via
        ``content_sha256``) the existing blob is reused and only the
        pointer's ``fetched_at`` is refreshed.
        """
        now = now if now is not None else time.time()
        with self._lock:
            self._ensure_dirs()
            content_sha = _sha256_bytes(body)
            blob_path = self._blob_path(content_sha)

            if not blob_path.exists():
                tmp = blob_path.with_suffix(".tmp")
                tmp.write_bytes(body)
                tmp.replace(blob_path)

            entry = CacheEntry(
                url=url,
                content_sha256=content_sha,
                fetched_at=now,
                content_type=content_type,
                mode=mode,
                pinned=pinned,
            )
            pointer_path = self._pointer_path(url)
            tmp_ptr = pointer_path.with_suffix(".tmp")
            tmp_ptr.write_text(json.dumps(entry.to_dict()))
            tmp_ptr.replace(pointer_path)

            self._evict_if_over_cap()
            return entry

    def refresh_pointer(self, entry: CacheEntry, *, now: float | None = None) -> CacheEntry:
        """Refresh ``fetched_at`` for an existing pointer (refetch confirmed unchanged)."""
        now = now if now is not None else time.time()
        with self._lock:
            self._ensure_dirs()
            entry.fetched_at = now
            entry.stale = False
            pointer_path = self._pointer_path(entry.url)
            tmp_ptr = pointer_path.with_suffix(".tmp")
            tmp_ptr.write_text(json.dumps(entry.to_dict()))
            tmp_ptr.replace(pointer_path)
            return entry

    def pin(self, url: str, pinned: bool = True) -> bool:
        """Toggle the ``pinned`` flag on a pointer. Returns False if absent."""
        with self._lock:
            path = self._pointer_path(url)
            if not path.exists():
                return False
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return False
            data["pinned"] = pinned
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(path)
            return True

    # -- maintenance ----------------------------------------------------

    def _blob_total_bytes(self) -> int:
        if not self._blobs_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self._blobs_dir.iterdir() if p.is_file())

    def _evict_if_over_cap(self) -> None:
        """LRU-evict blobs (by atime) until total ≤ ``max_bytes``."""
        if not self._blobs_dir.exists():
            return
        blobs = [p for p in self._blobs_dir.iterdir() if p.is_file()]
        total = sum(p.stat().st_size for p in blobs)
        if total <= self.max_bytes:
            return
        # Build set of blob hashes still referenced by a pinned pointer
        # so we don't evict pinned content out from under callers.
        pinned_hashes: set[str] = set()
        if self._pointers_dir.exists():
            for ptr in self._pointers_dir.iterdir():
                if not ptr.is_file():
                    continue
                try:
                    data = json.loads(ptr.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if data.get("pinned"):
                    sha = data.get("content_sha256")
                    if isinstance(sha, str):
                        pinned_hashes.add(sha)

        # Sort by access time, oldest first.
        blobs.sort(key=lambda p: p.stat().st_atime)
        for blob in blobs:
            if total <= self.max_bytes:
                break
            if blob.name in pinned_hashes:
                continue
            size = blob.stat().st_size
            try:
                blob.unlink()
            except OSError:
                continue
            total -= size
            self._reap_pointers_for_blob(blob.name)

    def _reap_pointers_for_blob(self, content_sha: str) -> None:
        if not self._pointers_dir.exists():
            return
        for ptr in self._pointers_dir.iterdir():
            if not ptr.is_file():
                continue
            try:
                data = json.loads(ptr.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if data.get("content_sha256") == content_sha:
                with contextlib.suppress(OSError):
                    ptr.unlink()

    def clear(self) -> None:
        """Remove all cached pointers and blobs. Primarily for tests."""
        with self._lock:
            for sub in (self._pointers_dir, self._blobs_dir):
                if not sub.exists():
                    continue
                for p in sub.iterdir():
                    if p.is_file():
                        with contextlib.suppress(OSError):
                            p.unlink()


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TTL_SECONDS",
    "CacheEntry",
    "CacheHit",
    "CacheL1",
    "UrlCache",
]
