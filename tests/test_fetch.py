"""Unit tests for ``althing.fetch`` (perimeter, cache, ladder).

The full integration test suite — including a stub HTTP server, redirect
chain fixtures, and DNS-rebinding harness — lives in hq-3o1r. This file
covers the no-network paths so coverage stays healthy alongside the
I-phase deliverable.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from althing.fetch.cache import CacheEntry, CacheHit, CacheL1, UrlCache
from althing.fetch.ladder import (
    AttachmentIntent,
    ExtractionFailed,
    LadderConfig,
    OnFailure,
    extract,
)
from althing.fetch.perimeter import (
    ALLOWED_CONTENT_TYPES,
    PerimeterDeny,
    ResolvedTarget,
    safe_resolve,
    sniff_and_validate,
)

# ---------------------------------------------------------------------------
# safe_resolve — denylist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, reason_substr",
    [
        ("127.0.0.1", "loopback"),
        ("169.254.169.254", "link-local"),  # IMDS — also link-local
        ("10.0.0.1", "private"),
        ("192.168.1.1", "private"),
        ("172.16.0.1", "private"),
        ("100.64.0.1", "100.64"),  # CGNAT
        ("::1", "loopback"),
        ("fc00::1", "private"),
        ("fe80::1", "link-local"),
        ("ff02::1", "multicast"),
        ("224.0.0.1", "multicast"),
        ("0.0.0.0", "unspecified"),
        ("::ffff:10.0.0.1", "private"),  # IPv4-mapped private
    ],
)
def test_safe_resolve_denies_unsafe_targets(host: str, reason_substr: str) -> None:
    with pytest.raises(PerimeterDeny) as exc:
        safe_resolve(host)
    assert reason_substr.lower() in str(exc.value).lower()


def test_safe_resolve_allows_public_ip_literal() -> None:
    target = safe_resolve("1.1.1.1")
    assert isinstance(target, ResolvedTarget)
    assert target.ip == "1.1.1.1"


def test_safe_resolve_strips_ipv6_brackets() -> None:
    target = safe_resolve("[2606:4700:4700::1111]")
    assert target.ip == "2606:4700:4700::1111"


def test_safe_resolve_empty_host() -> None:
    with pytest.raises(PerimeterDeny):
        safe_resolve("")


# ---------------------------------------------------------------------------
# sniff_and_validate
# ---------------------------------------------------------------------------


def test_sniff_accepts_png_header() -> None:
    head = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    assert sniff_and_validate(head, "image/png", None) == "image/png"


def test_sniff_accepts_pdf_header() -> None:
    assert sniff_and_validate(b"%PDF-1.4\n", "application/pdf", None) == "application/pdf"


def test_sniff_accepts_html_doctype() -> None:
    assert sniff_and_validate(b"<!DOCTYPE html><html></html>", "text/html", None) == "text/html"


def test_sniff_accepts_text_when_declared_text() -> None:
    # No magic match — we trust the declared text/* type.
    assert sniff_and_validate(b"hello world\n", "text/plain", None) == "text/plain"


def test_sniff_rejects_octet_stream_lying_about_html() -> None:
    with pytest.raises(PerimeterDeny):
        sniff_and_validate(b"<html></html>", "application/octet-stream", None)


def test_sniff_rejects_javascript() -> None:
    # text/javascript is outside the allowlist — must reject.
    with pytest.raises(PerimeterDeny):
        sniff_and_validate(b"function f(){}", "text/javascript", ALLOWED_CONTENT_TYPES)


def test_sniff_rejects_video() -> None:
    with pytest.raises(PerimeterDeny):
        sniff_and_validate(b"\x00\x00\x00 ftypmp42", "video/mp4", ALLOWED_CONTENT_TYPES)


def test_sniff_rejects_opaque_unsniffable() -> None:
    # Random binary, no declared type → can't tell what it is, refuse.
    with pytest.raises(PerimeterDeny):
        sniff_and_validate(b"\x12\x34\x56\x78garbage", "", None)


# ---------------------------------------------------------------------------
# UrlCache + CacheL1
# ---------------------------------------------------------------------------


def test_cache_store_and_lookup_within_ttl(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path, ttl_seconds=300)
    body = b"# hello\n"
    entry = cache.store("https://example.com/x", body, content_type="text/markdown", mode="markdown")
    assert entry.content_sha256

    hit = cache.lookup("https://example.com/x", "markdown")
    assert hit is not None
    assert hit.body == body
    assert hit.entry.stale is False


def test_cache_mode_filter_misses(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path, ttl_seconds=300)
    cache.store("https://example.com/x", b"x", content_type="text/markdown", mode="markdown")
    assert cache.lookup("https://example.com/x", "screenshot") is None


def test_cache_ttl_expiry_unpinned(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path, ttl_seconds=60)
    cache.store("https://example.com/x", b"x", content_type="text/markdown", mode="markdown")
    later = time.time() + 3600
    assert cache.lookup("https://example.com/x", "markdown", now=later) is None


def test_cache_pinned_returns_stale(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path, ttl_seconds=60)
    cache.store("https://example.com/x", b"x", content_type="text/markdown", mode="markdown", pinned=True)
    later = time.time() + 3600
    hit = cache.lookup("https://example.com/x", "markdown", now=later)
    assert hit is not None
    assert hit.entry.stale is True


def test_cache_pin_toggle(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path, ttl_seconds=60)
    cache.store("https://example.com/x", b"x", content_type="text/markdown", mode="markdown")
    assert cache.pin("https://example.com/x", True) is True
    later = time.time() + 3600
    hit = cache.lookup("https://example.com/x", "markdown", now=later)
    assert hit is not None and hit.entry.pinned is True


def test_cache_pin_missing_returns_false(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path)
    assert cache.pin("https://example.com/missing", True) is False


def test_cache_dedup_reuses_blob(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path)
    e1 = cache.store("https://a.example/1", b"same body", content_type="text/markdown")
    e2 = cache.store("https://b.example/2", b"same body", content_type="text/markdown")
    assert e1.content_sha256 == e2.content_sha256
    blobs = list((tmp_path / "blobs").iterdir())
    assert len(blobs) == 1


def test_cache_clear_wipes_disk(tmp_path: Path) -> None:
    cache = UrlCache(tmp_path)
    cache.store("https://example.com/x", b"x", content_type="text/markdown")
    cache.clear()
    assert cache.lookup("https://example.com/x") is None


def test_l1_isolated_per_mode() -> None:
    l1 = CacheL1()
    e_md = CacheEntry(
        url="https://example.com/x",
        content_sha256="a",
        fetched_at=time.time(),
        content_type="text/markdown",
        mode="markdown",
    )
    e_shot = CacheEntry(
        url="https://example.com/x",
        content_sha256="b",
        fetched_at=time.time(),
        content_type="image/png",
        mode="screenshot",
    )
    l1.put(CacheHit(entry=e_md, body=b"md"))
    l1.put(CacheHit(entry=e_shot, body=b"png"))
    assert l1.get("https://example.com/x", "markdown").body == b"md"
    assert l1.get("https://example.com/x", "screenshot").body == b"png"
    assert l1.get("https://example.com/x", "missing") is None
    l1.clear()
    assert l1.get("https://example.com/x", "markdown") is None


# ---------------------------------------------------------------------------
# ladder enums + ExtractionFailed
# ---------------------------------------------------------------------------


def test_attachment_intent_coerce_default_text() -> None:
    assert AttachmentIntent.coerce(None) is AttachmentIntent.TEXT
    assert AttachmentIntent.coerce("visual") is AttachmentIntent.VISUAL
    assert AttachmentIntent.coerce(AttachmentIntent.BOTH) is AttachmentIntent.BOTH


def test_attachment_intent_coerce_invalid() -> None:
    with pytest.raises(ValueError):
        AttachmentIntent.coerce("audio")


def test_on_failure_default_skip_question() -> None:
    assert OnFailure.coerce(None) is OnFailure.SKIP_QUESTION
    assert OnFailure.coerce("abort") is OnFailure.ABORT


def test_on_failure_invalid() -> None:
    with pytest.raises(ValueError):
        OnFailure.coerce("explode")


def test_extraction_failed_carries_url_and_reason() -> None:
    err = ExtractionFailed("https://x.example/page", "no good content")
    assert err.url == "https://x.example/page"
    assert err.reason == "no good content"
    assert "no good content" in str(err)


def test_ladder_extract_visual_without_playwright_raises(monkeypatch, tmp_path: Path) -> None:
    """When Playwright isn't available, visual-only intent must surface
    ExtractionFailed before we touch the network."""

    # Force the screenshot step to act as if Playwright isn't installed.
    import althing.fetch.ladder as ladder_mod

    monkeypatch.setattr(ladder_mod, "_step_screenshot", lambda url, cfg: None)

    cfg = LadderConfig(intent=AttachmentIntent.VISUAL)
    # Use a public IP literal so safe_resolve doesn't hit DNS, but we
    # still surface ExtractionFailed because the screenshot step is a no-op.
    with pytest.raises(ExtractionFailed):
        extract("https://1.1.1.1/", cfg)


def test_ladder_extract_perimeter_deny_for_private_target() -> None:
    """SSRF check fires before any network I/O for private targets."""
    cfg = LadderConfig(intent=AttachmentIntent.VISUAL)
    with pytest.raises(PerimeterDeny):
        extract("https://10.0.0.1/", cfg)
