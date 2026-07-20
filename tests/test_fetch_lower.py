"""Unit tests for ``althing.fetch.lower`` (URLBlock frame-stage lowering, hq-8iz3).

Covers the AC from the bead:

- Bank-ref URL attachment reaches the model as fetched markdown TextBlock.
- Multiple panelists in the same panel reuse cached content (L1 hit, no double-fetch).
- SSRF perimeter still enforced (private-IP URL is a hard error, not a wire send).
- URLBlock no longer reaches the wire — every block in the output is concrete.
- Tests cover happy path + fetch-failure + cache-reuse.

sy-550: a URL attachment that yields no usable content is now a **hard error**
by default (``AttachmentFetchError``) so a persona never answers blind on
missing content. The legacy placeholder behaviour is opt-in via
``allow_empty=True`` (wired to ``--allow-empty-attachments``). Fetch status
is recorded into an optional ``status_sink`` for auditability.

The ladder is monkeypatched to keep these tests offline; the network paths
themselves are exercised in ``tests/test_fetch.py`` and the integration
suite (``tests/attachments/``).
"""

from __future__ import annotations

import time

import pytest

from althing.fetch import lower as lower_mod
from althing.fetch.cache import CacheEntry, CacheHit, CacheL1
from althing.fetch.ladder import (
    AttachmentIntent,
    ExtractionFailed,
    LadderResult,
)
from althing.fetch.lower import AttachmentFetchError
from althing.fetch.perimeter import PerimeterDeny
from althing.llm.models import (
    ImageBlock,
    InlineSource,
    TextBlock,
    URLBlock,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ok_text_result(url: str, text: str = "Hello world from the fetched page.") -> LadderResult:
    return LadderResult(
        url=url,
        intent=AttachmentIntent.TEXT,
        text=text,
        text_mode="markdown",
        screenshot=None,
        screenshot_mode=None,
        final_url=url,
        redirect_chain=[],
        stale=False,
        fetched=True,
    )


def _ok_screenshot_result(url: str, png: bytes = b"\x89PNGfake") -> LadderResult:
    return LadderResult(
        url=url,
        intent=AttachmentIntent.VISUAL,
        text=None,
        text_mode=None,
        screenshot=png,
        screenshot_mode="screenshot",
        final_url=url,
        redirect_chain=[],
        stale=False,
        fetched=True,
    )


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


def test_non_url_blocks_pass_through_unchanged() -> None:
    inputs = [TextBlock(text="hi"), TextBlock(text="there")]
    out = lower_mod.lower_url_blocks(inputs)
    assert out == inputs


def test_urlblock_text_intent_lowers_to_text(monkeypatch) -> None:
    captured: list[tuple[str, AttachmentIntent]] = []

    def fake_extract(url, cfg):
        captured.append((url, cfg.intent))
        return _ok_text_result(url, "extracted markdown body")

    monkeypatch.setattr(lower_mod, "extract", fake_extract)

    blocks = [URLBlock(url="https://example.com/a", fetch_mode="auto")]
    out = lower_mod.lower_url_blocks(blocks)

    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert out[0].text == "extracted markdown body"
    # auto -> TEXT intent
    assert captured == [("https://example.com/a", AttachmentIntent.TEXT)]


def test_urlblock_screenshot_intent_lowers_to_image(monkeypatch) -> None:
    def fake_extract(url, cfg):
        assert cfg.intent is AttachmentIntent.VISUAL
        return _ok_screenshot_result(url, png=b"\x89PNGwhatever")

    monkeypatch.setattr(lower_mod, "extract", fake_extract)

    blocks = [URLBlock(url="https://example.com/b", fetch_mode="screenshot")]
    out = lower_mod.lower_url_blocks(blocks)

    assert len(out) == 1
    assert isinstance(out[0], ImageBlock)
    assert out[0].media_type == "image/png"
    assert isinstance(out[0].source, InlineSource)
    # base64-encoded payload, not the raw bytes.
    import base64

    assert base64.b64decode(out[0].source.data) == b"\x89PNGwhatever"


def test_urlblock_html_text_mode_maps_to_text_intent(monkeypatch) -> None:
    captured: list[AttachmentIntent] = []

    def fake_extract(url, cfg):
        captured.append(cfg.intent)
        return _ok_text_result(url)

    monkeypatch.setattr(lower_mod, "extract", fake_extract)

    lower_mod.lower_url_blocks([URLBlock(url="https://x/", fetch_mode="html_text")])
    lower_mod.lower_url_blocks([URLBlock(url="https://x/", fetch_mode="markdown")])

    assert captured == [AttachmentIntent.TEXT, AttachmentIntent.TEXT]


def test_lowered_blocks_are_interleaved_with_other_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        lower_mod,
        "extract",
        lambda url, cfg: _ok_text_result(url, "fetched"),
    )

    blocks = [
        TextBlock(text="prefix"),
        URLBlock(url="https://example.com/a"),
        TextBlock(text="suffix"),
    ]
    out = lower_mod.lower_url_blocks(blocks)

    assert [type(b).__name__ for b in out] == ["TextBlock", "TextBlock", "TextBlock"]
    assert [b.text for b in out] == ["prefix", "fetched", "suffix"]


# ---------------------------------------------------------------------------
# cache reuse (AC: multiple personas hit L1, no double-fetch)
# ---------------------------------------------------------------------------


def test_l1_hit_short_circuits_extract(monkeypatch) -> None:
    """When the L1 already holds a markdown blob for the URL, the ladder
    serves the result from cache without invoking step 1 / step 2.

    We pre-seed the L1 and assert that ``safe_fetch`` is never called.
    """
    import althing.fetch.ladder as ladder_mod

    l1 = CacheL1()
    url = "https://example.com/cached-page"
    body = b"# Cached page\n\n" + (b"This content was fetched once already. " * 20)
    entry = CacheEntry(
        url=url,
        content_sha256="deadbeef",
        fetched_at=time.time(),
        content_type="text/markdown",
        mode="markdown",
    )
    l1.put(CacheHit(entry=entry, body=body))

    fetch_calls: list[str] = []

    def fail_fetch(*args, **kwargs):
        fetch_calls.append(args[0] if args else "")
        raise AssertionError("safe_fetch should not be called on L1 hit")

    monkeypatch.setattr(ladder_mod, "safe_fetch", fail_fetch)

    # Pretend trafilatura isn't installed so even an HTML path would fail.
    monkeypatch.setattr(ladder_mod, "_step_trafilatura", lambda *_a, **_k: None)

    # First call: should hit L1 and produce the cached body as TextBlock.
    out1 = lower_mod.lower_url_blocks([URLBlock(url=url)], l1=l1)
    assert len(out1) == 1
    assert isinstance(out1[0], TextBlock)
    assert out1[0].text == body.decode("utf-8")

    # Second call (simulating a second panelist): also serves from L1.
    out2 = lower_mod.lower_url_blocks([URLBlock(url=url)], l1=l1)
    assert isinstance(out2[0], TextBlock)
    assert out2[0].text == body.decode("utf-8")

    assert fetch_calls == []  # AC: no double-fetch


# ---------------------------------------------------------------------------
# failure modes (sy-550: fetch failure → hard error by default)
# ---------------------------------------------------------------------------


def _empty_result(url: str) -> LadderResult:
    return LadderResult(
        url=url,
        intent=AttachmentIntent.TEXT,
        text=None,
        text_mode=None,
        screenshot=None,
        screenshot_mode=None,
        final_url=url,
        redirect_chain=[],
        stale=False,
        fetched=True,
    )


def test_extraction_failure_raises_by_default(monkeypatch) -> None:
    def boom(url, cfg):
        raise ExtractionFailed(url, "extraction <200 chars and screenshot_fallback disabled")

    monkeypatch.setattr(lower_mod, "extract", boom)

    with pytest.raises(AttachmentFetchError) as ei:
        lower_mod.lower_url_blocks([URLBlock(url="https://example.com/empty")])
    assert ei.value.url == "https://example.com/empty"
    assert "extraction failed" in ei.value.reason
    # The message names the URL + reason and points at the opt-out.
    assert "https://example.com/empty" in str(ei.value)
    assert "--allow-empty-attachments" in str(ei.value)


def test_perimeter_deny_raises_by_default(monkeypatch) -> None:
    """SSRF perimeter rejections (loopback/private addresses) are a hard
    error by default — the run must not proceed with empty content."""

    def deny(url, cfg):
        raise PerimeterDeny("no safe address for 'localhost': loopback")

    monkeypatch.setattr(lower_mod, "extract", deny)

    with pytest.raises(AttachmentFetchError) as ei:
        lower_mod.lower_url_blocks([URLBlock(url="http://localhost:4321/")])
    assert "perimeter denied" in ei.value.reason
    assert "loopback" in ei.value.reason
    # SSRF documentation breadcrumb is in the message.
    assert "SSRF-blocked" in str(ei.value)


def test_unexpected_exception_raises_by_default(monkeypatch) -> None:
    def kaboom(url, cfg):
        raise RuntimeError("network melted")

    monkeypatch.setattr(lower_mod, "extract", kaboom)

    with pytest.raises(AttachmentFetchError) as ei:
        lower_mod.lower_url_blocks([URLBlock(url="https://example.com/x")])
    assert "RuntimeError" in ei.value.reason


def test_empty_result_raises_by_default(monkeypatch) -> None:
    """A ladder success with no text *and* no screenshot is empty content —
    a hard error rather than a silently-dropped block."""
    monkeypatch.setattr(lower_mod, "extract", lambda url, cfg: _empty_result(url))
    with pytest.raises(AttachmentFetchError) as ei:
        lower_mod.lower_url_blocks([URLBlock(url="https://example.com/empty")])
    assert ei.value.reason == "fetch produced no content"


# ---------------------------------------------------------------------------
# sy-550: opt-out (--allow-empty-attachments) restores placeholder behaviour
# ---------------------------------------------------------------------------


def test_allow_empty_extraction_failure_emits_placeholder(monkeypatch) -> None:
    def boom(url, cfg):
        raise ExtractionFailed(url, "extraction <200 chars and screenshot_fallback disabled")

    monkeypatch.setattr(lower_mod, "extract", boom)

    out = lower_mod.lower_url_blocks([URLBlock(url="https://example.com/empty")], allow_empty=True)
    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "https://example.com/empty" in out[0].text
    assert "unavailable" in out[0].text.lower()


def test_allow_empty_perimeter_deny_emits_placeholder(monkeypatch) -> None:
    def deny(url, cfg):
        raise PerimeterDeny("private-ip target rejected: 10.0.0.1")

    monkeypatch.setattr(lower_mod, "extract", deny)

    out = lower_mod.lower_url_blocks([URLBlock(url="https://internal.example/")], allow_empty=True)
    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "perimeter" in out[0].text.lower()


def test_allow_empty_empty_result_emits_placeholder(monkeypatch) -> None:
    monkeypatch.setattr(lower_mod, "extract", lambda url, cfg: _empty_result(url))
    out = lower_mod.lower_url_blocks([URLBlock(url="https://example.com/empty")], allow_empty=True)
    assert len(out) == 1
    assert isinstance(out[0], TextBlock)
    assert "no content" in out[0].text


# ---------------------------------------------------------------------------
# sy-550: per-attachment fetch status recording (status_sink)
# ---------------------------------------------------------------------------


def test_status_sink_records_ok_on_success(monkeypatch) -> None:
    monkeypatch.setattr(lower_mod, "extract", lambda url, cfg: _ok_text_result(url, "body"))
    sink: list[dict] = []
    lower_mod.lower_url_blocks(
        [URLBlock(url="https://example.com/a", fetch_mode="markdown")],
        status_sink=sink,
    )
    assert sink == [{"url": "https://example.com/a", "status": "ok", "mode": "markdown"}]


def test_status_sink_records_failure_with_reason(monkeypatch) -> None:
    def deny(url, cfg):
        raise PerimeterDeny("no safe address for 'localhost': loopback")

    monkeypatch.setattr(lower_mod, "extract", deny)
    sink: list[dict] = []
    # allow_empty so the call returns rather than raising; status is still recorded.
    lower_mod.lower_url_blocks(
        [URLBlock(url="http://localhost:4321/", fetch_mode="markdown")],
        allow_empty=True,
        status_sink=sink,
    )
    assert len(sink) == 1
    assert sink[0]["url"] == "http://localhost:4321/"
    assert sink[0]["status"] == "failed"
    assert "loopback" in sink[0]["reason"]


def test_status_sink_records_failure_even_when_raising(monkeypatch) -> None:
    """The status record is appended before the hard error is raised, so an
    auditing caller that catches the error still sees the failed entry."""

    def boom(url, cfg):
        raise ExtractionFailed(url, "empty")

    monkeypatch.setattr(lower_mod, "extract", boom)
    sink: list[dict] = []
    with pytest.raises(AttachmentFetchError):
        lower_mod.lower_url_blocks([URLBlock(url="https://x/")], status_sink=sink)
    assert sink and sink[0]["status"] == "failed"


# ---------------------------------------------------------------------------
# AC: URLBlock never reaches the wire
# ---------------------------------------------------------------------------


def test_lowering_removes_all_urlblocks(monkeypatch) -> None:
    """Property test: after lowering, no URLBlock survives — the wire
    serializer's URLBlock guard (anthropic.py) should never trip."""

    monkeypatch.setattr(
        lower_mod,
        "extract",
        lambda url, cfg: _ok_text_result(url),
    )

    blocks = [
        URLBlock(url="https://a/"),
        TextBlock(text="middle"),
        URLBlock(url="https://b/", fetch_mode="screenshot"),
    ]
    # screenshot URL: route through extract too — same fake.
    monkeypatch.setattr(
        lower_mod,
        "extract",
        lambda url, cfg: _ok_screenshot_result(url) if cfg.intent is AttachmentIntent.VISUAL else _ok_text_result(url),
    )
    out = lower_mod.lower_url_blocks(blocks)

    assert not any(isinstance(b, URLBlock) for b in out)
    types = [type(b).__name__ for b in out]
    assert types == ["TextBlock", "TextBlock", "ImageBlock"]
