"""Content-type ladder for URL attachments (hq-hqlp §1).

Drives extraction based on a question's ``attachment_intent``:

* ``"text"``    → markdown negotiation, then trafilatura on HTML.
* ``"visual"``  → Playwright screenshot only (skips 1 + 2).
* ``"both"``    → text path *and* screenshot, emitted side-by-side.

Steps:

1. **markdown negotiation** — ``GET`` with ``Accept: text/markdown,
   text/html;q=0.9``. If the server responds with ``text/markdown``
   (Cloudflare's Markdown-for-Agents and similar), return it directly.
2. **trafilatura** — on returned HTML, run trafilatura with
   ``output_format="markdown"``. trafilatura already cascades through
   readability + justext internally.
3. **Playwright screenshot** — invoked only when the intent demands
   visuals, or when text extraction yielded <200 chars and the
   per-question ``screenshot_fallback`` flag is set.

When extraction yields <200 chars and screenshot fallback is
unavailable, the ladder raises ``ExtractionFailed`` so the orchestrator
can apply the per-attachment ``on_failure`` policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from synth_panel.fetch.cache import CacheL1, UrlCache
from synth_panel.fetch.perimeter import (
    DEFAULT_TIMEOUT,
    FetchResult,
    PerimeterDeny,
    safe_fetch,
)

# Minimum extracted-text length below which we treat the extraction as
# a soft failure. Tuned in hq-hqlp §1 against trafilatura sweep results.
MIN_EXTRACTION_CHARS = 200


class AttachmentIntent(str, Enum):
    """Per-question attachment intent (matches the YAML field).

    String-valued so it round-trips through YAML / JSON without a
    custom encoder.
    """

    TEXT = "text"
    VISUAL = "visual"
    BOTH = "both"

    @classmethod
    def coerce(cls, value: str | AttachmentIntent | None) -> AttachmentIntent:
        if value is None:
            return cls.TEXT
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown attachment_intent {value!r}; expected one of {[m.value for m in cls]}") from exc


class OnFailure(str, Enum):
    """Per-attachment failure policy (hq-hqlp §5)."""

    ABORT = "abort"
    SKIP_QUESTION = "skip_question"
    PLACEHOLDER = "placeholder"

    @classmethod
    def coerce(cls, value: str | OnFailure | None) -> OnFailure:
        if value is None:
            return cls.SKIP_QUESTION
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"unknown on_failure {value!r}; expected one of {[m.value for m in cls]}") from exc


class ExtractionFailed(Exception):
    """Raised when the ladder cannot produce useful content for the URL.

    Carries the original URL and the reason so the orchestrator can
    surface a structured ``skipped_reason`` in the result envelope.
    """

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{url}: {reason}")
        self.url = url
        self.reason = reason


@dataclass
class LadderConfig:
    """Per-attachment knobs for the ladder.

    ``screenshot_fallback`` defaults to False — visual fetches are
    opt-in to keep the unbounded-fetch cost cliff from hq-bqrw §6
    closed by default.
    """

    intent: AttachmentIntent = AttachmentIntent.TEXT
    on_failure: OnFailure = OnFailure.SKIP_QUESTION
    screenshot_fallback: bool = False
    pin: bool = False
    cache: UrlCache | None = None
    l1: CacheL1 | None = None
    user_agent: str = "synthpanel-fetch/1.0 (+https://synthpanel.dev)"
    # Optional: override timeouts / size caps for this attachment.
    max_bytes: int | None = None
    screenshot_timeout_ms: int = 15_000


@dataclass
class LadderResult:
    """Outcome of a successful ladder run."""

    url: str
    intent: AttachmentIntent
    text: str | None  # extracted markdown / plain text (steps 1 + 2)
    text_mode: str | None  # "markdown" | "html-extracted" | None
    screenshot: bytes | None  # PNG bytes from step 3, if produced
    screenshot_mode: str | None  # "screenshot" | None
    final_url: str
    redirect_chain: list[str] = field(default_factory=list)
    stale: bool = False  # cache returned past TTL because pinned
    fetched: bool = True  # False when fully served from cache


# ---------------------------------------------------------------------------
# step 1 — markdown negotiation
# ---------------------------------------------------------------------------


def _step_markdown(url: str, cfg: LadderConfig) -> tuple[str, FetchResult] | None:
    """Try ``Accept: text/markdown``. Returns (markdown, fetch) on hit, else ``None``."""
    accept = "text/markdown, text/html;q=0.9, */*;q=0.5"
    fetch = safe_fetch(
        url,
        accept=accept,
        max_bytes=cfg.max_bytes if cfg.max_bytes is not None else 8 * 1024 * 1024,
        timeout=DEFAULT_TIMEOUT,
        user_agent=cfg.user_agent,
    )
    if fetch.content_type == "text/markdown":
        try:
            return fetch.body.decode("utf-8", errors="replace"), fetch
        except UnicodeDecodeError:
            return None
    # HTML — caller will hand the bytes to step 2.
    return None, fetch  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# step 2 — trafilatura
# ---------------------------------------------------------------------------


def _step_trafilatura(html_bytes: bytes, base_url: str) -> str | None:
    """Run trafilatura on ``html_bytes``, returning markdown or ``None``."""
    try:
        import trafilatura  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        html = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None

    try:
        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            url=base_url,
        )
    except Exception:
        return None

    if not extracted:
        return None
    return extracted.strip() or None


# ---------------------------------------------------------------------------
# step 3 — Playwright screenshot
# ---------------------------------------------------------------------------


def _step_screenshot(url: str, cfg: LadderConfig) -> bytes | None:
    """Capture a full-page PNG via Playwright. ``None`` if unavailable.

    Playwright is gated behind the ``synthpanel[visual]`` extra; if it
    isn't installed the function returns ``None`` and the caller
    decides how to surface the gap (extraction-failed vs. text-only).
    """
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                ctx = browser.new_context(user_agent=cfg.user_agent)
                page = ctx.new_page()
                page.goto(url, timeout=cfg.screenshot_timeout_ms, wait_until="domcontentloaded")
                png = page.screenshot(full_page=True, type="png")
                ctx.close()
                return png
            finally:
                browser.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------


def _try_l1(cfg: LadderConfig, url: str, mode: str) -> bytes | None:
    if cfg.l1 is None:
        return None
    hit = cfg.l1.get(url, mode)
    return hit.body if hit is not None else None


def _try_disk(cfg: LadderConfig, url: str, mode: str) -> tuple[bytes, bool] | None:
    """Return (body, stale) from disk cache, else ``None``."""
    if cfg.cache is None:
        return None
    hit = cfg.cache.lookup(url, mode)
    if hit is None:
        return None
    if cfg.l1 is not None:
        cfg.l1.put(hit)
    return hit.body, hit.entry.stale


def _store(
    cfg: LadderConfig,
    url: str,
    body: bytes,
    *,
    content_type: str,
    mode: str,
) -> None:
    if cfg.cache is None:
        return
    entry = cfg.cache.store(url, body, content_type=content_type, mode=mode, pinned=cfg.pin)
    if cfg.l1 is not None:
        from synth_panel.fetch.cache import CacheHit

        cfg.l1.put(CacheHit(entry=entry, body=body))


def extract(url: str, cfg: LadderConfig | None = None) -> LadderResult:
    """Run the full content-type ladder for ``url``.

    The returned :class:`LadderResult` carries either a ``text``
    payload, a ``screenshot`` payload, or both, depending on intent.

    Raises
    ------
    PerimeterDeny
        Re-raised from :func:`safe_fetch` for security violations.
    ExtractionFailed
        Step 2 yielded <200 chars and no screenshot fallback was
        permitted.
    """
    cfg = cfg or LadderConfig()
    intent = cfg.intent

    text: str | None = None
    text_mode: str | None = None
    screenshot: bytes | None = None
    screenshot_mode: str | None = None
    final_url = url
    redirect_chain: list[str] = []
    stale = False
    fetched = False

    # ---------- text path (steps 1 + 2) ----------
    if intent in (AttachmentIntent.TEXT, AttachmentIntent.BOTH):
        # cache: try markdown first, then html-extracted.
        for mode_try in ("markdown", "html-extracted"):
            l1_body = _try_l1(cfg, url, mode_try)
            if l1_body is not None:
                text = l1_body.decode("utf-8", errors="replace")
                text_mode = mode_try
                break
            disk = _try_disk(cfg, url, mode_try)
            if disk is not None:
                body, was_stale = disk
                text = body.decode("utf-8", errors="replace")
                text_mode = mode_try
                stale = stale or was_stale
                break

        if text is None:
            # Live fetch — step 1.
            try:
                outcome = _step_markdown(url, cfg)
            except (PerimeterDeny, Exception) as exc:
                # Bubble PerimeterDeny up; treat anything else as text-step failure.
                if isinstance(exc, PerimeterDeny):
                    raise
                outcome = None

            fetched = True
            if outcome is not None and isinstance(outcome[0], str):
                text = outcome[0]
                text_mode = "markdown"
                fr = outcome[1]
                final_url = fr.url
                redirect_chain = list(fr.redirect_chain)
                _store(
                    cfg,
                    url,
                    text.encode("utf-8"),
                    content_type=fr.content_type,
                    mode="markdown",
                )
            elif outcome is not None:
                # Step 1 returned HTML bytes — feed step 2.
                _, fr = outcome
                final_url = fr.url
                redirect_chain = list(fr.redirect_chain)
                if fr.content_type == "text/html":
                    extracted = _step_trafilatura(fr.body, fr.url)
                    if extracted is not None:
                        text = extracted
                        text_mode = "html-extracted"
                        _store(
                            cfg,
                            url,
                            extracted.encode("utf-8"),
                            content_type="text/markdown",
                            mode="html-extracted",
                        )
                elif fr.content_type == "text/plain":
                    text = fr.body.decode("utf-8", errors="replace")
                    text_mode = "html-extracted"
                    _store(
                        cfg,
                        url,
                        text.encode("utf-8"),
                        content_type="text/plain",
                        mode="html-extracted",
                    )

        # Apply min-length gate — step 3 fallback only if asked for.
        if (
            intent == AttachmentIntent.TEXT
            and (text is None or len(text) < MIN_EXTRACTION_CHARS)
            and not cfg.screenshot_fallback
        ):
            raise ExtractionFailed(
                url,
                "extraction <200 chars and screenshot_fallback disabled",
            )

    # ---------- visual path (step 3) ----------
    needs_screenshot = intent in (AttachmentIntent.VISUAL, AttachmentIntent.BOTH) or (
        intent == AttachmentIntent.TEXT
        and cfg.screenshot_fallback
        and (text is None or len(text) < MIN_EXTRACTION_CHARS)
    )
    if needs_screenshot:
        l1_body = _try_l1(cfg, url, "screenshot")
        if l1_body is not None:
            screenshot = l1_body
            screenshot_mode = "screenshot"
        else:
            disk = _try_disk(cfg, url, "screenshot")
            if disk is not None:
                screenshot = disk[0]
                screenshot_mode = "screenshot"
                stale = stale or disk[1]
            else:
                fetched = True
                # Re-validate the URL's perimeter even though the
                # screenshot path bypasses safe_fetch — Playwright will
                # do its own DNS resolution. We sniff the host first
                # so we deny obviously-private targets up front.
                from urllib.parse import urlparse

                from synth_panel.fetch.perimeter import safe_resolve

                host = urlparse(url).hostname
                if host:
                    safe_resolve(host)  # raises PerimeterDeny on failure
                png = _step_screenshot(url, cfg)
                if png is not None:
                    screenshot = png
                    screenshot_mode = "screenshot"
                    _store(cfg, url, png, content_type="image/png", mode="screenshot")

    # ---------- final extraction-failed gate ----------
    if intent == AttachmentIntent.VISUAL and screenshot is None:
        raise ExtractionFailed(url, "screenshot unavailable (playwright not installed?)")
    if intent == AttachmentIntent.BOTH and text is None and screenshot is None:
        raise ExtractionFailed(url, "text extraction and screenshot both failed")

    return LadderResult(
        url=url,
        intent=intent,
        text=text,
        text_mode=text_mode,
        screenshot=screenshot,
        screenshot_mode=screenshot_mode,
        final_url=final_url,
        redirect_chain=redirect_chain,
        stale=stale,
        fetched=fetched,
    )


__all__ = [
    "MIN_EXTRACTION_CHARS",
    "AttachmentIntent",
    "ExtractionFailed",
    "LadderConfig",
    "LadderResult",
    "OnFailure",
    "extract",
]
