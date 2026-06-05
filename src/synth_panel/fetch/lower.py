"""Frame-stage URLBlock lowering (hq-8iz3, v1.0.2 G5).

Bridges :class:`synth_panel.llm.models.URLBlock` (a pre-fetch stub
emitted by ``_attachment_to_block`` for ``{type: url}`` attachments)
to the concrete TextBlock / ImageBlock pair the wire serializer
expects.

Runs at the orchestrator's frame stage — after
:func:`synth_panel.prompts.build_question_blocks` and before the
turn input reaches the LLM client. Multiple personas in the same
panel share a :class:`CacheL1`, so a single URL is fetched at most
once per run regardless of how many panelists reference it.

The fetcher (perimeter + ladder + on-disk cache) is implemented in
:mod:`synth_panel.fetch`. This module is a thin adapter: map
``URLBlock.fetch_mode`` → :class:`AttachmentIntent`, dispatch the
ladder, splice the result back into the block list, and — by default —
turn fetch errors into a **hard error** so a run never silently
proceeds with empty attachment content (sy-550).

Historically (G5) a failed fetch was logged as a WARNING and replaced
with a placeholder TextBlock; the persona then answered blind and the
run reported a 0% failure rate even though the entire premise ("react
to this page") never reached the model. That silent degrade is the
bug fixed here: an attachment that yields no usable content is now an
:class:`AttachmentFetchError` by default, naming the URL and reason
(perimeter-denied / extraction-failed / timeout / empty). Callers who
genuinely want best-effort behaviour pass ``allow_empty=True`` (wired
to the ``--allow-empty-attachments`` CLI flag), which restores the
placeholder path.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterable

from synth_panel.fetch.cache import CacheL1, UrlCache
from synth_panel.fetch.ladder import (
    AttachmentIntent,
    ExtractionFailed,
    LadderConfig,
    LadderResult,
    extract,
)
from synth_panel.fetch.perimeter import PerimeterDeny
from synth_panel.llm.models import (
    ContentBlock,
    ImageBlock,
    InlineSource,
    TextBlock,
    URLBlock,
)

logger = logging.getLogger(__name__)


class AttachmentFetchError(RuntimeError):
    """A referenced ``type: url`` attachment yielded no usable content.

    Raised by :func:`lower_url_blocks` (unless ``allow_empty=True``) when a
    URL attachment cannot be fetched or extracted — perimeter denial
    (SSRF / loopback / private address), extraction failure, transport
    error, or an empty (no text + no screenshot) result. Carries the
    offending ``url`` and a human-readable ``reason`` so the orchestrator
    / CLI can fail the run with an actionable message instead of letting
    the persona answer blind (sy-550).
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        self.reason = reason
        super().__init__(
            f"URL attachment could not be fetched: {url} — {reason}. "
            "The persona would answer without the attachment content. "
            "Fix the URL/source, or pass --allow-empty-attachments to proceed "
            "best-effort. Note: loopback/private addresses (localhost, 127.0.0.1, "
            "10.x, 192.168.x, etc.) are SSRF-blocked — use an inline "
            "`type: html` / `type: document` attachment for local content."
        )


_FETCH_MODE_TO_INTENT: dict[str, AttachmentIntent] = {
    "auto": AttachmentIntent.TEXT,
    "html_text": AttachmentIntent.TEXT,
    "markdown": AttachmentIntent.TEXT,
    "screenshot": AttachmentIntent.VISUAL,
}


def _intent_for(block: URLBlock) -> AttachmentIntent:
    return _FETCH_MODE_TO_INTENT.get(block.fetch_mode, AttachmentIntent.TEXT)


def _result_to_blocks(result: LadderResult) -> list[ContentBlock]:
    """Splice a :class:`LadderResult` into one or more concrete blocks.

    Order matches Anthropic's image-then-text canonical ordering for
    multimodal turns: image first, then any extracted text.
    """
    out: list[ContentBlock] = []
    if result.screenshot is not None:
        data = base64.b64encode(result.screenshot).decode("ascii")
        out.append(ImageBlock(source=InlineSource(data=data), media_type="image/png"))
    if result.text is not None:
        out.append(TextBlock(text=result.text))
    return out


def _placeholder(url: str, reason: str) -> TextBlock:
    """Build a placeholder block for the ``allow_empty=True`` best-effort path.

    Only reached when the caller explicitly opted out of the hard-fail
    default (``--allow-empty-attachments``). The persona at least sees that
    the URL was unavailable rather than silently receiving nothing.
    """
    return TextBlock(text=f"[URL attachment unavailable: {url} — {reason}]")


def lower_url_blocks(
    blocks: Iterable[ContentBlock],
    *,
    l1: CacheL1 | None = None,
    cache: UrlCache | None = None,
    allow_empty: bool = False,
    status_sink: list[dict[str, str]] | None = None,
) -> list[ContentBlock]:
    """Return ``blocks`` with every :class:`URLBlock` replaced by its fetched content.

    Non-URLBlock entries pass through unchanged. URLBlocks are
    resolved via the hq-gmju content ladder; the per-run ``l1`` and
    cross-run ``cache`` are consulted before any network call so 15
    panelists referencing the same URL share a single fetch.

    By default (``allow_empty=False``) a URL that yields no usable content
    — perimeter denial, extraction failure, transport error, or an empty
    result — raises :class:`AttachmentFetchError` naming the URL + reason,
    so the run fails loudly instead of proceeding with empty attachment
    content (sy-550). Pass ``allow_empty=True`` to restore the legacy
    best-effort behaviour, where such failures become placeholder
    TextBlocks instead.

    When ``status_sink`` is provided, a per-URL status record
    (``{"url", "status", "reason"?, "mode"}``) is appended for every
    URLBlock processed, so the caller can persist fetch outcomes in the
    saved result for auditability. ``status`` is ``"ok"`` on success and
    ``"failed"`` on any failure (whether it raised or was tolerated).
    """
    out: list[ContentBlock] = []

    def _record(url: str, status: str, mode: str, reason: str | None = None) -> None:
        if status_sink is None:
            return
        rec: dict[str, str] = {"url": url, "status": status, "mode": mode}
        if reason is not None:
            rec["reason"] = reason
        status_sink.append(rec)

    for block in blocks:
        if not isinstance(block, URLBlock):
            out.append(block)
            continue

        intent = _intent_for(block)
        cfg = LadderConfig(intent=intent, l1=l1, cache=cache)
        try:
            result = extract(block.url, cfg)
        except PerimeterDeny as exc:
            reason = f"perimeter denied: {exc}"
        except ExtractionFailed as exc:
            reason = f"extraction failed: {exc.reason}"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        else:
            lowered = _result_to_blocks(result)
            if lowered:
                _record(block.url, "ok", block.fetch_mode)
                out.extend(lowered)
                continue
            reason = "fetch produced no content"

        # Failure path (any of the branches above set ``reason``).
        _record(block.url, "failed", block.fetch_mode, reason)
        if allow_empty:
            logger.warning(
                "URLBlock lowering: %s for %s — proceeding with placeholder (--allow-empty-attachments)",
                reason,
                block.url,
            )
            out.append(_placeholder(block.url, reason))
            continue
        logger.error("URLBlock lowering: %s for %s", reason, block.url)
        raise AttachmentFetchError(block.url, reason)

    return out


__all__ = ["AttachmentFetchError", "lower_url_blocks"]
