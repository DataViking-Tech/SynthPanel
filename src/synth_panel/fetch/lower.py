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
ladder, splice the result back into the block list, and turn fetch
errors into a placeholder TextBlock so the persona at least sees
that the URL was unavailable.
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
    """Build a placeholder block when fetch / extraction fails.

    G5 hotfix scope honors the bead's default ``on_failure`` policy
    (skip_question is the per-attachment default) by emitting a
    placeholder so the run doesn't abort. The orchestrator can choose
    to raise instead in a future revision once explicit per-attachment
    failure policy plumbing lands.
    """
    return TextBlock(text=f"[URL attachment unavailable: {url} — {reason}]")


def lower_url_blocks(
    blocks: Iterable[ContentBlock],
    *,
    l1: CacheL1 | None = None,
    cache: UrlCache | None = None,
) -> list[ContentBlock]:
    """Return ``blocks`` with every :class:`URLBlock` replaced by its fetched content.

    Non-URLBlock entries pass through unchanged. URLBlocks are
    resolved via the hq-gmju content ladder; the per-run ``l1`` and
    cross-run ``cache`` are consulted before any network call so 15
    panelists referencing the same URL share a single fetch.

    Failures (perimeter denial, extraction <200 chars, etc.) are
    surfaced as placeholder TextBlocks rather than raising — see
    :func:`_placeholder`.
    """
    out: list[ContentBlock] = []
    for block in blocks:
        if not isinstance(block, URLBlock):
            out.append(block)
            continue

        intent = _intent_for(block)
        cfg = LadderConfig(intent=intent, l1=l1, cache=cache)
        try:
            result = extract(block.url, cfg)
        except PerimeterDeny as exc:
            logger.warning(
                "URLBlock lowering: perimeter denied %s: %s",
                block.url,
                exc,
            )
            out.append(_placeholder(block.url, f"perimeter denied: {exc}"))
            continue
        except ExtractionFailed as exc:
            logger.warning(
                "URLBlock lowering: extraction failed for %s: %s",
                block.url,
                exc.reason,
            )
            out.append(_placeholder(block.url, exc.reason))
            continue
        except Exception as exc:
            logger.warning(
                "URLBlock lowering: unexpected error for %s: %s: %s",
                block.url,
                type(exc).__name__,
                exc,
            )
            out.append(_placeholder(block.url, f"{type(exc).__name__}: {exc}"))
            continue

        lowered = _result_to_blocks(result)
        if not lowered:
            out.append(_placeholder(block.url, "fetch produced no content"))
        else:
            out.extend(lowered)
    return out


__all__ = ["lower_url_blocks"]
