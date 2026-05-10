"""Model capability registry and pre-flight checks (hq-vw6o).

Some Claude models do NOT support vision — notably Claude 3.5 Haiku
(``claude-3-5-haiku-20241022``). Sending image attachments to a text-only
model burns tokens silently: the provider accepts the request, the model
responds without seeing the image ("I don't see an attached image"), and
the user is billed for the round trip. The empirical signal that produced
this module: a 15-persona dogfood panel against base64 image attachments
on ``openrouter/anthropic/claude-3.5-haiku`` cost ~\\$3 and returned
"no image" from every persona.

``assert_supports_attachments`` scans a :class:`CompletionRequest` for
:class:`ImageBlock` / :class:`DocumentBlock` payloads and raises
:class:`LLMError` (``BAD_REQUEST``) when the resolved model is on the
text-only list. ``LLMClient`` calls it after alias resolution but before
the HTTP send so the failure surfaces as a clear, actionable error
instead of a silent burn.

The list is conservative: only models empirically confirmed text-only
are included. False positives would over-block valid runs.
"""

from __future__ import annotations

from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    DocumentBlock,
    ImageBlock,
)

# Substrings (lowercased) that identify a model with no vision capability.
# Match is by ``substring in model.lower()`` so a single entry catches the
# OpenRouter routing prefix (``openrouter/anthropic/claude-3.5-haiku``),
# Anthropic-native ID (``claude-3-5-haiku-20241022``), and short alias
# forms (``claude-3-5-haiku``, ``claude-3.5-haiku``).
#
# Adding to this list:
# - Confirm via official provider docs that the model lacks vision.
# - Prefer specific substrings (``claude-3-5-haiku``) over short tokens
#   (``haiku``) — short tokens collide with vision-capable variants such
#   as Haiku 4.5 (``claude-haiku-4-5-20251001``), which DOES support
#   vision.
_NO_VISION_PATTERNS: tuple[str, ...] = (
    "claude-3-5-haiku",
    "claude-3.5-haiku",
)


def model_supports_vision(model: str) -> bool:
    """Return ``False`` iff *model* matches a known text-only pattern."""
    m = model.lower()
    return not any(p in m for p in _NO_VISION_PATTERNS)


def _request_has_visual_attachments(request: CompletionRequest) -> bool:
    for msg in request.messages:
        for block in msg.content:
            if isinstance(block, (ImageBlock, DocumentBlock)):
                return True
    return False


def assert_supports_attachments(request: CompletionRequest) -> None:
    """Raise ``LLMError(BAD_REQUEST)`` when *request* carries image or
    document attachments and ``request.model`` is on the text-only list.

    Invoked from :meth:`LLMClient.send` / :meth:`LLMClient.stream` after
    alias resolution and before any HTTP is issued, so a misrouted
    multimodal call fails fast with a clear message instead of silently
    burning tokens (hq-vw6o).
    """
    if not _request_has_visual_attachments(request):
        return
    if model_supports_vision(request.model):
        return
    raise LLMError(
        f"model {request.model!r} does not support image or document "
        "attachments. Claude 3.5 Haiku is text-only — the upstream API "
        "still accepts the request and returns a no-image response, "
        "burning tokens silently. Switch to a vision-capable model "
        "(haiku → claude-haiku-4-5-20251001, sonnet, opus, or "
        "openrouter/anthropic/claude-haiku-4.5) or remove the "
        "attachments from the run.",
        LLMErrorCategory.BAD_REQUEST,
    )
