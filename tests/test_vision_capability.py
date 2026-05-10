"""Pre-flight vision-capability check (hq-vw6o).

Locks in the regression that v1.0.2 left open: a panel run with image
attachments against ``openrouter/anthropic/claude-3.5-haiku`` (or any
``claude-3-5-haiku-*`` route) silently burned ~\\$3 because the upstream
text-only model accepted the request and replied "I don't see an attached
image." These tests verify that:

* Visual attachments + a known text-only model raise ``LLMError``
  (BAD_REQUEST) before the HTTP call leaves the client.
* Text-only requests against the same models pass through (the model is
  text-only, not broken — it should still serve text panels).
* Vision-capable models (Sonnet, Haiku 4.5) keep working with images.
"""

from __future__ import annotations

import pytest

from synth_panel.llm.capabilities import (
    assert_supports_attachments,
    model_supports_vision,
)
from synth_panel.llm.errors import LLMError, LLMErrorCategory
from synth_panel.llm.models import (
    CompletionRequest,
    DocumentBlock,
    ImageBlock,
    InlineSource,
    InputMessage,
    TextBlock,
)


def _request(model: str, blocks: list) -> CompletionRequest:
    return CompletionRequest(
        model=model,
        max_tokens=64,
        messages=[InputMessage(role="user", content=blocks)],
    )


def _image_block() -> ImageBlock:
    return ImageBlock(source=InlineSource(data="abcd"), media_type="image/png")


def _document_block() -> DocumentBlock:
    return DocumentBlock(
        source=InlineSource(data="JVBERi0="),
        media_type="application/pdf",
    )


# ---------------------------------------------------------------------------
# model_supports_vision — pattern matcher
# ---------------------------------------------------------------------------


class TestModelSupportsVision:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-3-5-haiku-20241022",
            "claude-3-5-haiku",
            "anthropic/claude-3.5-haiku",
            "anthropic/claude-3-5-haiku",
            "openrouter/anthropic/claude-3.5-haiku",
            "openrouter/anthropic/claude-3-5-haiku-20241022",
            "Claude-3-5-Haiku-20241022",  # case-insensitive
        ],
    )
    def test_known_text_only_models_return_false(self, model: str) -> None:
        assert model_supports_vision(model) is False

    @pytest.mark.parametrize(
        "model",
        [
            "claude-haiku-4-5-20251001",  # Haiku 4.5 supports vision
            "openrouter/anthropic/claude-haiku-4.5",
            "claude-sonnet-4-6",
            "claude-opus-4-6",
            "claude-3-haiku-20240307",  # Haiku 3 supports vision
            "gpt-4o",
            "gemini-2.5-flash",
            "grok-3",
        ],
    )
    def test_vision_capable_models_return_true(self, model: str) -> None:
        assert model_supports_vision(model) is True


# ---------------------------------------------------------------------------
# assert_supports_attachments — pre-flight gate
# ---------------------------------------------------------------------------


class TestAssertSupportsAttachments:
    def test_image_on_haiku_3_5_raises_bad_request(self) -> None:
        req = _request("claude-3-5-haiku-20241022", [_image_block()])
        with pytest.raises(LLMError) as exc_info:
            assert_supports_attachments(req)
        assert exc_info.value.category is LLMErrorCategory.BAD_REQUEST
        msg = str(exc_info.value)
        assert "claude-3-5-haiku" in msg.lower()
        # Guidance to switch must be in the message — the bead is
        # explicitly about preventing future $3 burns, so the recovery
        # path must be obvious from the error alone.
        assert "haiku-4-5" in msg or "haiku-4.5" in msg
        assert "sonnet" in msg.lower()

    def test_image_on_openrouter_haiku_3_5_alias_raises(self) -> None:
        req = _request(
            "openrouter/anthropic/claude-3.5-haiku",
            [TextBlock(text="describe"), _image_block()],
        )
        with pytest.raises(LLMError) as exc_info:
            assert_supports_attachments(req)
        assert exc_info.value.category is LLMErrorCategory.BAD_REQUEST

    def test_document_on_haiku_3_5_raises(self) -> None:
        # PDF attachments share the same capability gate as images:
        # Claude 3.5 Haiku has no document-vision either.
        req = _request("claude-3-5-haiku-20241022", [_document_block()])
        with pytest.raises(LLMError):
            assert_supports_attachments(req)

    def test_text_only_request_on_haiku_3_5_passes(self) -> None:
        # The model is text-only, not broken — text panels must still run.
        req = _request("claude-3-5-haiku-20241022", [TextBlock(text="hi")])
        assert_supports_attachments(req)  # does not raise

    def test_image_on_sonnet_passes(self) -> None:
        req = _request("claude-sonnet-4-6", [_image_block()])
        assert_supports_attachments(req)

    def test_image_on_haiku_4_5_passes(self) -> None:
        # Critical false-positive guard: a too-broad "haiku" pattern
        # would block the recommended fix path.
        req = _request("claude-haiku-4-5-20251001", [_image_block()])
        assert_supports_attachments(req)

    def test_image_on_openrouter_haiku_4_5_passes(self) -> None:
        req = _request(
            "openrouter/anthropic/claude-haiku-4.5", [_image_block()]
        )
        assert_supports_attachments(req)

    def test_empty_messages_passes(self) -> None:
        req = CompletionRequest(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            messages=[],
        )
        assert_supports_attachments(req)
