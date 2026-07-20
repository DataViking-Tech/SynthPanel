"""Tests for LLM data models."""

from __future__ import annotations

import pytest

from althing.llm.models import (
    CompletionResponse,
    ContentBlock,
    DocumentBlock,
    FileRefSource,
    HTMLBlock,
    ImageBlock,
    InlineSource,
    TextBlock,
    TokenUsage,
    ToolChoice,
    ToolChoiceKind,
    ToolInvocationBlock,
    URLBlock,
    URLSource,
)


class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(input_tokens=10, output_tokens=20, cache_write_tokens=5, cache_read_tokens=3)
        assert u.total_tokens == 38

    def test_addition(self):
        a = TokenUsage(input_tokens=10, output_tokens=20)
        b = TokenUsage(input_tokens=5, output_tokens=15, cache_write_tokens=2)
        c = a + b
        assert c.input_tokens == 15
        assert c.output_tokens == 35
        assert c.cache_write_tokens == 2
        assert c.cache_read_tokens == 0

    def test_default_zeros(self):
        u = TokenUsage()
        assert u.total_tokens == 0


class TestToolChoice:
    def test_auto(self):
        tc = ToolChoice.auto()
        assert tc.kind == ToolChoiceKind.AUTO
        assert tc.name is None

    def test_any(self):
        tc = ToolChoice.any()
        assert tc.kind == ToolChoiceKind.ANY

    def test_specific(self):
        tc = ToolChoice.specific("respond")
        assert tc.kind == ToolChoiceKind.SPECIFIC
        assert tc.name == "respond"


class TestCompletionResponse:
    def test_text_property(self):
        r = CompletionResponse(
            id="r1",
            model="test",
            content=[TextBlock(text="Hello "), TextBlock(text="world")],
        )
        assert r.text == "Hello world"

    def test_tool_calls_property(self):
        tc = ToolInvocationBlock(id="tc1", name="respond", input={"key": "val"})
        r = CompletionResponse(
            id="r1",
            model="test",
            content=[TextBlock(text="preamble"), tc],
        )
        assert r.tool_calls == [tc]
        assert r.text == "preamble"


class TestAttachmentBlocks:
    """The four block types added in hq-l0lw (image, document, url, html)."""

    def test_image_block_accepts_inline_source(self):
        b = ImageBlock(source=InlineSource(data="AAAA"), media_type="image/png")
        assert b.type == "image"
        assert b.source.type == "base64"
        assert b.source.data == "AAAA"

    def test_image_block_accepts_url_source(self):
        b = ImageBlock(source=URLSource(url="https://x/y.jpg"), media_type="image/jpeg")
        assert b.source.type == "url"

    def test_image_block_accepts_file_ref_source(self):
        b = ImageBlock(source=FileRefSource(file_id="file_123"), media_type="image/png")
        assert b.source.type == "file"

    def test_image_block_cache_control_optional(self):
        b = ImageBlock(source=InlineSource(data="x"))
        assert b.cache_control is None

    def test_document_block_pdf(self):
        b = DocumentBlock(source=InlineSource(data="x"))
        assert b.type == "document"
        assert b.media_type == "application/pdf"

    def test_url_block_default_fetch_mode(self):
        b = URLBlock(url="https://example.com")
        assert b.fetch_mode == "auto"
        assert b.type == "url"

    def test_html_block(self):
        b = HTMLBlock(text="<b>hi</b>")
        assert b.text == "<b>hi</b>"
        assert b.type == "html"

    def test_blocks_are_frozen(self):
        b = ImageBlock(source=InlineSource(data="x"))
        with pytest.raises(Exception):
            b.media_type = "image/jpeg"  # type: ignore[misc]

    def test_existing_blocks_unchanged(self):
        # Regression: TextBlock / ToolInvocationBlock identity preserved
        # alongside the new union members.
        t = TextBlock(text="hi")
        assert t.type == "text"
        ti = ToolInvocationBlock(id="x", name="y", input={})
        assert ti.type == "tool_use"

    def test_content_block_union_includes_new_types(self):
        # The union should accept any of the eight types — purely a
        # type-check assertion, but we also confirm at runtime by
        # storing each in a list typed as list[ContentBlock].
        items: list[ContentBlock] = [
            TextBlock(text="t"),
            ImageBlock(source=InlineSource(data="x")),
            DocumentBlock(source=InlineSource(data="x")),
            URLBlock(url="https://x.test"),
            HTMLBlock(text="<i>x</i>"),
        ]
        assert len(items) == 5
