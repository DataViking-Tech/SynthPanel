"""v1.0.1 hotfix coverage for the multimodal-attachments wiring gaps.

These tests lock in two regressions surfaced during the 2026-05-09 dogfood
panel against dataviking.tech preview tiles:

* **G1** — :func:`synth_panel.llm.providers._openai_format._content_to_openai`
  used to silently drop ``ImageBlock`` / ``DocumentBlock`` / ``HTMLBlock``,
  emitting only the question text. Every persona on the OpenRouter /
  OpenAI-compat path responded "I don't see an attached image" even when the
  orchestrator had emitted the multimodal blocks correctly.
* **G3** — bank-ref strings (``question.attachments = ["hero_creative_v3"]``)
  were filtered out at ``orchestrator.py:879-883`` because the dict-only
  filter dropped strings before resolution. The bank-ref pattern is the
  *canonical* per the hq-xzsm data-model design.

A regression on either bug means panels appear to run successfully but
return persona responses that ignore the attachments — silent failure with
no signal in tokens, latency, or warnings.
"""
from __future__ import annotations

import pytest

from synth_panel.llm.models import (
    DocumentBlock,
    HTMLBlock,
    ImageBlock,
    InlineSource,
    TextBlock,
    URLSource,
)
from synth_panel.llm.providers._openai_format import _content_to_openai
from synth_panel.orchestrator import _resolve_question_attachment_refs


# ---------------------------------------------------------------------------
# G1 — _openai_format multimodal serialisation
# ---------------------------------------------------------------------------


class TestOpenAIFormatImageBlock:
    def test_image_block_with_inline_base64_emits_data_uri(self):
        block = ImageBlock(
            source=InlineSource(data="abcd1234"),
            media_type="image/png",
        )
        out = _content_to_openai([TextBlock(text="caption"), block])
        assert isinstance(out, list)
        assert out[0] == {"type": "text", "text": "caption"}
        assert out[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,abcd1234"},
        }

    def test_image_block_with_url_source_emits_url(self):
        block = ImageBlock(
            source=URLSource(url="https://example.com/foo.png"),
            media_type="image/png",
        )
        out = _content_to_openai([block, TextBlock(text="describe it")])
        assert out[0] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/foo.png"},
        }

    def test_text_only_still_returns_string(self):
        # Single-text-block fast path must remain intact for back-compat.
        out = _content_to_openai([TextBlock(text="just text")])
        assert out == "just text"


class TestOpenAIFormatDocumentBlock:
    def test_document_block_inline_base64_emits_file_payload(self):
        block = DocumentBlock(
            source=InlineSource(data="JVBERi0xLj=="),
            media_type="application/pdf",
        )
        out = _content_to_openai([block, TextBlock(text="summarise")])
        assert isinstance(out, list)
        assert out[0]["type"] == "file"
        assert "application/pdf" in out[0]["file"]["file_data"]


class TestOpenAIFormatHTMLBlock:
    def test_html_block_lowers_to_text(self):
        block = HTMLBlock(text="<p>hello</p>")
        out = _content_to_openai([block, TextBlock(text="and now this")])
        assert isinstance(out, list)
        assert out[0] == {"type": "text", "text": "<p>hello</p>"}


class TestOpenAIFormatRegression:
    def test_imageblock_must_not_silently_drop(self):
        # Pre-fix behaviour: image blocks fell through, producing parts == [].
        # Lock in that we now emit at least one part for the image.
        block = ImageBlock(source=InlineSource(data="x"), media_type="image/jpeg")
        out = _content_to_openai([block])
        assert isinstance(out, list)
        assert any(p.get("type") == "image_url" for p in out), (
            "ImageBlock dropped — _content_to_openai is back to the pre-G1 silent-drop"
        )


# ---------------------------------------------------------------------------
# G3 — bank-ref resolution
# ---------------------------------------------------------------------------


class TestResolveBankRefs:
    def test_string_ref_resolves_to_inline_dict(self):
        bank = {
            "hero1": {"type": "image", "media_type": "image/png",
                      "source": {"type": "base64", "data": "AAA"}},
        }
        questions = [{"text": "react", "attachments": ["hero1"]}]
        out = _resolve_question_attachment_refs(questions, bank)
        assert len(out) == 1
        assert out[0]["attachments"] == [
            {"type": "image", "media_type": "image/png",
             "source": {"type": "base64", "data": "AAA"}}
        ]

    def test_dict_ref_passes_through_unchanged(self):
        bank = {"hero1": {"type": "image", "media_type": "image/png",
                          "source": {"type": "url", "url": "https://x.test/img"}}}
        inline = {"type": "html", "text": "<p>inline</p>"}
        questions = [{"text": "mix", "attachments": ["hero1", inline]}]
        out = _resolve_question_attachment_refs(questions, bank)
        atts = out[0]["attachments"]
        assert atts[0]["type"] == "image"
        assert atts[1] is inline or atts[1] == inline

    def test_unresolved_ref_raises(self):
        with pytest.raises(ValueError, match="bank entry"):
            _resolve_question_attachment_refs(
                [{"text": "x", "attachments": ["nope"]}],
                {"yes": {"type": "html", "text": "hi"}},
            )

    def test_no_bank_passes_questions_through(self):
        # Legacy v0.12.0 instruments without bank: behaviour preserved.
        qs = [{"text": "q", "attachments": [{"type": "image", "media_type": "image/png",
                                              "source": {"type": "base64", "data": "A"}}]}]
        out = _resolve_question_attachment_refs(qs, None)
        assert out == qs

    def test_question_without_attachments_unchanged(self):
        qs = [{"text": "no attachment"}]
        out = _resolve_question_attachment_refs(qs, {"x": {"type": "html", "text": "x"}})
        assert out == qs

    def test_resolved_dict_is_a_copy_not_alias(self):
        # Mutations to a resolved attachment must not bleed back to the bank.
        bank_entry = {"type": "image", "media_type": "image/png",
                      "source": {"type": "base64", "data": "data"}}
        bank = {"img": bank_entry}
        out = _resolve_question_attachment_refs(
            [{"text": "x", "attachments": ["img"]}], bank
        )
        out[0]["attachments"][0]["mutated"] = True
        assert "mutated" not in bank_entry

    def test_non_string_non_dict_ref_raises(self):
        with pytest.raises(ValueError, match="must be a string or mapping"):
            _resolve_question_attachment_refs(
                [{"text": "x", "attachments": [42]}],
                {"yes": {"type": "html", "text": "hi"}},
            )
