"""hq-ilke regression: legacy single-round path must resolve bank-ref attachments.

Before this fix, ``run_panel_parallel`` (the legacy single-round entry point)
silently dropped bank-ref strings (``question.attachments = ["hero"]``) at the
downstream dict-only filter — panelists responded as if no attachment had been
provided. The bank-ref pattern is the canonical hq-xzsm shape, so this hit the
common case for v1 instruments authored with an attachment bank.

Resolution previously happened only inside ``run_multi_round_panel``; the v1 /
linear-v2 path went directly to ``run_panel_parallel`` and never invoked
``_resolve_question_attachment_refs``. The fix routes the attachment bank into
``run_panel_parallel`` as the new ``attachment_bank`` parameter and resolves
refs internally before any worker spawns.

A regression here means panels appear to run successfully but every persona
sees text-only content — silent data loss with no log line, no error.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.llm.models import (
    CompletionResponse,
    ImageBlock,
    InputMessage,
    StopReason,
    TextBlock,
)
from synth_panel.llm.models import (
    TokenUsage as LLMTokenUsage,
)
from synth_panel.orchestrator import run_panel_parallel


def _ack_response() -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text="ack")],
        stop_reason=StopReason.END_TURN,
        usage=LLMTokenUsage(input_tokens=10, output_tokens=2),
    )


def _system_prompt(persona: dict[str, Any]) -> str:
    return f"You are {persona.get('name', 'Anonymous')}."


def _question_prompt(question: dict[str, Any]) -> str:
    return question.get("text", str(question)) if isinstance(question, dict) else str(question)


def _capture_client() -> tuple[MagicMock, list[Any]]:
    captured: list[Any] = []
    lock = threading.Lock()

    def send(request):
        with lock:
            captured.append(request)
        return _ack_response()

    client = MagicMock()
    client.send = MagicMock(side_effect=send)
    return client, captured


def _flatten_blocks(messages: list[InputMessage]) -> list[Any]:
    blocks: list[Any] = []
    for msg in messages:
        if isinstance(msg.content, list):
            blocks.extend(msg.content)
    return blocks


class TestAttachmentBankResolvesBankRefs:
    """v1 instrument + bank-ref attachment shape must reach the model."""

    def test_bank_ref_string_resolved_when_attachment_bank_passed(self):
        bank = {
            "hero": {
                "type": "image",
                "media_type": "image/png",
                "source": {"type": "base64", "data": "AAAA"},
            }
        }
        questions = [{"text": "React to the hero", "attachments": ["hero"]}]
        client, captured = _capture_client()

        results, _registry, _sessions = run_panel_parallel(
            client=client,
            personas=[{"name": "Alice"}],
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            attachment_bank=bank,
        )

        assert len(results) == 1
        assert results[0].error is None

        # The image block must reach the model. Before the fix, every
        # request went out as text-only because the dict-only filter
        # dropped the bare string ref.
        assert captured, "no request reached the client"
        all_blocks = []
        for req in captured:
            all_blocks.extend(_flatten_blocks(req.messages))
        image_blocks = [b for b in all_blocks if isinstance(b, ImageBlock)]
        assert image_blocks, "ImageBlock missing from request — bank-ref silently dropped"
        # Source must be the bank entry's data, not a renamed/empty source.
        sources = [getattr(b.source, "data", None) for b in image_blocks]
        assert "AAAA" in sources

    def test_bank_omitted_keeps_legacy_dict_only_behaviour(self):
        # Without ``attachment_bank``, bare bank-ref strings still fall out
        # at the downstream filter — preserves the v0.12.0 contract for
        # callers that don't have an Instrument in scope. The fix is
        # opt-in: callers must pass the bank to get resolution.
        questions = [{"text": "react", "attachments": ["hero"]}]
        client, captured = _capture_client()

        results, _registry, _sessions = run_panel_parallel(
            client=client,
            personas=[{"name": "Alice"}],
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
        )

        assert results[0].error is None
        all_blocks = []
        for req in captured:
            all_blocks.extend(_flatten_blocks(req.messages))
        # No ImageBlock — string ref had no bank to resolve against and was
        # filtered out, same as before the fix.
        assert not [b for b in all_blocks if isinstance(b, ImageBlock)]

    def test_inline_dict_attachments_still_pass_through(self):
        # The new code path must not regress callers that already use
        # inline-dict shape: those bypass the resolver but the bank, when
        # supplied, must not interfere.
        bank = {"unused": {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "ZZZ"}}}
        inline = {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "BBBB"}}
        questions = [{"text": "react", "attachments": [inline]}]
        client, captured = _capture_client()

        run_panel_parallel(
            client=client,
            personas=[{"name": "Alice"}],
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            attachment_bank=bank,
        )

        all_blocks = []
        for req in captured:
            all_blocks.extend(_flatten_blocks(req.messages))
        image_data = [getattr(b.source, "data", None) for b in all_blocks if isinstance(b, ImageBlock)]
        assert "BBBB" in image_data
        # Unused bank entries do NOT bleed into the request.
        assert "ZZZ" not in image_data

    def test_unresolved_ref_raises(self):
        # When the bank is provided but a ref doesn't resolve, surface
        # the failure loudly — silent fallback is exactly the bug class
        # this fix exists to prevent.
        bank = {"hero": {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "AAAA"}}}
        questions = [{"text": "react", "attachments": ["ghost"]}]
        client, _captured = _capture_client()

        with pytest.raises(ValueError, match="bank entry"):
            run_panel_parallel(
                client=client,
                personas=[{"name": "Alice"}],
                questions=questions,
                model="sonnet",
                system_prompt_fn=_system_prompt,
                question_prompt_fn=_question_prompt,
                attachment_bank=bank,
            )

    def test_attachment_bank_and_panel_shared_attachments_mutually_exclusive(self):
        # Multi-round callers pre-compute panel_shared_attachments and pass
        # that. Legacy callers pass attachment_bank. Passing both signals a
        # caller-side wiring bug — reject loudly.
        bank = {"hero": {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "AAAA"}}}
        questions = [{"text": "react", "attachments": ["hero"]}]
        client, _captured = _capture_client()

        with pytest.raises(ValueError, match="mutually exclusive"):
            run_panel_parallel(
                client=client,
                personas=[{"name": "Alice"}],
                questions=questions,
                model="sonnet",
                system_prompt_fn=_system_prompt,
                question_prompt_fn=_question_prompt,
                attachment_bank=bank,
                panel_shared_attachments=[
                    {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "X"}}
                ],
            )

    def test_shared_bank_entry_lifted_to_panel_shared(self):
        # Two questions referencing the same bank entry triggers the G2
        # lift: the entry should appear in the system block (panel_shared
        # path), and per-question lists become empty for that ref. We
        # only assert that the image survives end-to-end; exact lift
        # placement is covered by test_attachments_v1_0_2_panel_shared.
        bank = {"hero": {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": "SHARED"}}}
        questions = [
            {"text": "q1", "attachments": ["hero"]},
            {"text": "q2", "attachments": ["hero"]},
        ]
        client, captured = _capture_client()

        run_panel_parallel(
            client=client,
            personas=[{"name": "Alice"}],
            questions=questions,
            model="sonnet",
            system_prompt_fn=_system_prompt,
            question_prompt_fn=_question_prompt,
            attachment_bank=bank,
        )

        # Both turns should see the shared image (once via shared block,
        # not duplicated via per-question). The smoking-gun bug-symptom
        # was zero ImageBlocks anywhere in the conversation.
        all_blocks = []
        for req in captured:
            all_blocks.extend(_flatten_blocks(req.messages))
        shared_data = [getattr(b.source, "data", None) for b in all_blocks if isinstance(b, ImageBlock)]
        assert "SHARED" in shared_data
