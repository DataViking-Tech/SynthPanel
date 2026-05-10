"""v1.0.2 G2 (hq-ovxl): panel-shared attachment lifting.

Bank entries referenced by ≥2 questions in a round are lifted onto
``run_panel_parallel(panel_shared_attachments=...)`` so they emit once
before the per-question cache_control marker (hq-0pbp canonical block
order) — letting Anthropic's prefix cache hit on the second persona +
question pair instead of paying the per-question cost every time.

A regression here means panels still *function* but pay full uncached
cost on every shared bank reference — silent economics failure.
"""

from __future__ import annotations

from synth_panel.orchestrator import (
    _compute_panel_shared,
    _resolve_question_attachment_refs,
)


def _img(data: str) -> dict:
    return {"type": "image", "media_type": "image/png", "source": {"type": "base64", "data": data}}


class TestComputePanelShared:
    def test_single_use_bank_entry_stays_per_question(self):
        bank = {"hero": _img("AAA")}
        questions = [{"text": "q1", "attachments": ["hero"]}, {"text": "q2"}]
        shared, ids = _compute_panel_shared(questions, bank)
        assert shared == []
        assert ids == set()

    def test_two_or_more_references_become_shared(self):
        bank = {"hero": _img("AAA"), "logo": _img("BBB")}
        questions = [
            {"text": "q1", "attachments": ["hero", "logo"]},
            {"text": "q2", "attachments": ["hero"]},
            {"text": "q3", "attachments": ["logo"]},
        ]
        shared, ids = _compute_panel_shared(questions, bank)
        assert ids == {"hero", "logo"}
        # Both bank entries lifted, deep-copied (not aliased).
        assert len(shared) == 2
        assert {s["source"]["data"] for s in shared} == {"AAA", "BBB"}
        assert shared[0] is not bank["hero"]

    def test_inline_dicts_never_become_shared(self):
        # Repeated inline dicts have no stable identity; placement is
        # author-deliberate and we don't second-guess it.
        inline = _img("INLINE")
        bank = {"hero": _img("AAA")}
        questions = [
            {"text": "q1", "attachments": [inline, "hero"]},
            {"text": "q2", "attachments": [inline]},
        ]
        shared, ids = _compute_panel_shared(questions, bank)
        # Only the bank entry counts toward sharing — and it's only used
        # by one question, so nothing is lifted.
        assert shared == []
        assert ids == set()

    def test_empty_or_missing_bank_returns_empty(self):
        questions = [{"text": "q1", "attachments": ["hero"]}]
        assert _compute_panel_shared(questions, None) == ([], set())
        assert _compute_panel_shared(questions, {}) == ([], set())

    def test_duplicate_ref_within_single_question_not_double_counted(self):
        # A question that lists the same bank ref twice does NOT count
        # as two references — the rule is "≥2 distinct questions".
        bank = {"hero": _img("AAA")}
        questions = [{"text": "q1", "attachments": ["hero", "hero"]}]
        shared, ids = _compute_panel_shared(questions, bank)
        assert shared == []
        assert ids == set()

    def test_unknown_ref_silently_ignored(self):
        # Unknown refs are surfaced as a hard ValueError by
        # _resolve_question_attachment_refs at the resolution site;
        # _compute_panel_shared just doesn't count them.
        bank = {"hero": _img("AAA")}
        questions = [
            {"text": "q1", "attachments": ["hero", "ghost"]},
            {"text": "q2", "attachments": ["hero"]},
        ]
        shared, ids = _compute_panel_shared(questions, bank)
        assert ids == {"hero"}
        assert len(shared) == 1


class TestExcludeRefsStripsFromPerQuestion:
    """The ``exclude_refs`` plumbing prevents double-emission once the
    G2 lift moves a bank entry onto ``panel_shared_attachments``."""

    def test_excluded_ref_dropped_from_per_question_attachments(self):
        bank = {"hero": _img("AAA"), "logo": _img("BBB")}
        questions = [
            {"text": "q1", "attachments": ["hero", "logo"]},
            {"text": "q2", "attachments": ["hero"]},
        ]
        out = _resolve_question_attachment_refs(questions, bank, exclude_refs={"hero"})
        # q1 keeps logo (not excluded), drops hero (lifted).
        assert len(out[0]["attachments"]) == 1
        assert out[0]["attachments"][0]["source"]["data"] == "BBB"
        # q2 had only hero — now empty.
        assert out[1]["attachments"] == []

    def test_excluded_ref_does_not_drop_inline_dicts(self):
        # exclude_refs only applies to bank-ref strings; inline dicts
        # always pass through untouched.
        inline = _img("INLINE")
        bank = {"hero": _img("AAA")}
        questions = [{"text": "q1", "attachments": ["hero", inline]}]
        out = _resolve_question_attachment_refs(questions, bank, exclude_refs={"hero"})
        assert out[0]["attachments"] == [inline]

    def test_no_exclude_refs_preserves_existing_behaviour(self):
        # Default call (no kwarg) matches the v1.0.1 behaviour locked in
        # by test_attachments_v1_0_1_wiring.py.
        bank = {"hero": _img("AAA")}
        questions = [{"text": "q1", "attachments": ["hero"]}]
        out = _resolve_question_attachment_refs(questions, bank)
        assert len(out[0]["attachments"]) == 1
        assert out[0]["attachments"][0]["source"]["data"] == "AAA"


class TestRoundTripLiftAndStrip:
    """End-to-end (helper-level): compute the shared set, then resolve
    the per-question lists with that set excluded. The two outputs
    together must reconstruct each original attachment exactly once —
    no double emission, no dropped attachments."""

    def test_ac_15_persona_3_question_panel_with_shared_images(self):
        # Stand-in for the AC: 3 questions, 2 of which share 5 images.
        bank = {f"img{i}": _img(f"D{i}") for i in range(5)}
        bank["q3_only"] = _img("Q3")
        questions = [
            {"text": "q1", "attachments": [f"img{i}" for i in range(5)]},
            {"text": "q2", "attachments": [f"img{i}" for i in range(5)]},
            {"text": "q3", "attachments": ["q3_only"]},
        ]

        shared, ids = _compute_panel_shared(questions, bank)
        resolved = _resolve_question_attachment_refs(questions, bank, exclude_refs=ids)

        # All 5 shared images lifted to panel_shared.
        assert len(shared) == 5
        assert ids == {f"img{i}" for i in range(5)}
        # q1 + q2 attachments fully lifted out (now empty per-question).
        assert resolved[0]["attachments"] == []
        assert resolved[1]["attachments"] == []
        # q3's single-use bank entry stays per-question (not lifted).
        assert len(resolved[2]["attachments"]) == 1
        assert resolved[2]["attachments"][0]["source"]["data"] == "Q3"
