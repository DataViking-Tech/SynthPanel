"""Tests for sp-4g6a: per-question sub-chunk + auto-escalate on overflow."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.models import (
    CompletionResponse,
    TokenUsage,
    ToolInvocationBlock,
)
from synth_panel.orchestrator import PanelistResult
from synth_panel.synthesis import (
    _ESCALATION_MODEL,
    _MAP_PROMPT_TEMPLATE,
    STRATEGY_MAP_REDUCE,
    MapChunkOverflowError,
    SynthesisResult,
    _partition_panelists_for_context,
    _sub_chunk_question_synthesis,
    synthesize_panel_mapreduce,
)

# --- Fixtures ---


def _tool_response(marker: str, input_tokens: int = 100, output_tokens: int = 50) -> CompletionResponse:
    data = {
        "summary": f"summary-{marker}",
        "themes": [f"theme-{marker}"],
        "agreements": [f"agree-{marker}"],
        "disagreements": [f"disagree-{marker}"],
        "surprises": [f"surprise-{marker}"],
        "recommendation": f"rec-{marker}",
    }
    return CompletionResponse(
        id=f"synth-{marker}",
        model="claude-sonnet-4-6",
        content=[ToolInvocationBlock(id="tc1", name="synthesize", input=data)],
        usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _SequenceClient:
    """Thread-safe mock LLM client returning pre-seeded responses in order."""

    def __init__(self, responses: list[CompletionResponse]):
        self._responses = list(responses)
        self._idx = 0
        self.call_count = 0
        self._lock = threading.Lock()
        self.sent_requests: list = []

    def send(self, request, **kwargs):
        with self._lock:
            resp = self._responses[self._idx]
            self._idx += 1
            self.call_count += 1
            self.sent_requests.append(request)
        return resp


def _fat_panelists(n: int, question_text: str, chars_per_response: int) -> list[PanelistResult]:
    """Fabricate n panelists whose single-question response is `chars_per_response` long."""
    blob = "x" * chars_per_response
    return [
        PanelistResult(
            persona_name=f"Panelist{i}",
            responses=[{"question": question_text, "response": blob}],
            usage=ZERO_USAGE,
        )
        for i in range(n)
    ]


_Q_OVERFLOW = [{"text": "What drives you mad about the product?"}]


# --- (A) Sub-chunking ---


class TestPartition:
    def test_greedy_partition_splits_when_overflow(self):
        """Partition splits panelists into batches each fitting the limit."""
        # 10 panelists, 80k chars each. Each ≈ 20k tokens. Limit 60k tokens
        # (after the 2k scaffold floor) — expect ~3 panelists per batch.
        panelists = _fat_panelists(10, _Q_OVERFLOW[0]["text"], 80_000)
        batches = _partition_panelists_for_context(
            panelists, _Q_OVERFLOW[0], _MAP_PROMPT_TEMPLATE, context_limit=60_000
        )
        assert batches is not None
        assert sum(len(b) for b in batches) == 10
        assert len(batches) >= 2
        # No single batch exceeds the limit
        from synth_panel.synthesis import estimate_single_pass_tokens

        for b in batches:
            assert estimate_single_pass_tokens(b, _Q_OVERFLOW, prompt=_MAP_PROMPT_TEMPLATE) <= 60_000

    def test_partition_returns_none_when_single_panelist_overflows(self):
        """A single panelist whose response exceeds the limit signals failure."""
        panelists = _fat_panelists(3, _Q_OVERFLOW[0]["text"], 500_000)  # ~125k tokens each
        batches = _partition_panelists_for_context(
            panelists, _Q_OVERFLOW[0], _MAP_PROMPT_TEMPLATE, context_limit=20_000
        )
        assert batches is None


class TestSubChunkWithinQuestion:
    def test_sub_chunk_within_question_when_overflow(self):
        """When a question overflows, sub-chunking yields a valid per-question summary.

        End-to-end: 20 fat panelists, long responses, sonnet (200k ctx). Each
        panelist's response is ~40k tokens, so the 20-panelist call would
        estimate at ~800k tokens >> 192k limit. Sub-chunker must partition
        into multiple batches, map each, and inner-reduce.
        """
        panelists = _fat_panelists(20, _Q_OVERFLOW[0]["text"], 160_000)
        context_limit = 200_000 - 8_000
        expected_batches = _partition_panelists_for_context(
            panelists, _Q_OVERFLOW[0], _MAP_PROMPT_TEMPLATE, context_limit
        )
        assert expected_batches is not None and len(expected_batches) >= 2
        n_batches = len(expected_batches)

        responses = [_tool_response(f"batch-{i}") for i in range(n_batches)]
        responses.append(_tool_response("inner-reduce"))
        client = _SequenceClient(responses)

        result, batch_count = _sub_chunk_question_synthesis(
            client,
            panelists,
            _Q_OVERFLOW[0],
            _MAP_PROMPT_TEMPLATE,
            model="sonnet",
            panelist_model=None,
            context_limit=context_limit,
            temperature=None,
            top_p=None,
        )

        assert isinstance(result, SynthesisResult)
        assert batch_count == n_batches
        assert result.summary == "summary-inner-reduce"
        # Total calls = batch_count + 1 (inner reduce)
        assert client.call_count == batch_count + 1

    def test_reduce_phase_consumes_batch_summaries_correctly(self):
        """Inner reduce sees every batch summary and nothing else.

        We use a MagicMock client and inspect each call's user content.
        """
        panelists = _fat_panelists(10, _Q_OVERFLOW[0]["text"], 160_000)
        context_limit = 200_000 - 8_000
        expected_batches = _partition_panelists_for_context(
            panelists, _Q_OVERFLOW[0], _MAP_PROMPT_TEMPLATE, context_limit
        )
        assert expected_batches is not None
        n_batches = len(expected_batches)

        responses = [_tool_response(f"batch-{i}") for i in range(n_batches)]
        responses.append(_tool_response("INNER"))
        client = MagicMock()
        client.send.side_effect = responses

        _sub_chunk_question_synthesis(
            client,
            panelists,
            _Q_OVERFLOW[0],
            _MAP_PROMPT_TEMPLATE,
            model="sonnet",
            panelist_model=None,
            context_limit=context_limit,
            temperature=None,
            top_p=None,
        )

        # Last call is the inner reduce
        inner_request = client.send.call_args_list[-1].args[0]
        inner_text = inner_request.messages[0].content[0].text
        # Inner reduce prompt appears
        assert "combining partial summaries" in inner_text
        # Every preceding batch's summary flows in as a synthetic panelist
        n_batch_calls = len(client.send.call_args_list) - 1
        for i in range(n_batch_calls):
            assert f"summary-batch-{i}" in inner_text

    def test_sub_chunk_aggregates_cost_and_tokens(self):
        """Returned result totals usage/cost across every batch + inner reduce."""
        panelists = _fat_panelists(10, _Q_OVERFLOW[0]["text"], 160_000)
        context_limit = 200_000 - 8_000
        expected = _partition_panelists_for_context(panelists, _Q_OVERFLOW[0], _MAP_PROMPT_TEMPLATE, context_limit)
        assert expected is not None
        n_batches = len(expected)

        responses = [_tool_response(f"batch-{i}", input_tokens=100, output_tokens=10) for i in range(n_batches)]
        responses.append(_tool_response("inner", input_tokens=500, output_tokens=200))
        client = _SequenceClient(responses)

        result, batch_count = _sub_chunk_question_synthesis(
            client,
            panelists,
            _Q_OVERFLOW[0],
            _MAP_PROMPT_TEMPLATE,
            model="sonnet",
            panelist_model=None,
            context_limit=context_limit,
            temperature=None,
            top_p=None,
        )
        # batch_count * 110 + 700 inner
        assert result.usage.total_tokens == (batch_count * 110) + 700
        assert batch_count == n_batches

    def test_single_panelist_overflow_raises(self):
        """When even one panelist's response exceeds the limit, sub-chunk raises."""
        panelists = _fat_panelists(3, _Q_OVERFLOW[0]["text"], 800_000)
        client = _SequenceClient([])

        with pytest.raises(MapChunkOverflowError) as excinfo:
            _sub_chunk_question_synthesis(
                client,
                panelists,
                _Q_OVERFLOW[0],
                _MAP_PROMPT_TEMPLATE,
                model="sonnet",
                panelist_model=None,
                context_limit=50_000,
                temperature=None,
                top_p=None,
            )
        assert excinfo.value.diagnostic.get("reason") == "single_panelist_overflow"


# --- (A) integrated into map-reduce ---


class TestMapReduceAutoSubChunks:
    def test_map_reduce_sub_chunks_overflowing_question_by_default(self):
        """synthesize_panel_mapreduce handles per-chunk overflow via sub-chunking."""
        # 1 overflowing question + 1 fitting question. Verify pipeline succeeds.
        panelists = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[
                    {"question": "Big open-ended", "response": "x" * 160_000},
                    {"question": "Small", "response": "quick"},
                ],
                usage=ZERO_USAGE,
            )
            for i in range(12)
        ]
        questions = [{"text": "Big open-ended"}, {"text": "Small"}]
        # Expect: many batch calls for Q0 + 1 inner reduce for Q0, 1 map for Q1, 1 outer reduce.
        # Seed plenty.
        responses = [_tool_response(f"call-{i}") for i in range(20)]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="sonnet",
            max_workers=1,
        )
        assert result.strategy == STRATEGY_MAP_REDUCE
        assert result.per_question_synthesis is not None
        assert 0 in result.per_question_synthesis
        assert 1 in result.per_question_synthesis

        # The overflowing question's map_breakdown entry should be sub_chunked.
        assert result.map_cost_breakdown is not None
        q0_entry = result.map_cost_breakdown[0]
        assert q0_entry.get("sub_chunked") is True
        assert q0_entry.get("batch_count", 0) >= 2
        # Q1 was small, no sub-chunking.
        q1_entry = result.map_cost_breakdown[1]
        assert q1_entry.get("sub_chunked") is not True


# --- (B) auto-escalate ---


class TestAutoEscalate:
    def test_auto_escalate_respects_flag(self, capsys):
        """auto_escalate=True swaps model to the 1M-ctx target and warns."""
        # Question that overflows sonnet (200k) but fits gemini-flash-lite (1M).
        # 20 panelists, 120k chars each ≈ 30k tokens each → ~600k tokens total,
        # under 1M-8k effective escalated limit.
        panelists = _fat_panelists(20, "Overflowing Q", 120_000)
        questions = [{"text": "Overflowing Q"}]

        responses = [_tool_response("escalated-map"), _tool_response("outer-reduce")]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="sonnet",
            max_workers=1,
            auto_escalate=True,
        )

        # Warning appeared on stderr mentioning both models.
        captured = capsys.readouterr()
        assert "auto-escalated" in captured.err
        assert _ESCALATION_MODEL in captured.err

        # Breakdown records the escalation.
        assert result.map_cost_breakdown is not None
        entry = result.map_cost_breakdown[0]
        assert entry.get("escalated_model") == _ESCALATION_MODEL
        # Only 2 calls: one escalated map, one outer reduce — no sub-chunking.
        assert client.call_count == 2

    def test_auto_escalate_false_uses_sub_chunk(self, capsys):
        """Flag off preserves the deterministic sub-chunk path (no escalation warning)."""
        panelists = _fat_panelists(12, "Overflowing Q", 120_000)
        questions = [{"text": "Overflowing Q"}]
        responses = [_tool_response(f"call-{i}") for i in range(15)]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="sonnet",
            max_workers=1,
            auto_escalate=False,
        )

        captured = capsys.readouterr()
        assert "auto-escalated" not in captured.err
        assert result.map_cost_breakdown is not None
        assert result.map_cost_breakdown[0].get("sub_chunked") is True
        assert result.map_cost_breakdown[0].get("escalated_model") is None

    def test_auto_escalate_falls_back_to_sub_chunk_when_even_escalated_overflows(self, capsys):
        """If escalated model's context is also insufficient, still sub-chunk."""
        # Very long responses per panelist: 800k chars ≈ 200k tokens each.
        # With 20 panelists that's ~4M tokens > 1M even escalated → must
        # sub-chunk using the escalated model's 1M window.
        panelists = _fat_panelists(20, "Overflowing Q", 800_000)
        questions = [{"text": "Overflowing Q"}]
        responses = [_tool_response(f"call-{i}") for i in range(30)]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="sonnet",
            max_workers=1,
            auto_escalate=True,
        )
        entry = result.map_cost_breakdown[0]
        # Sub-chunked, on the escalated model.
        assert entry.get("sub_chunked") is True
        assert entry.get("model") == _ESCALATION_MODEL


# --- End-to-end at simulated large-n ---


class TestLargeN:
    def test_simulated_n150_long_form_instrument(self):
        """n=150 panelists x 5 long-form questions succeeds under sub-chunking.

        Simulates the conditions of the real audit that caught sp-4g6a:
        n~100 on product-feedback overflowed haiku's 192k effective limit.
        Here we exaggerate per-panelist response length so sub-chunking
        is definitely triggered, then assert the pipeline produces a
        non-fallback SynthesisResult.
        """
        q_texts = [
            "What frustrates you most?",
            "What would you change first?",
            "How does this compare to alternatives?",
            "What would make you recommend this?",
            "What have we missed?",
        ]
        questions = [{"text": t} for t in q_texts]
        panelists = [
            PanelistResult(
                persona_name=f"P{i}",
                responses=[{"question": t, "response": "x" * 30_000} for t in q_texts],
                usage=ZERO_USAGE,
            )
            for i in range(150)
        ]

        # Be generous with pre-seeded responses: multiple batches per question
        # × 5 questions + 5 inner reduces + 1 outer reduce. Upper-bound ~200.
        responses = [_tool_response(f"c{i}") for i in range(400)]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="haiku",
            max_workers=1,
        )
        assert result.strategy == STRATEGY_MAP_REDUCE
        assert not result.is_fallback
        assert result.per_question_synthesis is not None
        assert set(result.per_question_synthesis.keys()) == {0, 1, 2, 3, 4}
        # Every question on haiku (200k) with ~30k-token responses × 150
        # panelists must have sub-chunked.
        for entry in result.map_cost_breakdown:
            assert entry.get("sub_chunked") is True


# --- sp-34f7: long-question-prefix regression ---


class TestLongQuestionPrefix:
    """Regression for sp-34f7: question-text prefix must factor into the
    sub-chunk token budget.

    ``_format_panelist_data`` echoes the question text once per panelist's
    response block (``  Q: <text>\\n  A: <answer>``), so a ~2.5k-token
    product-brief question duplicated across 100 panelists contributes
    ~250k tokens on its own — enough to overflow haiku's 192k window even
    with trivially short answers. The audit run
    ``ensemble_100_v099_ctx2/synthpanel__product-feedback.json`` failed
    this way before sub-chunking was in place; these tests pin that the
    estimator and partitioner both see the prefix cost and split
    accordingly.
    """

    def _long_brief_question_text(self) -> str:
        # ~13k chars ≈ 3.3k tokens under the 4-chars-per-token heuristic.
        # Mirrors the product-feedback Q1 shape that tripped the audit
        # (ensemble_100_v099_ctx2): a short stem + a multi-paragraph
        # brief whose content dwarfs the actual ask. Sized so 101
        # repetitions push the map body past haiku's 192k effective
        # limit with trivially short answers.
        header = "On a scale of 0 to 10, how likely are you to recommend SynthPanel to a peer? Context follows.\n\n"
        brief = (
            "SynthPanel is an open-source Python package and MCP server that "
            "orchestrates synthetic focus groups — panels of LLM-powered "
            "personas that answer research questions in parallel. "
        ) * 50
        return header + brief

    def test_partition_accounts_for_repeated_question_prefix(self):
        """Short answers + very long Q text still forces partitioning.

        If partition ignored the per-panelist Q echo, it would return a
        single batch and the downstream map call would overflow at
        runtime. Asserting ``len(batches) >= 2`` pins that the estimator
        charges each panelist slot for the duplicated prefix.
        """
        q_text = self._long_brief_question_text()
        # Trivially short answers: prefix must be the overflow driver.
        panelists = _fat_panelists(100, q_text, chars_per_response=50)
        q = {"text": q_text}

        # Sanity: the full-panel single-pass estimate must exceed haiku's
        # effective limit — otherwise the scenario isn't actually
        # exercising the overflow path.
        from synth_panel.synthesis import estimate_single_pass_tokens

        full_est = estimate_single_pass_tokens(panelists, [q], prompt=_MAP_PROMPT_TEMPLATE)
        assert full_est > 192_000, f"scenario must overflow haiku; got {full_est}"

        batches = _partition_panelists_for_context(panelists, q, _MAP_PROMPT_TEMPLATE, context_limit=192_000)
        assert batches is not None, "sub-chunking must be able to split"
        assert len(batches) >= 2, f"expected multiple batches, got {len(batches)}"
        assert sum(len(b) for b in batches) == 100

        # Every batch fits the limit once the prefix cost is counted.
        for b in batches:
            est_b = estimate_single_pass_tokens(b, [q], prompt=_MAP_PROMPT_TEMPLATE)
            assert est_b <= 192_000, f"batch of {len(b)} still overflows: {est_b} > 192000"

    def test_map_reduce_sub_chunks_long_prefix_on_haiku(self):
        """End-to-end: n=100 + long-brief Q on haiku succeeds via sub-chunking.

        Reproduces the shape of the audit failure
        (``ensemble_100_v099_ctx2/synthpanel__product-feedback.json``):
        n=100 panelists answering a long-brief question on haiku. The
        current sub-chunk fallback must carry the run to completion
        rather than raising ``MapChunkOverflowError``.
        """
        q_text = self._long_brief_question_text()
        panelists = _fat_panelists(100, q_text, chars_per_response=50)
        questions = [{"text": q_text}]

        # Upper-bound responses: a handful of batches + 1 inner reduce + 1 outer reduce.
        responses = [_tool_response(f"call-{i}") for i in range(20)]
        client = _SequenceClient(responses)

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            questions,
            model="haiku",
            max_workers=1,
        )
        assert result.strategy == STRATEGY_MAP_REDUCE
        assert result.per_question_synthesis is not None
        assert 0 in result.per_question_synthesis
        assert result.map_cost_breakdown is not None
        entry = result.map_cost_breakdown[0]
        assert entry.get("sub_chunked") is True
        assert entry.get("batch_count", 0) >= 2
