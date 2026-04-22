"""sp-exu6: --synthesis-strategy=auto must route overflow to map-reduce.

Covers the sp-avmm x sp-9rzu interaction that previously caused
``--synthesis-strategy=auto`` to fail loud on oversized panels instead of
routing them through map-reduce. The bug was the order of operations in
``cmd_panel_run``: the overflow pre-flight ran BEFORE strategy selection,
so any panel too large for a single pass was rejected regardless of
whether map-reduce could have handled it.

Acceptance criteria:

1. ``select_strategy`` runs first; the single-pass pre-flight only fires
   when the resolved strategy is ``single``.
2. ``auto`` routes overflow to ``map-reduce``.
3. Map-reduce has its own per-map overflow guard.
4. ``--synthesis-strategy=single`` preserves the loud sp-avmm rejection.
5. ``--synthesis-strategy=map-reduce`` bypasses the whole-panel overflow
   guard (per-map guard still applies).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.models import CompletionResponse, ToolInvocationBlock
from synth_panel.llm.models import TokenUsage as LLMTokenUsage
from synth_panel.main import main
from synth_panel.orchestrator import PanelistResult
from synth_panel.synthesis import (
    MapChunkOverflowError,
    SynthesisResult,
    synthesize_panel_mapreduce,
)

# --- helpers --------------------------------------------------------------


def _write_personas(path, n: int = 2) -> str:
    names = [f"Panelist{i}" for i in range(n)]
    body = "personas:\n" + "\n".join(f"  - name: {name}" for name in names) + "\n"
    path.write_text(body)
    return str(path)


def _write_instrument(path, question: str = "Q1") -> str:
    path.write_text(f"instrument:\n  questions:\n    - text: {question}\n")
    return str(path)


def _tool_response(marker: str) -> CompletionResponse:
    return CompletionResponse(
        id=f"synth-{marker}",
        model="claude-haiku-4-5",
        content=[
            ToolInvocationBlock(
                id=f"tc-{marker}",
                name="synthesize",
                input={
                    "summary": f"summary-{marker}",
                    "themes": [f"theme-{marker}"],
                    "agreements": [f"agree-{marker}"],
                    "disagreements": [f"disagree-{marker}"],
                    "surprises": [f"surprise-{marker}"],
                    "recommendation": f"rec-{marker}",
                },
            )
        ],
        usage=LLMTokenUsage(input_tokens=100, output_tokens=50),
    )


# --- acceptance tests -----------------------------------------------------


class TestAutoRoutesOverflowToMapReduce:
    @patch("synth_panel.cli.commands.synthesize_panel_mapreduce")
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_auto_falls_over_to_map_reduce_when_single_overflows(
        self, _mock_client, mock_run, mock_synth_single, mock_synth_mr, capsys, tmp_path
    ):
        """sp-exu6 AC 1, 2, 6: auto + overflow → map-reduce (not hard fail)."""
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.orchestrator import WorkerRegistry

        # Large enough to overflow single-pass haiku (200k - 8k headroom).
        giant = "A" * 800_000
        huge_results = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[{"question": "Q1", "response": giant}],
                usage=ZERO_USAGE,
                model="haiku",
            )
            for i in range(2)
        ]
        mock_run.return_value = (huge_results, WorkerRegistry(), {})
        # map-reduce succeeds — the point of the test is that we reach it.
        mock_synth_mr.return_value = SynthesisResult(
            summary="ok",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            usage=TokenUsage(),
            cost=CostEstimate(),
            strategy="map-reduce",
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
                "--synthesis-model",
                "haiku",
                "--synthesis-strategy",
                "auto",
            ]
        )

        # Must NOT reject pre-flight; must route to map-reduce instead.
        assert code == 0, "auto must route overflow to map-reduce, not fail"
        mock_synth_single.assert_not_called()
        mock_synth_mr.assert_called_once()


class TestSingleStillRejectsOnOverflow:
    @patch("synth_panel.cli.commands.synthesize_panel_mapreduce")
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_single_still_rejects_on_overflow(
        self, _mock_client, mock_run, mock_synth_single, mock_synth_mr, capsys, tmp_path
    ):
        """sp-exu6 AC 4: explicit single preserves the sp-avmm pre-flight."""
        from synth_panel.orchestrator import WorkerRegistry

        giant = "A" * 800_000
        huge_results = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[{"question": "Q1", "response": giant}],
                usage=ZERO_USAGE,
                model="haiku",
            )
            for i in range(2)
        ]
        mock_run.return_value = (huge_results, WorkerRegistry(), {})

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
                "--synthesis-model",
                "haiku",
                "--synthesis-strategy",
                "single",
            ]
        )

        assert code == 2
        captured = capsys.readouterr()
        assert "pre-flight" in captured.err.lower() or "exceeds" in captured.err.lower()
        mock_synth_single.assert_not_called()
        mock_synth_mr.assert_not_called()

        payload = json.loads(captured.out)
        err = payload["synthesis_error"]
        assert err["error_type"] == "synthesis_context_overflow"


class TestMapReduceSkipsPanelLevelOverflow:
    @patch("synth_panel.cli.commands.synthesize_panel_mapreduce")
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_map_reduce_bypasses_panel_level_overflow(
        self, _mock_client, mock_run, mock_synth_single, mock_synth_mr, capsys, tmp_path
    ):
        """sp-exu6 AC 5: explicit map-reduce skips the whole-panel pre-flight."""
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.orchestrator import WorkerRegistry

        giant = "A" * 800_000
        huge_results = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[{"question": "Q1", "response": giant}],
                usage=ZERO_USAGE,
                model="haiku",
            )
            for i in range(2)
        ]
        mock_run.return_value = (huge_results, WorkerRegistry(), {})
        mock_synth_mr.return_value = SynthesisResult(
            summary="ok",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="",
            usage=TokenUsage(),
            cost=CostEstimate(),
            strategy="map-reduce",
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
                "--synthesis-model",
                "haiku",
                "--synthesis-strategy",
                "map-reduce",
            ]
        )

        assert code == 0
        mock_synth_single.assert_not_called()
        mock_synth_mr.assert_called_once()


class TestMapReducePerChunkOverflowCheck:
    def test_map_reduce_per_chunk_overflow_check_unit(self):
        """sp-exu6 AC 3 / sp-4g6a: per-map guard fires when a single
        panelist's response overflows the synthesis model's context.

        With sp-4g6a, map-reduce attempts sub-chunking before giving up,
        so the overflow must be large enough that even a single panelist's
        response cannot fit. Then the sub-chunker raises
        ``MapChunkOverflowError`` with ``reason=single_panelist_overflow``.
        """
        giant = "A" * 800_000  # ~200k tokens per panelist — exceeds haiku limit alone
        fat_panelists = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[{"question": "Q1", "response": giant}],
                usage=ZERO_USAGE,
            )
            for i in range(2)
        ]
        client = MagicMock()

        with pytest.raises(MapChunkOverflowError) as excinfo:
            synthesize_panel_mapreduce(
                client,
                fat_panelists,
                [{"text": "Q1"}],
                model="haiku",
                max_workers=1,
            )

        diag = excinfo.value.diagnostic
        assert diag["question_index"] == 0
        assert diag["estimated_tokens"] > diag["effective_limit"]
        assert diag.get("reason") == "single_panelist_overflow"
        # Must short-circuit before any LLM call.
        client.send.assert_not_called()

    def test_map_reduce_per_chunk_ok_for_small_chunks(self):
        """Per-chunk guard does NOT fire when each chunk fits — the bug
        being guarded against is a single oversized question, not merely
        many small ones."""
        # 2 questions × 2 panelists, short answers — fits comfortably.
        panelists = [
            PanelistResult(
                persona_name=f"Panelist{i}",
                responses=[
                    {"question": "Q1", "response": "short answer one"},
                    {"question": "Q2", "response": "short answer two"},
                ],
                usage=ZERO_USAGE,
            )
            for i in range(2)
        ]
        # 2 map calls + 1 reduce call
        responses = [_tool_response(f"map-{i}") for i in range(2)]
        responses.append(_tool_response("reduce"))
        client = MagicMock()
        client.send.side_effect = responses

        result = synthesize_panel_mapreduce(
            client,
            panelists,
            [{"text": "Q1"}, {"text": "Q2"}],
            model="haiku",
            max_workers=1,
        )
        assert result.summary == "summary-reduce"
        # 2 maps + 1 reduce
        assert client.send.call_count == 3


class TestCliMapChunkOverflowErrorShape:
    @patch("synth_panel.cli.commands.synthesize_panel_mapreduce")
    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.cli.commands.run_panel_parallel")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_per_chunk_overflow_becomes_structured_payload(
        self, _mock_client, mock_run, mock_synth_single, mock_synth_mr, capsys, tmp_path
    ):
        """A raised MapChunkOverflowError becomes a structured
        ``synthesis_map_chunk_overflow`` error in the CLI envelope."""
        from synth_panel.orchestrator import WorkerRegistry

        mock_run.return_value = (
            [
                PanelistResult(
                    persona_name="P",
                    responses=[{"question": "Q1", "response": "short"}],
                    usage=ZERO_USAGE,
                    model="haiku",
                )
            ],
            WorkerRegistry(),
            {},
        )
        mock_synth_mr.side_effect = MapChunkOverflowError(
            "per-chunk overflow: question 0 too big",
            diagnostic={
                "question_index": 0,
                "question_text": "Q1",
                "estimated_tokens": 999_999,
                "effective_limit": 192_000,
                "synthesis_model": "haiku",
            },
        )

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--personas",
                _write_personas(tmp_path / "personas.yaml"),
                "--instrument",
                _write_instrument(tmp_path / "survey.yaml"),
                "--synthesis-model",
                "haiku",
                "--synthesis-strategy",
                "map-reduce",
            ]
        )

        assert code == 2
        payload = json.loads(capsys.readouterr().out)
        err = payload["synthesis_error"]
        assert err["error_type"] == "synthesis_map_chunk_overflow"
        assert err["diagnostic"]["question_index"] == 0
