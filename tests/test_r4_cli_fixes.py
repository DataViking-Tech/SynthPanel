"""Regression tests for the gh-r4 naive-user CLI fixes.

Covers:

* Routing — a one-entry weight-free ``--models`` spec is a standard
  single-model panel run (synthesis included), not an "ensemble" that
  silently skips synthesis and exits 0 with a bare "Ensemble complete".
* Cost — a provider-reported cost of exactly $0 for a run that consumed
  tokens (OpenRouter BYOK) must not zero out the recorded run cost or
  starve :class:`~synth_panel.cost.CostGate`; the local pricing-table
  estimate is used instead.
* Ensemble stdout — genuine (>=2 model) ensemble runs now state on stdout
  that synthesis was skipped and print the exact follow-up command.
* ``cost show <result-id>`` — per-run cost breakdown alias over the
  inspect data.
* Help text — ``--personas`` / ``--instrument`` document bundled
  pack/instrument names, not only YAML paths.
"""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import CostGate, TokenUsage, resolve_cost
from synth_panel.main import main
from synth_panel.persistence import ConversationMessage
from synth_panel.runtime import TurnSummary


def _mock_turn_summary(text: str = "I think so.") -> TurnSummary:
    usage = TokenUsage(input_tokens=1000, output_tokens=200)
    msg = ConversationMessage(
        role="assistant",
        content=[{"type": "text", "text": text}],
        usage=usage,
    )
    return TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)


def _mock_synthesis_result():
    from synth_panel.cost import CostEstimate
    from synth_panel.synthesis import SynthesisResult

    return SynthesisResult(
        summary="Test synthesis summary",
        themes=["theme1"],
        agreements=[],
        disagreements=[],
        surprises=[],
        recommendation="Do X",
        usage=TokenUsage(input_tokens=100, output_tokens=50),
        cost=CostEstimate(),
        model="modelA",
    )


@pytest.fixture
def panel_files(tmp_path):
    personas_file = tmp_path / "personas.yaml"
    personas_file.write_text("personas:\n  - name: Alice\n    age: 30\n  - name: Bob\n    age: 40\n")
    survey_file = tmp_path / "survey.yaml"
    survey_file.write_text("instrument:\n  questions:\n    - text: What do you think?\n")
    return personas_file, survey_file


class TestSingleModelsSpecRouting:
    """A one-entry ``--models`` spec must take the standard single-model path."""

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_single_entry_models_runs_synthesis(
        self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, panel_files
    ):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary()
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()
        personas_file, survey_file = panel_files

        code = main(
            [
                "panel",
                "run",
                "--models",
                "modelA",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        captured = capsys.readouterr()
        # Synthesis ran (it never did on the old ensemble path).
        mock_synth.assert_called_once()
        assert "SYNTHESIS" in captured.out
        assert "Test synthesis summary" in captured.out
        # No "Ensemble" phrasing for a single-model run.
        assert "Ensemble complete" not in captured.out
        # The demotion is announced.
        assert "single model" in captured.err
        # End-of-run summary states synthesis status + where results live.
        assert "RUN SUMMARY" in captured.out
        assert "Synthesis: completed" in captured.out
        assert "not saved" in captured.out

    @patch("synth_panel.cli.commands.synthesize_panel")
    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_single_entry_models_json_matches_single_model_shape(
        self, mock_client_cls, mock_runtime_cls, mock_synth, capsys, panel_files
    ):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary()
        mock_runtime_cls.return_value = mock_runtime
        mock_synth.return_value = _mock_synthesis_result()
        personas_file, survey_file = panel_files

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--models",
                "modelA",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["persona_count"] == 2
        assert data["synthesis"] is not None
        assert data["synthesis"]["summary"] == "Test synthesis summary"
        # Nonzero recorded cost with token usage (regression for gh-r4 #2).
        assert data["panelist_cost"] not in (None, "$0.0000")

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_two_model_ensemble_states_synthesis_skipped(self, mock_client_cls, mock_runtime_cls, capsys, panel_files):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary()
        mock_runtime_cls.return_value = mock_runtime
        personas_file, survey_file = panel_files

        code = main(
            [
                "panel",
                "run",
                "--models",
                "modelA,modelB",
                "--skip-preflight",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "Ensemble complete" in out
        assert "Synthesis: skipped" in out
        assert "panel synthesize" in out
        # Recorded cost + panel dimensions are on stdout, not hidden.
        assert "Cost:" in out
        assert "2 model(s)" in out

    @patch("synth_panel.orchestrator.AgentRuntime")
    @patch("synth_panel.cli.commands.LLMClient")
    def test_two_model_ensemble_json_carries_synthesis_status(
        self, mock_client_cls, mock_runtime_cls, capsys, panel_files
    ):
        mock_runtime = MagicMock()
        mock_runtime.run_turn.return_value = _mock_turn_summary()
        mock_runtime_cls.return_value = mock_runtime
        personas_file, survey_file = panel_files

        code = main(
            [
                "--output-format",
                "json",
                "panel",
                "run",
                "--models",
                "modelA,modelB",
                "--skip-preflight",
                "--personas",
                str(personas_file),
                "--instrument",
                str(survey_file),
            ]
        )
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["synthesis"] is None
        assert data["synthesis_status"] == "skipped"


class TestZeroProviderCostFallback:
    """resolve_cost must not record $0 for token-consuming runs (gh-r4 #2)."""

    MODEL = "openrouter/google/gemini-2.5-flash-lite"

    def test_zero_provider_cost_with_tokens_uses_local_estimate(self):
        usage = TokenUsage(input_tokens=33_368, output_tokens=7_590, provider_reported_cost=0)
        cost = resolve_cost(usage, self.MODEL)
        assert cost.total_cost > 0

    def test_nonzero_provider_cost_still_wins(self):
        usage = TokenUsage(input_tokens=33_368, output_tokens=7_590, provider_reported_cost=Decimal("0.0041"))
        cost = resolve_cost(usage, self.MODEL)
        assert cost.total_cost == pytest.approx(0.0041)

    def test_zero_provider_cost_without_tokens_stays_zero(self):
        usage = TokenUsage(input_tokens=0, output_tokens=0, provider_reported_cost=0)
        cost = resolve_cost(usage, self.MODEL)
        assert cost.total_cost == 0

    def test_cost_gate_sees_accrued_cost_despite_zero_provider_cost(self):
        """CostGate must accrue nonzero spend so --max-cost can trip."""
        from synth_panel.orchestrator import run_panel_parallel

        personas = [{"name": f"P{i}"} for i in range(4)]
        questions = [{"text": "Q?"}]

        usage = TokenUsage(input_tokens=50_000, output_tokens=10_000, provider_reported_cost=0)
        msg = ConversationMessage(role="assistant", content=[{"type": "text", "text": "ok"}], usage=usage)
        summary = TurnSummary(assistant_messages=[msg], iterations=1, usage=usage)

        gate = CostGate(max_cost_usd=0.001, total_panelists=len(personas))
        with patch("synth_panel.orchestrator.AgentRuntime") as mock_runtime_cls:
            mock_runtime = MagicMock()
            mock_runtime.run_turn.return_value = summary
            mock_runtime_cls.return_value = mock_runtime
            run_panel_parallel(
                client=MagicMock(),
                personas=personas,
                questions=questions,
                model=self.MODEL,
                system_prompt_fn=lambda p: "system",
                question_prompt_fn=lambda q: q["text"],
                cost_gate=gate,
                max_workers=1,
            )

        snap = gate.snapshot()
        assert snap["running_cost_usd"] > 0
        # 60k tokens on a priced model projects far beyond a $0.001 cap.
        assert snap["halted"] is True


class TestCostShow:
    """`cost show <result-id>` surfaces the per-run cost breakdown."""

    def _write_result(self, tmp_path):
        data = {
            "results": [
                {
                    "persona": "Alice",
                    "model": "modelA",
                    "responses": [{"question": "Q?", "response": "A."}],
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                }
            ],
            "model": "modelA",
            "total_cost": "$1.2345",
            "panelist_cost": "$1.2000",
            "total_usage": {"input_tokens": 100, "output_tokens": 20},
            "persona_count": 1,
            "question_count": 1,
        }
        p = tmp_path / "result-test.json"
        p.write_text(json.dumps(data))
        return p

    def test_cost_show_text(self, capsys, tmp_path):
        p = self._write_result(tmp_path)
        code = main(["cost", "show", str(p)])
        assert code == 0
        out = capsys.readouterr().out
        assert "Total cost:    $1.2345" in out
        assert "Panelist cost: $1.2000" in out
        assert "Synthesis:     not run" in out

    def test_cost_show_json(self, capsys, tmp_path):
        p = self._write_result(tmp_path)
        code = main(["--output-format", "json", "cost", "show", str(p)])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["cost"]["total_cost"] == "$1.2345"
        assert data["cost"]["synthesis"]["ran"] is False

    def test_cost_show_missing_result(self, capsys):
        code = main(["cost", "show", "result-does-not-exist"])
        assert code == 1
        assert "not found" in capsys.readouterr().err


class TestHelpText:
    def test_personas_help_mentions_bundled_packs(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        # Find the panel run subparser's help text.
        import argparse

        def _find(parser, names):
            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for name, sub in action.choices.items():
                        if name == names[0]:
                            if len(names) == 1:
                                return sub
                            return _find(sub, names[1:])
            return None

        run_parser = _find(parser, ["panel", "run"])
        assert run_parser is not None
        helps = {a.option_strings[0]: (a.help or "") for a in run_parser._actions if a.option_strings}
        assert "pack list" in helps["--personas"]
        assert "YAML" in helps["--personas"]
        assert "instruments list" in helps["--instrument"]
        assert "YAML" in helps["--instrument"]
