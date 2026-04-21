"""Tests for multi-model ensemble: per-persona model routing (sp-blend)."""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from synth_panel.cli.commands import assign_models_to_personas, parse_models_spec
from synth_panel.cli.parser import build_parser
from synth_panel.cost import ZERO_USAGE
from synth_panel.llm.models import (
    CompletionResponse,
    StopReason,
    TextBlock,
)
from synth_panel.llm.models import (
    TokenUsage as LLMTokenUsage,
)
from synth_panel.orchestrator import PanelistResult, run_panel_parallel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_response(text: str = "Hello!", usage: LLMTokenUsage | None = None) -> CompletionResponse:
    return CompletionResponse(
        id="resp-1",
        model="claude-sonnet",
        content=[TextBlock(text=text)],
        stop_reason=StopReason.END_TURN,
        usage=usage or LLMTokenUsage(input_tokens=10, output_tokens=5),
    )


def _simple_system_prompt(persona: dict[str, Any]) -> str:
    return f"You are {persona.get('name', 'Anonymous')}."


def _simple_question_prompt(question: dict[str, Any]) -> str:
    if isinstance(question, dict):
        return question.get("text", str(question))
    return str(question)


def _make_mock_client(responses: list[CompletionResponse] | None = None) -> MagicMock:
    client = MagicMock()
    if responses:
        lock = threading.Lock()
        resp_list = list(responses)

        def thread_safe_send(request):
            with lock:
                if resp_list:
                    return resp_list.pop(0)
            return _make_text_response("fallback")

        client.send = MagicMock(side_effect=thread_safe_send)
    else:
        client.send = MagicMock(return_value=_make_text_response())
    return client


# ---------------------------------------------------------------------------
# Tests: parse_models_spec
# ---------------------------------------------------------------------------


class TestParseModelsSpec:
    def test_basic_two_models(self):
        result = parse_models_spec("haiku:0.5,gemini-2.5-flash:0.5")
        assert result == [("haiku", 0.5), ("gemini-2.5-flash", 0.5)]

    def test_single_model(self):
        result = parse_models_spec("haiku:1.0")
        assert result == [("haiku", 1.0)]

    def test_three_models_unequal_weights(self):
        result = parse_models_spec("haiku:0.5,sonnet:0.3,gemini:0.2")
        assert len(result) == 3
        assert result[0] == ("haiku", 0.5)
        assert result[1] == ("sonnet", 0.3)
        assert result[2] == ("gemini", 0.2)

    def test_whitespace_tolerance(self):
        result = parse_models_spec(" haiku : 0.5 , gemini : 0.5 ")
        assert result == [("haiku", 0.5), ("gemini", 0.5)]

    def test_ensemble_no_weights(self):
        """Model names without weights are treated as ensemble spec."""
        result = parse_models_spec("haiku,sonnet")
        assert result == [("haiku", 1.0), ("sonnet", 1.0)]

    def test_ensemble_single_model(self):
        result = parse_models_spec("haiku")
        assert result == [("haiku", 1.0)]

    def test_ensemble_three_models(self):
        result = parse_models_spec("haiku,sonnet,gemini")
        assert result == [("haiku", 1.0), ("sonnet", 1.0), ("gemini", 1.0)]

    def test_mixed_raises(self):
        """Mixing weighted and unweighted entries raises."""
        with pytest.raises(ValueError, match="expected 'model:weight'"):
            parse_models_spec("haiku,sonnet:0.5")

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            parse_models_spec("haiku:abc")

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            parse_models_spec("haiku:0")

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            parse_models_spec("haiku:-1")

    def test_empty_spec_raises(self):
        with pytest.raises(ValueError, match="Empty --models spec"):
            parse_models_spec("")

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="Empty model name"):
            parse_models_spec(":0.5")


# ---------------------------------------------------------------------------
# Tests: assign_models_to_personas
# ---------------------------------------------------------------------------


class TestAssignModelsToPersonas:
    def test_even_split_two_models(self):
        personas = [{"name": f"P{i}"} for i in range(10)]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        haiku_count = sum(1 for v in result.values() if v == "haiku")
        gemini_count = sum(1 for v in result.values() if v == "gemini")
        assert haiku_count == 5
        assert gemini_count == 5
        assert warnings == []

    def test_uneven_split(self):
        personas = [{"name": f"P{i}"} for i in range(10)]
        spec = [("haiku", 0.7), ("gemini", 0.3)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        haiku_count = sum(1 for v in result.values() if v == "haiku")
        gemini_count = sum(1 for v in result.values() if v == "gemini")
        assert haiku_count == 7
        assert gemini_count == 3
        assert warnings == []

    def test_yaml_override_takes_precedence(self):
        personas = [
            {"name": "Alice", "model": "opus"},
            {"name": "Bob"},
            {"name": "Charlie"},
        ]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result, _ = assign_models_to_personas(personas, spec, "sonnet")
        assert result["Alice"] == "opus"
        # Bob and Charlie get weighted assignment
        assert result["Bob"] in ("haiku", "gemini")
        assert result["Charlie"] in ("haiku", "gemini")

    def test_all_yaml_overrides(self):
        personas = [
            {"name": "Alice", "model": "opus"},
            {"name": "Bob", "model": "haiku"},
        ]
        spec = [("gemini", 1.0)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        assert result["Alice"] == "opus"
        assert result["Bob"] == "haiku"
        # No weighted assignment occurred, so no warning for gemini.
        assert warnings == []

    def test_single_persona(self):
        personas = [{"name": "Solo"}]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result, _ = assign_models_to_personas(personas, spec, "sonnet")
        assert len(result) == 1
        assert result["Solo"] in ("haiku", "gemini")

    def test_all_assigned_to_single_model(self):
        personas = [{"name": f"P{i}"} for i in range(5)]
        spec = [("haiku", 1.0)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        assert all(v == "haiku" for v in result.values())
        assert warnings == []

    def test_equal_weights_four_models_six_personas(self):
        """sp-27rz regression: every model must receive >= 1 persona.

        Pre-fix, round() banker's-rounding made the last model silently
        collect 0 personas (2+2+2+0) and disappear from per_model_results.
        Hamilton's method now guarantees each model gets at least one.
        """
        personas = [{"name": f"P{i}"} for i in range(6)]
        spec = [("haiku", 0.25), ("gpt-4o-mini", 0.25), ("gemini-flash", 0.25), ("qwen3-plus", 0.25)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        counts = {m: sum(1 for v in result.values() if v == m) for m, _ in spec}
        assert sum(counts.values()) == 6
        # Every weighted model is represented.
        for m, _ in spec:
            assert counts[m] >= 1, f"{m} received 0 personas (regression of sp-27rz)"
        # And a fair allocation: 2+2+1+1 = 6.
        assert sorted(counts.values(), reverse=True) == [2, 2, 1, 1]
        assert warnings == []

    def test_min_one_guarantee_with_skewed_weights(self):
        """Even near-zero weights get one persona when personas suffice."""
        personas = [{"name": f"P{i}"} for i in range(4)]
        spec = [("heavy", 0.9), ("light", 0.1)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        counts = {m: sum(1 for v in result.values() if v == m) for m, _ in spec}
        assert counts["heavy"] >= 1
        assert counts["light"] >= 1
        assert sum(counts.values()) == 4
        assert warnings == []

    def test_warning_when_personas_fewer_than_models(self):
        """Personas < models: cannot give every model >=1, warn loudly."""
        personas = [{"name": "Solo"}]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result, warnings = assign_models_to_personas(personas, spec, "sonnet")
        # One model gets the persona, one is absent.
        assigned_models = set(result.values())
        assert len(assigned_models) == 1
        # The absent model's name appears in the warning.
        assert len(warnings) == 1
        missing_model = ({"haiku", "gemini"} - assigned_models).pop()
        assert missing_model in warnings[0]
        assert "0 of 1" in warnings[0]

    def test_total_assignment_count_preserved(self):
        """Allocation must distribute exactly ``len(personas)`` assignments."""
        personas = [{"name": f"P{i}"} for i in range(11)]
        spec = [("a", 0.33), ("b", 0.33), ("c", 0.34)]
        result, _ = assign_models_to_personas(personas, spec, "sonnet")
        assert len(result) == 11
        assert sum(1 for v in result.values() if v in {"a", "b", "c"}) == 11

    def test_determinism_same_inputs_same_output(self):
        """Same personas + spec → same assignment. No RNG involved."""
        personas = [{"name": f"P{i}"} for i in range(7)]
        spec = [("haiku", 0.33), ("gemini", 0.33), ("sonnet", 0.33)]
        first = assign_models_to_personas(personas, spec, "fallback")
        second = assign_models_to_personas(personas, spec, "fallback")
        assert first == second

    def test_uneven_division_first_absorbs_remainder(self):
        """7 personas across 3 equal models: Hamilton's method gives the
        extra seat to the first model in spec order (deterministic tie-break
        by position after equal fractional remainders)."""
        personas = [{"name": f"P{i}"} for i in range(7)]
        spec = [("a", 0.33), ("b", 0.33), ("c", 0.33)]
        result, _ = assign_models_to_personas(personas, spec, "fallback")
        counts = {m: 0 for m in ("a", "b", "c")}
        for v in result.values():
            counts[v] += 1
        assert counts["a"] == 3  # wins the tie on position
        assert counts["b"] == 2
        assert counts["c"] == 2

    def test_unnormalized_weights_equivalent_to_normalized(self):
        """a:2,b:3 should produce the same split as a:0.4,b:0.6."""
        personas = [{"name": f"P{i}"} for i in range(10)]
        unnorm = assign_models_to_personas(personas, [("a", 2.0), ("b", 3.0)], "fallback")
        norm = assign_models_to_personas(personas, [("a", 0.4), ("b", 0.6)], "fallback")
        assert unnorm == norm


# ---------------------------------------------------------------------------
# Tests: weight-sum validation (sp-zdul)
# ---------------------------------------------------------------------------


class TestCheckWeightSum:
    def test_exactly_one(self):
        from synth_panel.cli.commands import check_weight_sum

        total, ok = check_weight_sum([("a", 0.5), ("b", 0.5)])
        assert total == pytest.approx(1.0)
        assert ok is True

    def test_within_tolerance(self):
        from synth_panel.cli.commands import check_weight_sum

        _total, ok = check_weight_sum([("a", 0.34), ("b", 0.33), ("c", 0.33)])
        assert ok is True

    def test_outside_tolerance(self):
        from synth_panel.cli.commands import check_weight_sum

        total, ok = check_weight_sum([("a", 0.3), ("b", 0.3), ("c", 0.3)])
        assert total == pytest.approx(0.9)
        assert ok is False

    def test_normalizable_but_not_unit(self):
        """a:2,b:3 sums to 5, far from 1.0 — should flag."""
        from synth_panel.cli.commands import check_weight_sum

        total, ok = check_weight_sum([("a", 2.0), ("b", 3.0)])
        assert total == pytest.approx(5.0)
        assert ok is False


class TestFormatAssignmentBreakdown:
    def test_includes_each_persona(self):
        from synth_panel.cli.commands import format_assignment_breakdown

        text = format_assignment_breakdown({"Alice": "haiku", "Bob": "gemini"})
        assert "Alice" in text
        assert "Bob" in text
        assert "haiku" in text
        assert "gemini" in text
        assert "Model assignment:" in text

    def test_includes_totals_summary(self):
        from synth_panel.cli.commands import format_assignment_breakdown

        text = format_assignment_breakdown({"A": "haiku", "B": "haiku", "C": "gemini"})
        assert "Totals:" in text
        assert "haiku=2" in text
        assert "gemini=1" in text

    def test_empty_returns_empty_string(self):
        from synth_panel.cli.commands import format_assignment_breakdown

        assert format_assignment_breakdown({}) == ""


# ---------------------------------------------------------------------------
# Tests: CLI parser --models
# ---------------------------------------------------------------------------


class TestParserModelsFlag:
    def test_models_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            ["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml", "--models", "haiku:0.5,gemini:0.5"]
        )
        assert args.models == "haiku:0.5,gemini:0.5"

    def test_models_default_none(self):
        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert args.models is None


# ---------------------------------------------------------------------------
# Tests: orchestrator per-persona model routing
# ---------------------------------------------------------------------------


class TestOrchestratorPersonaModels:
    def test_persona_models_routing(self):
        """Each panelist result records its assigned model."""
        client = _make_mock_client()
        personas = [
            {"name": "Alice"},
            {"name": "Bob"},
        ]
        questions = [{"text": "Hello?"}]
        persona_models = {"Alice": "haiku", "Bob": "gemini"}

        results, _reg, _sess = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
            persona_models=persona_models,
        )

        result_map = {r.persona_name: r for r in results}
        assert result_map["Alice"].model == "haiku"
        assert result_map["Bob"].model == "gemini"

    def test_default_model_when_no_persona_models(self):
        """Without persona_models, all panelists use the global model."""
        client = _make_mock_client()
        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Hello?"}]

        results, _reg, _sess = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
        )

        for r in results:
            assert r.model == "sonnet"

    def test_partial_persona_models(self):
        """Personas not in persona_models dict fall back to global model."""
        client = _make_mock_client()
        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Hello?"}]
        persona_models = {"Alice": "haiku"}

        results, _reg, _sess = run_panel_parallel(
            client=client,
            personas=personas,
            questions=questions,
            model="sonnet",
            system_prompt_fn=_simple_system_prompt,
            question_prompt_fn=_simple_question_prompt,
            persona_models=persona_models,
        )

        result_map = {r.persona_name: r for r in results}
        assert result_map["Alice"].model == "haiku"
        assert result_map["Bob"].model == "sonnet"


# ---------------------------------------------------------------------------
# Tests: PanelistResult model field
# ---------------------------------------------------------------------------


class TestPanelistResultModel:
    def test_model_field_default_none(self):
        pr = PanelistResult(persona_name="Test", responses=[], usage=ZERO_USAGE)
        assert pr.model is None

    def test_model_field_set(self):
        pr = PanelistResult(persona_name="Test", responses=[], usage=ZERO_USAGE, model="haiku")
        assert pr.model == "haiku"


# ---------------------------------------------------------------------------
# Tests: is_ensemble_spec
# ---------------------------------------------------------------------------


class TestIsEnsembleSpec:
    def test_ensemble_spec(self):
        from synth_panel.cli.commands import is_ensemble_spec

        assert is_ensemble_spec("haiku,sonnet") is True
        assert is_ensemble_spec("haiku") is True

    def test_weighted_spec(self):
        from synth_panel.cli.commands import is_ensemble_spec

        assert is_ensemble_spec("haiku:0.5,sonnet:0.5") is False

    def test_mixed_spec(self):
        from synth_panel.cli.commands import is_ensemble_spec

        assert is_ensemble_spec("haiku,sonnet:0.5") is False


# ---------------------------------------------------------------------------
# Tests: ensemble_run (multi-model runner loop)
# ---------------------------------------------------------------------------


class TestEnsembleRun:
    """Test ensemble_run: sequential N panel runs with per-model cost aggregation."""

    def _mock_run_parallel(self, **kwargs):
        """Side effect for mocked run_panel_parallel."""
        from synth_panel.cost import TokenUsage as CostTokenUsage

        model = kwargs["model"]
        personas = kwargs["personas"]
        results = [
            PanelistResult(
                persona_name=p["name"],
                responses=[{"question": "Q1", "response": f"answer from {model}", "error": False}],
                usage=CostTokenUsage(input_tokens=100, output_tokens=50),
                model=model,
            )
            for p in personas
        ]
        sessions = {p["name"]: MagicMock() for p in personas}
        return results, MagicMock(), sessions

    def test_runs_once_per_model(self):
        """ensemble_run should call run_panel_parallel once per model."""
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            _ensemble_run(personas, questions, models, client)

        assert mock_rpp.call_count == 2
        assert mock_rpp.call_args_list[0][1]["model"] == "haiku"
        assert mock_rpp.call_args_list[1][1]["model"] == "sonnet"

    def test_per_model_results_stored_separately(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import EnsembleResult as _ER
        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        assert isinstance(result, _ER)
        assert len(result.model_results) == 2
        assert result.model_results[0].model == "haiku"
        assert result.model_results[1].model == "sonnet"

    def test_per_model_cost_breakdown(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        assert "haiku" in result.per_model_cost
        assert "sonnet" in result.per_model_cost
        assert result.per_model_cost["haiku"].startswith("$")
        assert result.per_model_cost["sonnet"].startswith("$")

    def test_total_cost_aggregated(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        assert result.total_cost.total_cost > 0
        model_costs_sum = sum(mr.cost.total_cost for mr in result.model_results)
        assert abs(result.total_cost.total_cost - model_costs_sum) < 1e-10

    def test_total_usage_aggregated(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        # Each model: 100 input + 50 output per persona, 1 persona, 2 models
        assert result.total_usage.input_tokens == 200
        assert result.total_usage.output_tokens == 100

    def test_empty_models_raises(self):
        from synth_panel.ensemble import ensemble_run as _ensemble_run

        with pytest.raises(ValueError, match="models list must not be empty"):
            _ensemble_run([], [], [], MagicMock())

    def test_all_input_models_produce_buckets(self):
        """sp-27rz: every input model must appear as a ModelRunResult bucket.

        Regression guard for the invariant that motivated the defensive
        check at the end of ensemble_run — a silent drop here is exactly
        the bug weighted-assign already closed, and the ensemble path
        must stay covered too.
        """
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet", "gemini"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        produced = {mr.model for mr in result.model_results}
        assert produced == set(models)

    def test_panelist_results_tagged_with_model(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        for mr in result.model_results:
            for pr in mr.panelist_results:
                assert pr.model == mr.model

    def test_metadata_counts(self):
        from unittest.mock import patch as _patch

        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}, {"name": "Carol"}]
        questions = [{"text": "Q1"}, {"text": "Q2"}]
        models = ["haiku"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            result = _ensemble_run(personas, questions, models, client)

        assert result.persona_count == 3
        assert result.question_count == 2
        assert result.models == ["haiku"]

    def test_cli_ensemble_output_per_persona_usage_cost(self):
        """sp-hwe regression: CLI ensemble output must attach per-persona usage + cost.

        Prior shape stripped results to ``{persona, responses}`` only, which
        blocked ensemble workflows from answering "which model/persona burned
        the most tokens?" without re-running. The fix routes results through
        ``format_panelist_result`` so each row carries ``usage``, ``cost``,
        ``error``, and ``model``.
        """
        from unittest.mock import patch as _patch

        from synth_panel._runners import format_panelist_result
        from synth_panel.ensemble import ensemble_run as _ensemble_run

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        questions = [{"text": "Q1"}]
        models = ["haiku", "sonnet"]
        client = MagicMock()

        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = lambda **kw: self._mock_run_parallel(**kw)
            ens = _ensemble_run(personas, questions, models, client)

        per_model_results = {
            mr.model: [format_panelist_result(pr, mr.model) for pr in mr.panelist_results] for mr in ens.model_results
        }

        assert set(per_model_results) == {"haiku", "sonnet"}
        for model_name, rows in per_model_results.items():
            assert len(rows) == 2
            for row in rows:
                # Core fields every ensemble consumer depends on.
                assert row["persona"] in {"Alice", "Bob"}
                assert row["model"] == model_name
                assert row["usage"] == {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                }
                assert row["cost"].startswith("$")
                assert row["cost"] != "$0.0000"
                assert row["error"] is None

    def test_format_panelist_result_honors_provider_reported(self):
        """sp-kvpx regression: per-panelist cost must use resolve_cost.

        ``format_panelist_result`` is the common formatter used by ensemble,
        mixed-model rollups, and SDK output. It previously always quoted
        the local pricing estimate, so per-persona ``cost`` in
        ``per_model_results[*].results[i]`` drifted from the top-level
        authoritative number any time the provider bill and the local
        rate differed (or the local entry was missing and fell through
        to DEFAULT_PRICING).
        """
        from synth_panel._runners import format_panelist_result
        from synth_panel.cost import TokenUsage as CostTokenUsage

        pr = PanelistResult(
            persona_name="Alice",
            responses=[{"question": "Q1", "response": "ok", "error": False}],
            usage=CostTokenUsage(input_tokens=100, output_tokens=50, provider_reported_cost=0.1234),
            model="haiku",
        )
        row = format_panelist_result(pr, "haiku")
        # Provider-reported cost wins over any local table estimate.
        assert row["cost"] == "$0.1234"

    def test_ensemble_cost_honors_provider_reported(self):
        """sp-kvpx regression: ensemble_run per-model cost must use resolve_cost.

        Prior to the fix, ``ensemble_run`` called
        ``estimate_cost(model_usage, lookup_pricing(model))`` directly,
        bypassing sp-j3vk's provider-reported precedence. A model whose
        panelists reported ``usage.provider_reported_cost=0.42`` would
        still be billed at the local-table rate, so ``per_model_cost``
        (and therefore ``cost_breakdown`` in the public payload) drifted
        from the authoritative top-level total.
        """
        from unittest.mock import patch as _patch

        from synth_panel.cost import TokenUsage as CostTokenUsage
        from synth_panel.ensemble import ensemble_run as _ensemble_run

        def mock_with_provider_cost(**kwargs):
            model = kwargs["model"]
            personas = kwargs["personas"]
            # Each panelist reports a wildly different cost from what the
            # local pricing table would estimate for 100/50 tokens. If the
            # fix works, that's the number that propagates; if it regresses,
            # the per-model cost stays at the local-table estimate.
            results = [
                PanelistResult(
                    persona_name=p["name"],
                    responses=[{"question": "Q1", "response": "ok", "error": False}],
                    usage=CostTokenUsage(
                        input_tokens=100,
                        output_tokens=50,
                        provider_reported_cost=0.42,
                    ),
                    model=model,
                )
                for p in personas
            ]
            return results, MagicMock(), {p["name"]: MagicMock() for p in personas}

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        with _patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = mock_with_provider_cost
            ens = _ensemble_run(personas, [{"text": "Q1"}], ["haiku"], MagicMock())

        # Two panelists × $0.42 provider-reported = $0.84 authoritative.
        assert ens.model_results[0].cost.total_cost == pytest.approx(0.84)
        assert ens.per_model_cost["haiku"] == "$0.8400"
        assert ens.total_cost.total_cost == pytest.approx(0.84)


# ---------------------------------------------------------------------------
# Tests: build_ensemble_output (public JSON shape)
# ---------------------------------------------------------------------------


class TestBuildEnsembleOutput:
    """build_ensemble_output must emit the documented run_panel shape."""

    def _fixture(self):
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.ensemble import EnsembleResult, ModelRunResult

        haiku_usage = TokenUsage(input_tokens=100, output_tokens=50)
        sonnet_usage = TokenUsage(input_tokens=200, output_tokens=80)
        haiku_cost = CostEstimate(input_cost=0.01, output_cost=0.02)
        sonnet_cost = CostEstimate(input_cost=0.05, output_cost=0.04)

        haiku_mr = ModelRunResult(
            model="haiku",
            panelist_results=[
                PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q1", "response": "a1", "error": False}],
                    usage=haiku_usage,
                    model="haiku",
                )
            ],
            usage=haiku_usage,
            cost=haiku_cost,
            sessions={},
        )
        sonnet_mr = ModelRunResult(
            model="sonnet",
            panelist_results=[
                PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q1", "response": "s1", "error": False}],
                    usage=sonnet_usage,
                    model="sonnet",
                )
            ],
            usage=sonnet_usage,
            cost=sonnet_cost,
            sessions={},
        )
        return EnsembleResult(
            model_results=[haiku_mr, sonnet_mr],
            models=["haiku", "sonnet"],
            total_usage=haiku_usage + sonnet_usage,
            total_cost=haiku_cost + sonnet_cost,
            per_model_cost={
                "haiku": haiku_cost.format_usd(),
                "sonnet": sonnet_cost.format_usd(),
            },
            per_model_usage={
                "haiku": haiku_usage.to_dict(),
                "sonnet": sonnet_usage.to_dict(),
            },
            persona_count=1,
            question_count=1,
        )

    def test_top_level_keys(self):
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        assert set(out.keys()) == {
            "per_model_results",
            "cost_breakdown",
            "models",
            "total_usage",
            "warnings",
            "cost_is_estimated",
            "metadata",
        }

    def test_priced_models_have_no_cost_warnings(self):
        """sp-nn8k: when every model has an explicit pricing tier, warnings
        is empty and cost_is_estimated is False."""
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        assert out["warnings"] == []
        assert out["cost_is_estimated"] is False

    def test_unpriced_model_surfaces_fallback_warning(self):
        """sp-nn8k: an unknown ensemble model must produce a DEFAULT_PRICING
        warning and flip cost_is_estimated to True."""
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.ensemble import EnsembleResult, ModelRunResult, build_ensemble_output

        usage = TokenUsage(input_tokens=10, output_tokens=5)
        cost = CostEstimate(input_cost=0.001, output_cost=0.001)
        mystery_mr = ModelRunResult(
            model="mystery-model-v9",
            panelist_results=[
                PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q1", "response": "a1", "error": False}],
                    usage=usage,
                    model="mystery-model-v9",
                )
            ],
            usage=usage,
            cost=cost,
            sessions={},
        )
        ens = EnsembleResult(
            model_results=[mystery_mr],
            models=["mystery-model-v9"],
            total_usage=usage,
            total_cost=cost,
            per_model_cost={"mystery-model-v9": cost.format_usd()},
            per_model_usage={"mystery-model-v9": usage.to_dict()},
            persona_count=1,
            question_count=1,
        )
        out = build_ensemble_output(ens)
        assert out["cost_is_estimated"] is True
        assert len(out["warnings"]) == 1
        assert "mystery-model-v9" in out["warnings"][0]
        assert "DEFAULT_PRICING fallback" in out["warnings"][0]

    def test_per_model_results_has_results_cost_usage(self):
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        pmr = out["per_model_results"]
        assert set(pmr.keys()) == {"haiku", "sonnet"}
        for model in ("haiku", "sonnet"):
            entry = pmr[model]
            assert set(entry.keys()) == {"results", "cost", "usage"}
            assert isinstance(entry["results"], list)
            assert entry["cost"].startswith("$")
            assert isinstance(entry["usage"], dict)
            assert "input_tokens" in entry["usage"]

    def test_cost_breakdown_nested(self):
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        cb = out["cost_breakdown"]
        assert set(cb.keys()) == {"by_model", "total"}
        assert set(cb["by_model"].keys()) == {"haiku", "sonnet"}
        assert cb["total"].startswith("$")

    def test_default_formatter_renders_panelists(self):
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        results = out["per_model_results"]["haiku"]["results"]
        assert len(results) == 1
        assert results[0]["persona"] == "Alice"
        assert results[0]["model"] == "haiku"
        assert results[0]["responses"][0]["response"] == "a1"

    def test_custom_formatter_used(self):
        from synth_panel.ensemble import build_ensemble_output

        def fmt(pr, model):
            return {"n": pr.persona_name, "m": model}

        out = build_ensemble_output(self._fixture(), panelist_formatter=fmt)
        results = out["per_model_results"]["sonnet"]["results"]
        assert results == [{"n": "Alice", "m": "sonnet"}]

    def test_total_usage_summed(self):
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        # haiku: 100+50, sonnet: 200+80
        assert out["total_usage"]["input_tokens"] == 300
        assert out["total_usage"]["output_tokens"] == 130

    def test_metadata_per_model_covers_all_ensemble_models(self):
        """sp-atvc: metadata.cost.per_model must list every ensemble model.

        Regression guard for the mayor audit where a 3-model ensemble
        reported only the first model in metadata.cost.per_model, hiding
        ~6x of the real spend.
        """
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        per_model = out["metadata"]["cost"]["per_model"]
        # Both ensemble models must have their own bucket.
        assert len(per_model) == 2
        # Each entry carries tokens + cost_usd.
        for entry in per_model.values():
            assert "tokens" in entry
            assert "cost_usd" in entry
            assert entry["tokens"] > 0
            assert entry["cost_usd"] > 0

    def test_metadata_total_cost_matches_ensemble(self):
        """metadata.cost.total_cost_usd must equal the summed per-model cost."""
        from synth_panel.ensemble import build_ensemble_output

        out = build_ensemble_output(self._fixture())
        meta_cost = out["metadata"]["cost"]
        summed = sum(e["cost_usd"] for e in meta_cost["per_model"].values())
        assert summed == pytest.approx(meta_cost["total_cost_usd"])


# ---------------------------------------------------------------------------
# Tests: build_mixed_model_rollup (sp-0h9x)
# ---------------------------------------------------------------------------


class TestBuildMixedModelRollup:
    """Non-ensemble panels should still emit per_model_results + cost_breakdown.

    sp-0h9x: sp-gl9 only populated these fields in the ensemble path. Mixed-
    model panels (via ``persona_models``) and even single-model panels must
    get the same rollup shape so downstream consumers (dashboards, CI gates)
    don't have to reconstruct it from ``rounds[].results[]``.
    """

    def _pr(self, name: str, model: str | None, inp: int = 100, out: int = 50) -> PanelistResult:
        from synth_panel.cost import TokenUsage

        return PanelistResult(
            persona_name=name,
            responses=[{"question": "Q1", "response": f"{name}-a", "error": False}],
            usage=TokenUsage(input_tokens=inp, output_tokens=out),
            model=model,
        )

    def test_single_model_one_entry(self):
        from synth_panel.ensemble import build_mixed_model_rollup

        prs = [self._pr("Alice", "haiku"), self._pr("Bob", "haiku")]
        pmr, cb = build_mixed_model_rollup(prs, default_model="haiku")
        assert list(pmr.keys()) == ["haiku"]
        assert len(pmr["haiku"]["results"]) == 2
        assert pmr["haiku"]["usage"]["input_tokens"] == 200
        assert pmr["haiku"]["usage"]["output_tokens"] == 100
        assert pmr["haiku"]["cost"].startswith("$")
        assert set(cb.keys()) == {"by_model", "total"}
        assert list(cb["by_model"].keys()) == ["haiku"]
        assert cb["total"].startswith("$")

    def test_mixed_model_groups_by_pr_model(self):
        from synth_panel.ensemble import build_mixed_model_rollup

        prs = [
            self._pr("Alice", "haiku", inp=100, out=50),
            self._pr("Bob", "sonnet", inp=200, out=80),
            self._pr("Carol", "haiku", inp=50, out=20),
        ]
        pmr, cb = build_mixed_model_rollup(prs, default_model="haiku")
        assert set(pmr.keys()) == {"haiku", "sonnet"}
        # Haiku gets Alice + Carol aggregated
        assert pmr["haiku"]["usage"]["input_tokens"] == 150
        assert pmr["haiku"]["usage"]["output_tokens"] == 70
        haiku_personas = [r["persona"] for r in pmr["haiku"]["results"]]
        assert haiku_personas == ["Alice", "Carol"]
        # Sonnet gets Bob only
        assert pmr["sonnet"]["usage"]["input_tokens"] == 200
        assert [r["persona"] for r in pmr["sonnet"]["results"]] == ["Bob"]
        # Breakdown covers both models
        assert set(cb["by_model"].keys()) == {"haiku", "sonnet"}

    def test_untagged_pr_falls_back_to_default_model(self):
        from synth_panel.ensemble import build_mixed_model_rollup

        prs = [self._pr("Alice", None), self._pr("Bob", None)]
        pmr, _cb = build_mixed_model_rollup(prs, default_model="haiku")
        assert list(pmr.keys()) == ["haiku"]
        assert len(pmr["haiku"]["results"]) == 2

    def test_empty_results(self):
        from synth_panel.ensemble import build_mixed_model_rollup

        pmr, cb = build_mixed_model_rollup([], default_model="haiku")
        assert pmr == {}
        assert cb == {"by_model": {}, "total": "$0.0000"}

    def test_custom_formatter(self):
        from synth_panel.ensemble import build_mixed_model_rollup

        def fmt(pr, model):
            return {"n": pr.persona_name, "m": model}

        prs = [self._pr("Alice", "haiku"), self._pr("Bob", "sonnet")]
        pmr, _cb = build_mixed_model_rollup(prs, default_model="haiku", panelist_formatter=fmt)
        assert pmr["haiku"]["results"] == [{"n": "Alice", "m": "haiku"}]
        assert pmr["sonnet"]["results"] == [{"n": "Bob", "m": "sonnet"}]

    def test_total_equals_sum_of_by_model(self):
        """cost_breakdown.total must equal the sum of by_model values."""
        from synth_panel.cost import CostEstimate, estimate_cost, lookup_pricing
        from synth_panel.ensemble import build_mixed_model_rollup

        prs = [
            self._pr("Alice", "haiku", inp=100, out=50),
            self._pr("Bob", "sonnet", inp=200, out=80),
        ]
        _pmr, cb = build_mixed_model_rollup(prs, default_model="haiku")

        # Re-derive expected total from pricing to confirm it aggregates
        expected = CostEstimate()
        for pr in prs:
            pricing, _ = lookup_pricing(pr.model)
            expected = expected + estimate_cost(pr.usage, pricing)
        assert cb["total"] == expected.format_usd()


# ---------------------------------------------------------------------------
# Tests: CLI parser --models ensemble format
# ---------------------------------------------------------------------------


class TestParserEnsembleModels:
    def test_models_ensemble_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            ["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml", "--models", "haiku,sonnet"]
        )
        assert args.models == "haiku,sonnet"


# ---------------------------------------------------------------------------
# Tests: blend_distributions
# ---------------------------------------------------------------------------


class TestBlendDistributions:
    """Tests for the blend_distributions function."""

    def _make_ensemble_result(
        self,
        models: list[str],
        model_responses: dict[str, list[list[dict]]],
    ):
        """Build an EnsembleResult from per-model response dicts.

        model_responses maps model -> list of panelist response lists.
        Each panelist response list is a list of response dicts.
        """
        from synth_panel.cost import CostEstimate, TokenUsage
        from synth_panel.ensemble import EnsembleResult, ModelRunResult

        model_results = []
        total_usage = ZERO_USAGE
        for m in models:
            panelist_results = []
            for i, responses in enumerate(model_responses[m]):
                pr = PanelistResult(
                    persona_name=f"P{i}",
                    responses=responses,
                    usage=TokenUsage(input_tokens=10, output_tokens=5),
                    model=m,
                )
                panelist_results.append(pr)

            mr = ModelRunResult(
                model=m,
                panelist_results=panelist_results,
                usage=TokenUsage(input_tokens=20, output_tokens=10),
                cost=CostEstimate(input_cost=0.001, output_cost=0.001),
                sessions={},
            )
            model_results.append(mr)
            total_usage = total_usage + mr.usage

        return EnsembleResult(
            model_results=model_results,
            models=models,
            total_usage=total_usage,
            total_cost=CostEstimate(input_cost=0.002, output_cost=0.002),
            per_model_cost={m: "$0.00" for m in models},
            per_model_usage={m: {"input_tokens": 20, "output_tokens": 10} for m in models},
            persona_count=len(model_responses[models[0]]),
            question_count=len(model_responses[models[0]][0]) if model_responses[models[0]] else 0,
        )

    def test_equal_weights_two_models_unanimous(self):
        """When both models agree, blended distribution has 100% for one option."""
        from synth_panel.ensemble import blend_distributions

        # Both models, both panelists say "A"
        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Pick one", "response": "A"}],
                    [{"question": "Pick one", "response": "A"}],
                ],
                "gemini": [
                    [{"question": "Pick one", "response": "A"}],
                    [{"question": "Pick one", "response": "A"}],
                ],
            },
        )

        result = blend_distributions(ensemble)
        assert len(result.questions) == 1
        assert result.questions[0].distribution["A"] == pytest.approx(1.0)
        assert result.weights["haiku"] == pytest.approx(0.5)
        assert result.weights["gemini"] == pytest.approx(0.5)

    def test_equal_weights_models_disagree(self):
        """When models fully disagree, each option gets 50%."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Pick one", "response": "A"}],
                ],
                "gemini": [
                    [{"question": "Pick one", "response": "B"}],
                ],
            },
        )

        result = blend_distributions(ensemble)
        dist = result.questions[0].distribution
        assert dist["A"] == pytest.approx(0.5)
        assert dist["B"] == pytest.approx(0.5)

    def test_custom_weights(self):
        """Custom weights bias toward the heavier model."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Pick one", "response": "A"}],
                ],
                "gemini": [
                    [{"question": "Pick one", "response": "B"}],
                ],
            },
        )

        result = blend_distributions(ensemble, weights={"haiku": 0.8, "gemini": 0.2})
        dist = result.questions[0].distribution
        assert dist["A"] == pytest.approx(0.8)
        assert dist["B"] == pytest.approx(0.2)
        assert result.weights["haiku"] == pytest.approx(0.8)
        assert result.weights["gemini"] == pytest.approx(0.2)

    def test_multiple_questions(self):
        """Blending works across multiple questions."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [
                        {"question": "Q1", "response": "A"},
                        {"question": "Q2", "response": "X"},
                    ],
                ],
                "gemini": [
                    [
                        {"question": "Q1", "response": "B"},
                        {"question": "Q2", "response": "X"},
                    ],
                ],
            },
        )

        result = blend_distributions(ensemble)
        assert len(result.questions) == 2
        # Q1: split between A and B
        assert result.questions[0].distribution["A"] == pytest.approx(0.5)
        assert result.questions[0].distribution["B"] == pytest.approx(0.5)
        # Q2: unanimous X
        assert result.questions[1].distribution["X"] == pytest.approx(1.0)

    def test_intra_model_distribution(self):
        """Multiple panelists within a model create a proper distribution."""
        from synth_panel.ensemble import blend_distributions

        # haiku: 2 say A, 1 says B -> haiku dist: A=2/3, B=1/3
        # gemini: all say B -> gemini dist: B=1.0
        # equal weights: A = 0.5*2/3 + 0.5*0 = 1/3, B = 0.5*1/3 + 0.5*1 = 2/3
        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Q1", "response": "A"}],
                    [{"question": "Q1", "response": "A"}],
                    [{"question": "Q1", "response": "B"}],
                ],
                "gemini": [
                    [{"question": "Q1", "response": "B"}],
                    [{"question": "Q1", "response": "B"}],
                    [{"question": "Q1", "response": "B"}],
                ],
            },
        )

        result = blend_distributions(ensemble)
        dist = result.questions[0].distribution
        assert dist["A"] == pytest.approx(1 / 3, abs=1e-9)
        assert dist["B"] == pytest.approx(2 / 3, abs=1e-9)

    def test_three_models(self):
        """Blending works with three models."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini", "gpt"],
            model_responses={
                "haiku": [[{"question": "Q1", "response": "A"}]],
                "gemini": [[{"question": "Q1", "response": "B"}]],
                "gpt": [[{"question": "Q1", "response": "C"}]],
            },
        )

        result = blend_distributions(ensemble)
        dist = result.questions[0].distribution
        assert dist["A"] == pytest.approx(1 / 3, abs=1e-9)
        assert dist["B"] == pytest.approx(1 / 3, abs=1e-9)
        assert dist["C"] == pytest.approx(1 / 3, abs=1e-9)

    def test_per_model_distributions_preserved(self):
        """per_model field on BlendedQuestion shows each model's raw distribution."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [[{"question": "Q1", "response": "A"}]],
                "gemini": [[{"question": "Q1", "response": "B"}]],
            },
        )

        result = blend_distributions(ensemble)
        bq = result.questions[0]
        assert bq.per_model["haiku"] == {"A": 1.0}
        assert bq.per_model["gemini"] == {"B": 1.0}

    def test_error_responses_excluded(self):
        """Responses flagged as errors are excluded from distributions."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Q1", "response": "A"}],
                    [{"question": "Q1", "response": "[error: timeout]", "error": True}],
                ],
                "gemini": [
                    [{"question": "Q1", "response": "B"}],
                ],
            },
        )

        result = blend_distributions(ensemble)
        dist = result.questions[0].distribution
        # haiku: only 1 valid response (A=1.0), gemini: B=1.0
        assert dist["A"] == pytest.approx(0.5)
        assert dist["B"] == pytest.approx(0.5)

    def test_empty_ensemble_raises(self):
        """Empty model_results raises ValueError."""
        from synth_panel.ensemble import EnsembleResult, blend_distributions

        empty = EnsembleResult(
            model_results=[],
            models=[],
            total_usage=ZERO_USAGE,
            total_cost=__import__("synth_panel.cost", fromlist=["CostEstimate"]).CostEstimate(),
            per_model_cost={},
            per_model_usage={},
            persona_count=0,
            question_count=0,
        )
        with pytest.raises(ValueError, match="no model results"):
            blend_distributions(empty)

    def test_structured_response_extraction(self):
        """Structured (dict) responses are handled correctly."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Q1", "response": {"answer": "Strongly Agree"}}],
                ],
                "gemini": [
                    [{"question": "Q1", "response": {"answer": "Disagree"}}],
                ],
            },
        )

        result = blend_distributions(ensemble)
        dist = result.questions[0].distribution
        assert "Strongly Agree" in dist
        assert "Disagree" in dist
        assert dist["Strongly Agree"] == pytest.approx(0.5)
        assert dist["Disagree"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: CLI parser --blend flag
# ---------------------------------------------------------------------------


class TestParserBlendFlag:
    def test_blend_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "panel",
                "run",
                "--personas",
                "p.yaml",
                "--instrument",
                "i.yaml",
                "--models",
                "haiku:0.5,gemini:0.5",
                "--blend",
            ]
        )
        assert args.blend is True

    def test_blend_default_false(self):
        parser = build_parser()
        args = parser.parse_args(["panel", "run", "--personas", "p.yaml", "--instrument", "i.yaml"])
        assert args.blend is False


# ---------------------------------------------------------------------------
# Tests: _match_to_option
# ---------------------------------------------------------------------------


class TestMatchToOption:
    """Tests for the _match_to_option helper."""

    def test_exact_match_case_insensitive(self):
        from synth_panel.ensemble import _match_to_option

        assert _match_to_option("Fully remote", ["Fully remote", "Hybrid 3 days"]) == "Fully remote"
        assert _match_to_option("fully remote", ["Fully remote", "Hybrid 3 days"]) == "Fully remote"
        assert _match_to_option("FULLY REMOTE", ["Fully remote", "Hybrid 3 days"]) == "Fully remote"

    def test_option_contained_in_response(self):
        from synth_panel.ensemble import _match_to_option

        options = ["Fully remote", "Hybrid 3 days", "In office"]
        assert _match_to_option("I'd definitely prefer fully remote work", options) == "Fully remote"
        assert _match_to_option("I think hybrid 3 days is best for me", options) == "Hybrid 3 days"

    def test_longest_match_wins(self):
        from synth_panel.ensemble import _match_to_option

        options = ["remote", "Fully remote"]
        assert _match_to_option("I want fully remote work", options) == "Fully remote"

    def test_response_contained_in_option(self):
        from synth_panel.ensemble import _match_to_option

        options = ["Fully remote work schedule", "Hybrid 3 days per week"]
        assert _match_to_option("hybrid 3 days", options) == "Hybrid 3 days per week"

    def test_no_match_returns_original(self):
        from synth_panel.ensemble import _match_to_option

        options = ["Fully remote", "Hybrid 3 days"]
        assert _match_to_option("Something else entirely", options) == "Something else entirely"

    def test_empty_options_returns_original(self):
        from synth_panel.ensemble import _match_to_option

        assert _match_to_option("anything", []) == "anything"


# ---------------------------------------------------------------------------
# Tests: blend_distributions with options
# ---------------------------------------------------------------------------


class TestBlendDistributionsWithOptions(TestBlendDistributions):
    """Tests for blend_distributions when questions define options."""

    def test_options_normalize_responses(self):
        """Responses are matched to defined options before aggregation."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Work preference?", "response": "I prefer fully remote work"}],
                    [{"question": "Work preference?", "response": "Hybrid 3 days sounds good"}],
                ],
                "gemini": [
                    [{"question": "Work preference?", "response": "fully remote for me"}],
                    [{"question": "Work preference?", "response": "I'd choose in office"}],
                ],
            },
        )
        questions = [
            {"text": "Work preference?", "options": ["Fully remote", "Hybrid 3 days", "In office"]},
        ]

        result = blend_distributions(ensemble, questions=questions)
        dist = result.questions[0].distribution
        # haiku: Fully remote=0.5, Hybrid 3 days=0.5
        # gemini: Fully remote=0.5, In office=0.5
        # blended (equal weight): Fully remote=0.5, Hybrid 3 days=0.25, In office=0.25
        assert "Fully remote" in dist
        assert "Hybrid 3 days" in dist
        assert "In office" in dist
        assert dist["Fully remote"] == pytest.approx(0.5)
        assert dist["Hybrid 3 days"] == pytest.approx(0.25)
        assert dist["In office"] == pytest.approx(0.25)

    def test_structured_response_with_options(self):
        """Structured responses (pick_one) are matched to defined options."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [
                    [{"question": "Q1", "response": {"choice": "fully remote", "reasoning": "I like it"}}],
                ],
                "gemini": [
                    [{"question": "Q1", "response": {"choice": "Hybrid 3 days", "reasoning": "balance"}}],
                ],
            },
        )
        questions = [
            {"text": "Q1", "options": ["Fully remote", "Hybrid 3 days", "In office"]},
        ]

        result = blend_distributions(ensemble, questions=questions)
        dist = result.questions[0].distribution
        assert "Fully remote" in dist
        assert "Hybrid 3 days" in dist
        assert dist["Fully remote"] == pytest.approx(0.5)
        assert dist["Hybrid 3 days"] == pytest.approx(0.5)

    def test_no_options_preserves_behavior(self):
        """Questions without options keep raw response values."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [[{"question": "Q1", "response": "A long answer"}]],
                "gemini": [[{"question": "Q1", "response": "B long answer"}]],
            },
        )
        questions = [{"text": "Q1"}]  # no options

        result = blend_distributions(ensemble, questions=questions)
        dist = result.questions[0].distribution
        assert "A long answer" in dist
        assert "B long answer" in dist

    def test_questions_none_preserves_behavior(self):
        """When questions is None, behavior is unchanged."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku", "gemini"],
            model_responses={
                "haiku": [[{"question": "Q1", "response": "A"}]],
                "gemini": [[{"question": "Q1", "response": "B"}]],
            },
        )

        result = blend_distributions(ensemble, questions=None)
        dist = result.questions[0].distribution
        assert dist["A"] == pytest.approx(0.5)
        assert dist["B"] == pytest.approx(0.5)

    def test_mixed_questions_some_with_options(self):
        """Only questions with options get matching; others pass through."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku"],
            model_responses={
                "haiku": [
                    [
                        {"question": "Q1", "response": "I like fully remote"},
                        {"question": "Q2", "response": "Free-form answer here"},
                    ],
                ],
            },
        )
        questions = [
            {"text": "Q1", "options": ["Fully remote", "Hybrid", "In office"]},
            {"text": "Q2"},  # no options
        ]

        result = blend_distributions(ensemble, questions=questions)
        assert result.questions[0].distribution == {"Fully remote": pytest.approx(1.0)}
        assert result.questions[1].distribution == {"Free-form answer here": pytest.approx(1.0)}

    def test_unmatched_response_passes_through(self):
        """Responses that don't match any option are kept as-is."""
        from synth_panel.ensemble import blend_distributions

        ensemble = self._make_ensemble_result(
            models=["haiku"],
            model_responses={
                "haiku": [
                    [{"question": "Q1", "response": "Something completely different"}],
                    [{"question": "Q1", "response": "Fully remote"}],
                ],
            },
        )
        questions = [
            {"text": "Q1", "options": ["Fully remote", "Hybrid"]},
        ]

        result = blend_distributions(ensemble, questions=questions)
        dist = result.questions[0].distribution
        assert "Fully remote" in dist
        assert "Something completely different" in dist
