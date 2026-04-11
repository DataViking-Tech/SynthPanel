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
        result = assign_models_to_personas(personas, spec, "sonnet")
        haiku_count = sum(1 for v in result.values() if v == "haiku")
        gemini_count = sum(1 for v in result.values() if v == "gemini")
        assert haiku_count == 5
        assert gemini_count == 5

    def test_uneven_split(self):
        personas = [{"name": f"P{i}"} for i in range(10)]
        spec = [("haiku", 0.7), ("gemini", 0.3)]
        result = assign_models_to_personas(personas, spec, "sonnet")
        haiku_count = sum(1 for v in result.values() if v == "haiku")
        gemini_count = sum(1 for v in result.values() if v == "gemini")
        assert haiku_count == 7
        assert gemini_count == 3

    def test_yaml_override_takes_precedence(self):
        personas = [
            {"name": "Alice", "model": "opus"},
            {"name": "Bob"},
            {"name": "Charlie"},
        ]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result = assign_models_to_personas(personas, spec, "sonnet")
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
        result = assign_models_to_personas(personas, spec, "sonnet")
        assert result["Alice"] == "opus"
        assert result["Bob"] == "haiku"

    def test_single_persona(self):
        personas = [{"name": "Solo"}]
        spec = [("haiku", 0.5), ("gemini", 0.5)]
        result = assign_models_to_personas(personas, spec, "sonnet")
        assert len(result) == 1
        assert result["Solo"] in ("haiku", "gemini")

    def test_all_assigned_to_single_model(self):
        personas = [{"name": f"P{i}"} for i in range(5)]
        spec = [("haiku", 1.0)]
        result = assign_models_to_personas(personas, spec, "sonnet")
        assert all(v == "haiku" for v in result.values())


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
