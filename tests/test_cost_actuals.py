"""Tests for OpenRouter cost-actuals plumbing (sy-ye1).

The OpenRouter provider already passes ``provider_reported_cost`` through
:class:`TokenUsage`. v1.4.0 surfaces that value (and the local pricing-table
estimate) as explicit ``cost_actual_usd`` / ``cost_estimated_usd`` fields on
:class:`EnsembleResult`, :class:`ModelRunResult`, and :class:`SynthesisResult`
so downstream budget-reconciliation consumers don't have to crack open raw
usage objects to tell which kind of number they're looking at.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from synth_panel.cost import (
    CostEstimate,
    TokenUsage,
    actual_cost_usd,
    local_estimate_usd,
)
from synth_panel.ensemble import (
    EnsembleResult,
    ModelRunResult,
    build_ensemble_output,
    ensemble_run,
)
from synth_panel.orchestrator import PanelistResult

# ---------------------------------------------------------------------------
# cost.py helpers
# ---------------------------------------------------------------------------


class TestCostHelpers:
    def test_local_estimate_usd_uses_local_table_not_provider(self):
        """``local_estimate_usd`` always uses the local pricing table.

        Even when ``provider_reported_cost`` is set on the usage object,
        the helper must ignore it and derive the cost from the local
        table — this powers the public ``cost_estimated_usd`` field that
        is meant to be the audit-grade "what we'd have computed locally"
        number.
        """
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            provider_reported_cost=Decimal("99.99"),
        )
        # haiku rates: $1/M in, $5/M out → exactly $6.00 from local table
        estimate = local_estimate_usd(usage, "haiku")
        assert estimate == pytest.approx(6.0, abs=1e-9)

    def test_actual_cost_usd_returns_provider_value_when_present(self):
        usage = TokenUsage(provider_reported_cost=Decimal("0.4242"))
        assert actual_cost_usd(usage) == pytest.approx(0.4242)

    def test_actual_cost_usd_returns_none_when_absent(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert actual_cost_usd(usage) is None


# ---------------------------------------------------------------------------
# ModelRunResult / EnsembleResult properties
# ---------------------------------------------------------------------------


def _panelist(persona: str, model: str, usage: TokenUsage) -> PanelistResult:
    return PanelistResult(
        persona_name=persona,
        responses=[{"question": "Q1", "response": "ok", "error": False}],
        usage=usage,
        model=model,
    )


class TestModelRunResultProperties:
    def test_actual_populated_when_provider_returns_cost(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            provider_reported_cost=Decimal("0.42"),
        )
        mr = ModelRunResult(
            model="haiku",
            panelist_results=[_panelist("Alice", "haiku", usage)],
            usage=usage,
            cost=CostEstimate(),
            sessions={},
        )
        assert mr.cost_actual_usd == pytest.approx(0.42)
        # Estimated from local haiku table for 100/50 tokens:
        # 100 * 1/1M + 50 * 5/1M = 0.0001 + 0.00025 = 0.00035
        assert mr.cost_estimated_usd == pytest.approx(0.00035, abs=1e-9)

    def test_actual_none_when_provider_silent(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        mr = ModelRunResult(
            model="haiku",
            panelist_results=[_panelist("Alice", "haiku", usage)],
            usage=usage,
            cost=CostEstimate(),
            sessions={},
        )
        assert mr.cost_actual_usd is None
        assert mr.cost_estimated_usd > 0


class TestEnsembleResultProperties:
    def test_ensemble_cost_actuals_surface_through_ensemble_run(self):
        """End-to-end: a panelist with provider_reported_cost flows up to
        ``EnsembleResult.cost_actual_usd`` and the per-model breakdown.
        """

        def mock_run_panel(**kwargs):
            model = kwargs["model"]
            personas = kwargs["personas"]
            results = [
                _panelist(
                    p["name"],
                    model,
                    TokenUsage(
                        input_tokens=200,
                        output_tokens=100,
                        provider_reported_cost=Decimal("0.10"),
                    ),
                )
                for p in personas
            ]
            return results, MagicMock(), {p["name"]: MagicMock() for p in personas}

        personas = [{"name": "Alice"}, {"name": "Bob"}]
        with patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = mock_run_panel
            ens = ensemble_run(personas, [{"text": "Q1"}], ["haiku"], MagicMock())

        # Two panelists × $0.10 provider-reported = $0.20 actual.
        assert ens.cost_actual_usd == pytest.approx(0.20)
        # Local estimate for 2*(200 in + 100 out) at haiku rates
        # ($1/M in, $5/M out): 2*(200*1/1M + 100*5/1M) = 2*(0.0002+0.0005) = 0.0014
        assert ens.cost_estimated_usd == pytest.approx(0.0014, abs=1e-9)

        breakdown = ens.per_model_breakdown
        assert len(breakdown) == 1
        entry = breakdown[0]
        assert entry["model"] == "haiku"
        assert entry["tokens_prompt"] == 400
        assert entry["tokens_completion"] == 200
        assert entry["tokens_total"] == 600
        assert entry["cost_actual_usd"] == pytest.approx(0.20)
        assert entry["cost_estimated_usd"] == pytest.approx(0.0014, abs=1e-9)

    def test_ensemble_actual_none_when_no_provider_reports_cost(self):
        """Direct-provider ensemble (Anthropic, OpenAI, Google) — none of
        these return per-call dollars, so the aggregate actual is None
        and consumers know to fall back to the estimate."""

        def mock_run_panel(**kwargs):
            model = kwargs["model"]
            personas = kwargs["personas"]
            results = [
                _panelist(
                    p["name"],
                    model,
                    TokenUsage(input_tokens=200, output_tokens=100),
                )
                for p in personas
            ]
            return results, MagicMock(), {p["name"]: MagicMock() for p in personas}

        with patch("synth_panel.ensemble.run_panel_parallel") as mock_rpp:
            mock_rpp.side_effect = mock_run_panel
            ens = ensemble_run([{"name": "Alice"}], [{"text": "Q1"}], ["haiku"], MagicMock())

        assert ens.cost_actual_usd is None
        assert ens.cost_estimated_usd > 0
        breakdown = ens.per_model_breakdown
        assert breakdown[0]["cost_actual_usd"] is None
        assert breakdown[0]["cost_estimated_usd"] > 0


# ---------------------------------------------------------------------------
# build_ensemble_output payload shape
# ---------------------------------------------------------------------------


class TestBuildEnsembleOutputSurfaces:
    def _make_ensemble(self, provider_cost: Decimal | None) -> EnsembleResult:
        usage = TokenUsage(
            input_tokens=200,
            output_tokens=100,
            provider_reported_cost=provider_cost,
        )
        mr = ModelRunResult(
            model="haiku",
            panelist_results=[_panelist("Alice", "haiku", usage)],
            usage=usage,
            cost=CostEstimate(),
            sessions={},
        )
        return EnsembleResult(
            model_results=[mr],
            models=["haiku"],
            total_usage=usage,
            total_cost=CostEstimate(),
            per_model_cost={"haiku": "$0.0000"},
            per_model_usage={"haiku": usage.to_dict()},
            persona_count=1,
            question_count=1,
        )

    def test_output_surfaces_cost_actual_and_estimate(self):
        ens = self._make_ensemble(provider_cost=Decimal("0.0500"))
        out = build_ensemble_output(ens)
        assert "cost_actual_usd" in out
        assert "cost_estimated_usd" in out
        assert "per_model_breakdown" in out
        assert out["cost_actual_usd"] == pytest.approx(0.05)
        assert out["cost_estimated_usd"] > 0
        assert isinstance(out["per_model_breakdown"], list)
        assert out["per_model_breakdown"][0]["cost_actual_usd"] == pytest.approx(0.05)

    def test_output_actual_is_null_for_direct_provider(self):
        ens = self._make_ensemble(provider_cost=None)
        out = build_ensemble_output(ens)
        assert out["cost_actual_usd"] is None
        assert out["cost_estimated_usd"] > 0
        assert out["per_model_breakdown"][0]["cost_actual_usd"] is None


# ---------------------------------------------------------------------------
# SynthesisResult exposes cost actuals
# ---------------------------------------------------------------------------


class TestSynthesisResultCostActuals:
    def test_synthesis_actual_populated_when_judge_reports_cost(self):
        from synth_panel.synthesis import SynthesisResult

        sr = SynthesisResult(
            summary="x",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="y",
            usage=TokenUsage(
                input_tokens=500,
                output_tokens=200,
                provider_reported_cost=Decimal("0.0123"),
            ),
            cost=CostEstimate(),
            model="haiku",
        )
        assert sr.cost_actual_usd == pytest.approx(0.0123)
        assert sr.cost_estimated_usd > 0

        d = sr.to_dict()
        assert d["cost_actual_usd"] == pytest.approx(0.0123)
        assert "cost_estimated_usd" in d
        # Existing field preserved for backward compatibility.
        assert "cost" in d

    def test_synthesis_actual_none_when_judge_silent(self):
        from synth_panel.synthesis import SynthesisResult

        sr = SynthesisResult(
            summary="x",
            themes=[],
            agreements=[],
            disagreements=[],
            surprises=[],
            recommendation="y",
            usage=TokenUsage(input_tokens=500, output_tokens=200),
            cost=CostEstimate(),
            model="haiku",
        )
        assert sr.cost_actual_usd is None
        assert sr.cost_estimated_usd > 0

        d = sr.to_dict()
        assert d["cost_actual_usd"] is None
        assert d["cost_estimated_usd"] > 0


# ---------------------------------------------------------------------------
# OpenRouter parser end-to-end: usage.cost → CompletionResponse.usage
# ---------------------------------------------------------------------------


class TestOpenRouterCostParse:
    def test_or_usage_cost_lands_on_token_usage(self):
        """Smoke test: the OR response shape with ``usage.cost`` parses
        into ``TokenUsage.provider_reported_cost``. This is the inbound
        plumbing the rest of sy-ye1 depends on, so guard it explicitly.
        """
        from synth_panel.llm.providers._openai_format import parse_openai_response

        data: dict[str, Any] = {
            "id": "gen-1",
            "model": "anthropic/claude-haiku",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "cost": 0.0042,
            },
        }
        resp = parse_openai_response(data, "anthropic/claude-haiku")
        assert resp.usage.provider_reported_cost is not None
        assert float(resp.usage.provider_reported_cost) == pytest.approx(0.0042)
