"""Tests for synth_panel.cost — SPEC.md Section 7."""

from synth_panel.cost import (
    ZERO_USAGE,
    BudgetError,
    BudgetGate,
    CostEstimate,
    ModelPricing,
    TokenUsage,
    UsageTracker,
    estimate_cost,
    format_summary,
    lookup_pricing,
    HAIKU_PRICING,
    SONNET_PRICING,
    OPUS_PRICING,
)

import pytest


# --- TokenUsage -----------------------------------------------------------


class TestTokenUsage:
    def test_defaults_are_zero(self):
        u = TokenUsage()
        assert u.total_tokens == 0

    def test_total_tokens(self):
        u = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=5,
        )
        assert u.total_tokens == 165

    def test_add(self):
        a = TokenUsage(input_tokens=10, output_tokens=20)
        b = TokenUsage(input_tokens=5, output_tokens=3, cache_read_input_tokens=7)
        c = a + b
        assert c.input_tokens == 15
        assert c.output_tokens == 23
        assert c.cache_read_input_tokens == 7

    def test_immutable(self):
        u = TokenUsage()
        with pytest.raises(AttributeError):
            u.input_tokens = 1  # type: ignore[misc]

    def test_roundtrip_dict(self):
        u = TokenUsage(10, 20, 30, 40)
        assert TokenUsage.from_dict(u.to_dict()) == u


# --- Pricing lookup -------------------------------------------------------


class TestPricingLookup:
    def test_haiku(self):
        p, est = lookup_pricing("claude-haiku-3.5")
        assert p is HAIKU_PRICING
        assert est is False

    def test_sonnet(self):
        p, est = lookup_pricing("claude-sonnet-4")
        assert p is SONNET_PRICING
        assert est is False

    def test_opus(self):
        p, est = lookup_pricing("claude-opus-4")
        assert p is OPUS_PRICING
        assert est is False

    def test_unknown_model_defaults(self):
        p, est = lookup_pricing("grok-3")
        assert p is SONNET_PRICING
        assert est is True

    def test_none_model_defaults(self):
        p, est = lookup_pricing(None)
        assert est is True


# --- CostEstimate ---------------------------------------------------------


class TestCostEstimate:
    def test_total_cost(self):
        c = CostEstimate(1.0, 2.0, 0.5, 0.25)
        assert c.total_cost == pytest.approx(3.75)

    def test_format_usd(self):
        c = CostEstimate(input_cost=0.015)
        assert c.format_usd() == "$0.0150"

    def test_add(self):
        a = CostEstimate(1.0, 2.0, 0.0, 0.0)
        b = CostEstimate(0.5, 0.5, 0.1, 0.1)
        c = a + b
        assert c.input_cost == pytest.approx(1.5)
        assert c.output_cost == pytest.approx(2.5)


# --- estimate_cost --------------------------------------------------------


class TestEstimateCost:
    def test_sonnet_pricing(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = estimate_cost(usage, SONNET_PRICING)
        assert cost.input_cost == pytest.approx(15.0)
        assert cost.output_cost == pytest.approx(75.0)

    def test_zero_usage(self):
        cost = estimate_cost(ZERO_USAGE)
        assert cost.total_cost == pytest.approx(0.0)

    def test_cache_tokens(self):
        usage = TokenUsage(
            cache_creation_input_tokens=1_000_000,
            cache_read_input_tokens=1_000_000,
        )
        cost = estimate_cost(usage, HAIKU_PRICING)
        assert cost.cache_creation_cost == pytest.approx(1.25)
        assert cost.cache_read_cost == pytest.approx(0.10)


# --- format_summary -------------------------------------------------------


class TestFormatSummary:
    def test_two_lines(self):
        usage = TokenUsage(100, 50, 0, 0)
        cost = estimate_cost(usage, SONNET_PRICING)
        s = format_summary("Test", usage, cost, model="claude-sonnet-4")
        lines = s.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("Test:")
        assert "total_tokens=150" in lines[0]
        assert "model=claude-sonnet-4" in lines[0]
        assert "cost breakdown:" in lines[1]

    def test_estimated_default_annotation(self):
        usage = TokenUsage(100, 50, 0, 0)
        cost = estimate_cost(usage)
        s = format_summary("X", usage, cost, is_estimated=True)
        assert "pricing=estimated-default" in s


# --- UsageTracker ---------------------------------------------------------


class TestUsageTracker:
    def test_empty_tracker(self):
        t = UsageTracker()
        assert t.turn_count == 0
        assert t.current_turn_usage is ZERO_USAGE
        assert t.cumulative_usage.total_tokens == 0

    def test_record_turns(self):
        t = UsageTracker()
        t.record_turn(TokenUsage(input_tokens=10, output_tokens=20))
        t.record_turn(TokenUsage(input_tokens=5, output_tokens=15))
        assert t.turn_count == 2
        assert t.cumulative_usage.input_tokens == 15
        assert t.cumulative_usage.output_tokens == 35
        assert t.current_turn_usage.input_tokens == 5

    def test_from_usages(self):
        usages = [
            TokenUsage(10, 20, 0, 0),
            TokenUsage(30, 40, 0, 0),
        ]
        t = UsageTracker.from_usages(usages)
        assert t.turn_count == 2
        assert t.cumulative_usage.total_tokens == 100

    def test_get_turn(self):
        t = UsageTracker()
        u = TokenUsage(42, 0, 0, 0)
        t.record_turn(u)
        assert t.get_turn(0) == u

    def test_summarise(self):
        t = UsageTracker()
        t.record_turn(TokenUsage(1000, 500, 0, 0))
        s = t.summarise(model="claude-sonnet-4")
        assert "total_tokens=1500" in s


# --- BudgetGate -----------------------------------------------------------


class TestBudgetGate:
    def test_under_budget(self):
        gate = BudgetGate(max_tokens=1000)
        gate.record_turn(TokenUsage(100, 100, 0, 0))
        gate.check(projected_tokens=100)  # Should not raise

    def test_exceeds_budget(self):
        gate = BudgetGate(max_tokens=500)
        gate.record_turn(TokenUsage(200, 200, 0, 0))
        with pytest.raises(BudgetError) as exc_info:
            gate.check(projected_tokens=200)
        assert exc_info.value.budget == 500
        assert exc_info.value.projected == 600

    def test_remaining(self):
        gate = BudgetGate(max_tokens=1000)
        gate.record_turn(TokenUsage(300, 0, 0, 0))
        assert gate.remaining == 700

    def test_remaining_clamps_to_zero(self):
        gate = BudgetGate(max_tokens=100)
        gate.record_turn(TokenUsage(200, 0, 0, 0))
        assert gate.remaining == 0

    def test_default_budget(self):
        gate = BudgetGate()
        assert gate.max_tokens == 2000
