"""Tests for synth_panel.cost — SPEC.md Section 7."""

from __future__ import annotations

from decimal import Decimal

import pytest

from synth_panel.cost import (
    DEEPSEEK_CHAT_PRICING,
    DEEPSEEK_V3_2_EXP_PRICING,
    DEEPSEEK_V3_2_PRICING,
    DEEPSEEK_V3_2_SPECIALE_PRICING,
    GEMINI_FLASH_LITE_PRICING,
    GEMINI_FLASH_PRICING,
    GEMINI_PRO_PRICING,
    GPT_4_1_MINI_PRICING,
    GPT_4O_MINI_PRICING,
    GPT_4O_PRICING,
    GPT_5_MINI_PRICING,
    HAIKU_PRICING,
    LLAMA_3_3_70B_PRICING,
    MISTRAL_MEDIUM_PRICING,
    OPUS_PRICING,
    QWEN3_6_PLUS_PRICING,
    QWEN3_MAX_PRICING,
    QWEN3_PLUS_PRICING,
    SONNET_PRICING,
    ZERO_USAGE,
    BudgetError,
    BudgetGate,
    CostEstimate,
    TokenUsage,
    UsageTracker,
    aggregate_per_model,
    build_cost_fallback_warnings,
    collect_estimated_models,
    estimate_cost,
    format_summary,
    lookup_pricing,
    lookup_pricing_by_provider,
)

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
        _p, est = lookup_pricing(None)
        assert est is True

    def test_gpt_5_mini(self):
        """sp-loil: gpt-5-mini must not silently fall back to SONNET rates."""
        p, est = lookup_pricing("gpt-5-mini")
        assert p is GPT_5_MINI_PRICING
        assert est is False

    def test_gpt_5_mini_openrouter_prefix(self):
        """Provider strings like 'openrouter/openai/gpt-5-mini' must resolve
        via substring match to GPT_5_MINI_PRICING, not DEFAULT (Sonnet)."""
        p, est = lookup_pricing("openrouter/openai/gpt-5-mini")
        assert p is GPT_5_MINI_PRICING
        assert est is False

    def test_gpt_5_mini_realistic_cost(self):
        """Regression for sp-loil: 13087/4846 tokens must cost ~$0.013, not ~$0.56."""
        usage = TokenUsage(input_tokens=13087, output_tokens=4846)
        pricing, _ = lookup_pricing("openrouter/openai/gpt-5-mini")
        cost = estimate_cost(usage, pricing)
        assert cost.total_cost == pytest.approx(0.01296, abs=1e-4)

    @pytest.mark.parametrize(
        "model, expected",
        [
            # sp-5ggf: common OpenRouter models must resolve to their own
            # rates, not fall through to DEFAULT ($15/$75 Opus-era rate).
            ("gpt-4o-mini", GPT_4O_MINI_PRICING),
            ("openrouter/openai/gpt-4o-mini", GPT_4O_MINI_PRICING),
            ("openai/gpt-4o", GPT_4O_PRICING),
            ("openrouter/openai/gpt-4.1-mini", GPT_4_1_MINI_PRICING),
            ("openrouter/deepseek/deepseek-chat-v3", DEEPSEEK_CHAT_PRICING),
            ("openrouter/deepseek/deepseek-chat-v3.1", DEEPSEEK_CHAT_PRICING),
            ("openrouter/qwen/qwen3-plus", QWEN3_PLUS_PRICING),
            ("openrouter/mistralai/mistral-medium-3", MISTRAL_MEDIUM_PRICING),
            ("openrouter/meta-llama/llama-3.3-70b-instruct", LLAMA_3_3_70B_PRICING),
            # sp-oshf: newer OpenRouter IDs with version suffixes the old
            # substring keys could not match.
            ("openrouter/deepseek/deepseek-v3.2", DEEPSEEK_V3_2_PRICING),
            ("deepseek-v3.2", DEEPSEEK_V3_2_PRICING),
            ("openrouter/deepseek/deepseek-v3.2-speciale", DEEPSEEK_V3_2_SPECIALE_PRICING),
            ("openrouter/deepseek/deepseek-v3.2-exp", DEEPSEEK_V3_2_EXP_PRICING),
            ("openrouter/qwen/qwen3.6-plus", QWEN3_6_PLUS_PRICING),
            ("qwen3.6-plus", QWEN3_6_PLUS_PRICING),
            ("openrouter/qwen/qwen3-max", QWEN3_MAX_PRICING),
            # sp-9gcm: bare ``deepseek-v3`` (OpenRouter's short route that
            # resolves to v3.2) must hit the v3.2 entry, not DEFAULT_PRICING.
            ("deepseek-v3", DEEPSEEK_V3_2_PRICING),
            ("openrouter/deepseek/deepseek-v3", DEEPSEEK_V3_2_PRICING),
            # sp-9gcm: ``gemini-flash-lite`` must win over the generic
            # ``gemini`` substring (which used to overstate cost by ~41%).
            ("gemini-flash-lite", GEMINI_FLASH_LITE_PRICING),
            ("openrouter/google/gemini-2.5-flash-lite", GEMINI_FLASH_LITE_PRICING),
        ],
    )
    def test_openrouter_common_models(self, model, expected):
        p, est = lookup_pricing(model)
        assert p is expected
        assert est is False

    def test_deepseek_v3_bare_not_default(self):
        """sp-9gcm regression: ``deepseek-v3`` must not fall through to
        DEFAULT_PRICING (SONNET rates), which would overstate cost ~15x."""
        p, est = lookup_pricing("deepseek-v3")
        assert p is DEEPSEEK_V3_2_PRICING
        assert p is not SONNET_PRICING
        assert est is False

    def test_gemini_flash_lite_precedence_over_gemini(self):
        """sp-9gcm regression: ``gemini-flash-lite`` must win over the bare
        ``gemini`` key — order in _PRICING_TABLE matters, same pattern as
        ``gpt-4o-mini`` vs ``gpt-4o``."""
        p, _ = lookup_pricing("gemini-flash-lite")
        assert p is GEMINI_FLASH_LITE_PRICING
        assert p is not GEMINI_FLASH_PRICING

    def test_deepseek_v3_2_variant_precedence(self):
        """sp-oshf: ``-speciale`` and ``-exp`` suffixes must win over the bare
        ``deepseek-v3.2`` key — order in _PRICING_TABLE matters."""
        p, _ = lookup_pricing("openrouter/deepseek/deepseek-v3.2-speciale")
        assert p is DEEPSEEK_V3_2_SPECIALE_PRICING
        assert p is not DEEPSEEK_V3_2_PRICING
        p, _ = lookup_pricing("openrouter/deepseek/deepseek-v3.2-exp")
        assert p is DEEPSEEK_V3_2_EXP_PRICING
        assert p is not DEEPSEEK_V3_2_PRICING

    def test_gpt_4o_mini_substring_precedence_over_gpt_4o(self):
        """``gpt-4o-mini`` must win over ``gpt-4o`` — order in _PRICING_TABLE matters."""
        p, _ = lookup_pricing("openai/gpt-4o-mini")
        assert p is GPT_4O_MINI_PRICING
        assert p is not GPT_4O_PRICING

    def test_gpt_4o_mini_realistic_cost_not_sonnet(self):
        """Regression for sp-5ggf: 11144in/1655out for gpt-4o-mini must cost
        ~$0.0027, not ~$0.2913 (the Opus-rate fallthrough observed in the wild)."""
        usage = TokenUsage(input_tokens=11144, output_tokens=1655)
        pricing, _ = lookup_pricing("openrouter/openai/gpt-4o-mini")
        cost = estimate_cost(usage, pricing)
        # 11144 * 0.15/M + 1655 * 0.60/M = 0.001672 + 0.000993 = 0.002665
        assert cost.total_cost == pytest.approx(0.002665, abs=1e-5)


# --- sp-nn8k: cost fallback warnings -------------------------------------


class TestCostFallbackWarnings:
    """build_cost_fallback_warnings / collect_estimated_models."""

    def test_no_models_returns_empty(self):
        assert build_cost_fallback_warnings([]) == []
        assert collect_estimated_models([]) == []

    def test_skips_none_and_blank(self):
        assert build_cost_fallback_warnings([None, "", "sonnet"]) == []

    def test_priced_model_produces_no_warning(self):
        assert build_cost_fallback_warnings(["claude-sonnet-4-6"]) == []

    def test_unpriced_model_produces_one_warning(self):
        warnings = build_cost_fallback_warnings(["mystery-model-v9"])
        assert len(warnings) == 1
        w = warnings[0]
        assert "mystery-model-v9" in w
        assert "DEFAULT_PRICING fallback" in w
        assert "explicit pricing" in w

    def test_deduplicates_same_model(self):
        warnings = build_cost_fallback_warnings(["novel-x", "novel-x", "novel-x"])
        assert len(warnings) == 1

    def test_preserves_first_seen_order(self):
        warnings = build_cost_fallback_warnings(["alpha-unknown", "beta-unknown"])
        assert "'alpha-unknown'" in warnings[0]
        assert "'beta-unknown'" in warnings[1]

    def test_mixed_priced_and_estimated(self):
        """Only the unpriced model(s) surface — priced models are silent."""
        warnings = build_cost_fallback_warnings(["sonnet", "my-local-7b", "haiku", "exotic-13b"])
        assert len(warnings) == 2
        joined = "\n".join(warnings)
        assert "my-local-7b" in joined
        assert "exotic-13b" in joined
        assert "sonnet" not in joined
        assert "haiku" not in joined

    def test_collect_estimated_models_matches_warnings(self):
        """collect_estimated_models returns the same identifiers the warnings mention."""
        models = ["sonnet", "novel-a", "haiku", "novel-b", "novel-a"]
        estimated = collect_estimated_models(models)
        assert estimated == ["novel-a", "novel-b"]


# --- Provider-string pricing lookup --------------------------------------


class TestProviderLookup:
    """Cover the 7 R-observed `config.provider` formats plus negatives.

    The provider string is the synthbench-side identifier that pairs a
    bucket prefix (delivery channel) with an inner model string. The
    helper must (a) parse the bucket + inner correctly, (b) refuse to
    fall back to SONNET pricing for unrecognised inner strings, and
    (c) treat baselines / ensembles / self-hosted as unpriced.
    """

    @pytest.mark.parametrize(
        "provider, expected",
        [
            # 7 positive cases — valid bucket + recognised inner.
            ("synthpanel/claude-sonnet-4", SONNET_PRICING),
            ("synthpanel/claude-sonnet-4 t=0.85 tpl=current", SONNET_PRICING),
            ("synthpanel/claude-haiku-4-5 t=0.85 profile=foo tpl=minimal", HAIKU_PRICING),
            ("openrouter/anthropic/claude-haiku-4-5", HAIKU_PRICING),
            ("openrouter/google/gemini-2.5-flash", GEMINI_FLASH_PRICING),
            ("raw-anthropic/claude-opus-4-6", OPUS_PRICING),
            ("raw-gemini/gemini-2.5-pro", GEMINI_PRO_PRICING),
            # sp-loil: gpt-5-mini via OpenRouter must price at its own rate.
            ("openrouter/openai/gpt-5-mini", GPT_5_MINI_PRICING),
            ("raw-openai/gpt-5-mini", GPT_5_MINI_PRICING),
            # sp-5ggf: common OpenRouter models must resolve via provider lookup.
            ("openrouter/openai/gpt-4o-mini", GPT_4O_MINI_PRICING),
            ("raw-openai/gpt-4o", GPT_4O_PRICING),
            ("openrouter/openai/gpt-4.1-mini", GPT_4_1_MINI_PRICING),
            ("openrouter/deepseek/deepseek-chat-v3", DEEPSEEK_CHAT_PRICING),
            ("openrouter/qwen/qwen3-plus", QWEN3_PLUS_PRICING),
            ("openrouter/mistralai/mistral-medium-3", MISTRAL_MEDIUM_PRICING),
            ("openrouter/meta-llama/llama-3.3-70b-instruct", LLAMA_3_3_70B_PRICING),
            # sp-oshf: newer version-suffixed IDs via provider-string route.
            ("openrouter/deepseek/deepseek-v3.2", DEEPSEEK_V3_2_PRICING),
            ("openrouter/deepseek/deepseek-v3.2-speciale", DEEPSEEK_V3_2_SPECIALE_PRICING),
            ("openrouter/deepseek/deepseek-v3.2-exp", DEEPSEEK_V3_2_EXP_PRICING),
            ("openrouter/qwen/qwen3.6-plus", QWEN3_6_PLUS_PRICING),
            ("openrouter/qwen/qwen3-max", QWEN3_MAX_PRICING),
        ],
    )
    def test_positive_lookup(self, provider: str, expected: object) -> None:
        pricing, is_estimated = lookup_pricing_by_provider(provider)
        assert pricing is expected
        assert is_estimated is False

    @pytest.mark.parametrize(
        "provider",
        [
            # 5 negative cases — unpriced by design.
            "ollama/llama3",
            "random-baseline",
            "majority-baseline",
            "population-average-baseline",
            "ensemble/3-model-blend",
            # 1 edge case — valid bucket but inner has no priced match.
            # Refuses to silently fall back to SONNET.
            "raw-openai/gpt-5-unreleased",
        ],
    )
    def test_negative_lookup(self, provider: str) -> None:
        pricing, is_estimated = lookup_pricing_by_provider(provider)
        assert pricing is None
        assert is_estimated is False


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
        # sp-cxyb: Sonnet 4.5 rates are $3/M input, $15/M output.
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = estimate_cost(usage, SONNET_PRICING)
        assert cost.input_cost == pytest.approx(3.0)
        assert cost.output_cost == pytest.approx(15.0)

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


# --- resolve_cost (sp-j3vk) -----------------------------------------------


class TestResolveCost:
    """sp-j3vk: provider-reported cost must override local pricing table."""

    def test_falls_back_to_local_when_provider_absent(self):
        from synth_panel.cost import resolve_cost

        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
        cost = resolve_cost(usage, "claude-haiku-3.5")
        assert cost.total_cost == pytest.approx(1.0)

    def test_provider_reported_overrides_local(self):
        """Stale local table must not corrupt the actual billed cost."""
        from synth_panel.cost import resolve_cost

        # Local table would compute this at haiku rates ($1/M input). The
        # provider says we were billed $0.42 (BYOK discount, promo, etc.).
        # resolve_cost must return the provider value, not the local estimate.
        usage = TokenUsage(
            input_tokens=1_000_000,
            output_tokens=0,
            provider_reported_cost=0.42,
        )
        cost = resolve_cost(usage, "claude-haiku-3.5")
        assert cost.total_cost == pytest.approx(0.42)

    def test_provider_reported_preserved_when_local_would_be_zero(self):
        """ollama/self-hosted case — provider reports cost with zero tokens."""
        from synth_panel.cost import resolve_cost

        usage = TokenUsage(provider_reported_cost=0.05)
        cost = resolve_cost(usage, "ollama/llama3")
        assert cost.total_cost == pytest.approx(0.05)

    def test_divergence_warning_fires_above_20_percent(self, caplog):
        """sp-j3vk: when local estimate wildly disagrees with provider, warn."""
        from synth_panel.cost import resolve_cost

        # Local: 1M input × $1/M = $1.00. Provider: $0.10 (90% divergence).
        usage = TokenUsage(
            input_tokens=1_000_000,
            provider_reported_cost=0.10,
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="synth_panel.cost"):
            cost = resolve_cost(usage, "claude-haiku-3.5")
        assert cost.total_cost == pytest.approx(0.10)
        assert any("diverges" in rec.message for rec in caplog.records)

    def test_divergence_silent_within_20_percent(self, caplog):
        from synth_panel.cost import resolve_cost

        # Local: $1.00. Provider: $1.05 → 4.7% divergence, no warning.
        usage = TokenUsage(
            input_tokens=1_000_000,
            provider_reported_cost=1.05,
        )
        import logging

        with caplog.at_level(logging.WARNING, logger="synth_panel.cost"):
            resolve_cost(usage, "claude-haiku-3.5")
        assert not any("diverges" in rec.message for rec in caplog.records)


class TestTokenUsageProviderCost:
    """TokenUsage carries provider-reported cost through addition and (de)ser."""

    def test_field_defaults_none(self):
        u = TokenUsage()
        assert u.provider_reported_cost is None
        assert u.reasoning_tokens == 0
        assert u.cached_tokens == 0

    def test_addition_sums_provider_cost(self):
        a = TokenUsage(provider_reported_cost=0.1, reasoning_tokens=5, cached_tokens=3)
        b = TokenUsage(provider_reported_cost=0.2, reasoning_tokens=7, cached_tokens=4)
        c = a + b
        assert c.provider_reported_cost == pytest.approx(0.3)
        assert c.reasoning_tokens == 12
        assert c.cached_tokens == 7

    def test_addition_preserves_none_when_both_absent(self):
        a = TokenUsage(input_tokens=1)
        b = TokenUsage(input_tokens=2)
        c = a + b
        assert c.provider_reported_cost is None

    def test_addition_treats_missing_as_zero_when_one_present(self):
        a = TokenUsage(provider_reported_cost=0.5)
        b = TokenUsage()
        assert (a + b).provider_reported_cost == pytest.approx(0.5)
        assert (b + a).provider_reported_cost == pytest.approx(0.5)

    def test_roundtrip_dict_preserves_provider_cost(self):
        u = TokenUsage(
            input_tokens=10,
            output_tokens=20,
            provider_reported_cost=0.123,
            reasoning_tokens=5,
            cached_tokens=2,
        )
        assert TokenUsage.from_dict(u.to_dict()) == u


# --- aggregate_per_model --------------------------------------------------


class _FakePanelist:
    """Minimal stand-in for PanelistResult — only model + usage matter."""

    def __init__(self, model: str | None, usage: TokenUsage):
        self.model = model
        self.usage = usage


class TestAggregatePerModel:
    """sp-atvc: verify multi-model panelist usage gets bucketed correctly."""

    def test_single_model_single_bucket(self):
        prs = [
            _FakePanelist("haiku", TokenUsage(input_tokens=100, output_tokens=50)),
            _FakePanelist("haiku", TokenUsage(input_tokens=200, output_tokens=80)),
        ]
        per_usage, per_cost = aggregate_per_model(prs, "haiku")
        assert set(per_usage.keys()) == {"haiku"}
        assert per_usage["haiku"].input_tokens == 300
        assert per_usage["haiku"].output_tokens == 130
        assert per_cost["haiku"].total_cost > 0

    def test_multi_model_separate_buckets(self):
        """Panelists routed to different providers populate one bucket each."""
        prs = [
            _FakePanelist("haiku", TokenUsage(input_tokens=100, output_tokens=50)),
            _FakePanelist("gemini-2.5-flash", TokenUsage(input_tokens=100, output_tokens=50)),
            _FakePanelist("gpt-4o-mini", TokenUsage(input_tokens=100, output_tokens=50)),
        ]
        per_usage, _per_cost = aggregate_per_model(prs, "haiku")
        assert set(per_usage.keys()) == {"haiku", "gemini-2.5-flash", "gpt-4o-mini"}
        # Each bucket carries only its own tokens — NOT the summed total.
        for m in per_usage:
            assert per_usage[m].input_tokens == 100
            assert per_usage[m].output_tokens == 50

    def test_unset_model_falls_back_to_default(self):
        prs = [
            _FakePanelist(None, TokenUsage(input_tokens=10, output_tokens=5)),
            _FakePanelist("haiku", TokenUsage(input_tokens=10, output_tokens=5)),
        ]
        per_usage, _ = aggregate_per_model(prs, "haiku")
        # Both collapse into the default model bucket.
        assert set(per_usage.keys()) == {"haiku"}
        assert per_usage["haiku"].input_tokens == 20

    def test_per_model_cost_uses_each_providers_pricing(self):
        """Bug repro: aggregated cost must price each bucket separately.

        Before sp-atvc the caller summed all usage then priced at the default
        model's rate, which under-reported total cost when cheap models
        (gemini-flash, gpt-mini) carried traffic routed away from a pricier
        default.
        """
        # All three get 1M input tokens. With correct per-model pricing the
        # total cost equals the sum of each provider's input-per-million rate.
        big = TokenUsage(input_tokens=1_000_000)
        prs = [
            _FakePanelist("haiku", big),
            _FakePanelist("gemini-2.5-flash", big),
            _FakePanelist("gpt-4o-mini", big),
        ]
        _usage, per_cost = aggregate_per_model(prs, "haiku")
        total_per_model = sum(c.total_cost for c in per_cost.values())
        # Reference: the wrong way — price 3M tokens at haiku rate only.
        wrong_single_rate = estimate_cost(TokenUsage(input_tokens=3_000_000), HAIKU_PRICING).total_cost
        # The correct sum must differ from the single-rate estimate (the bug).
        assert total_per_model != pytest.approx(wrong_single_rate)
        # And each bucket must be individually non-zero.
        for m in per_cost:
            assert per_cost[m].total_cost > 0

    def test_empty_iterable(self):
        per_usage, per_cost = aggregate_per_model([], "haiku")
        assert per_usage == {}
        assert per_cost == {}


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


# --- Float drift regression (sp-cost-float-drift) --------------------------


class TestCostFloatDrift:
    def test_token_usage_cost_no_float_drift(self):
        """Repeated record_turn over 5000 micro-costs must agree with Decimal sum to 4dp."""
        rng = __import__('random').Random(42)
        costs = [rng.uniform(0.0001, 0.003) for _ in range(5000)]

        # Sum via UsageTracker (the real aggregation path)
        tracker = UsageTracker()
        for c in costs:
            tracker.record_turn(TokenUsage(provider_reported_cost=c))

        # Sum via Decimal (gold standard)
        expected = sum(Decimal(str(c)) for c in costs)

        actual = tracker.cumulative_usage.provider_reported_cost
        assert actual is not None
        assert abs(float(expected) - actual) < 0.0001, (
            f"Cost drift: tracker={actual}, decimal={float(expected)}, "
            f"diff={abs(float(expected) - actual)}"
        )
