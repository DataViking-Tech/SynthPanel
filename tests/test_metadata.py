"""Tests for synth_panel.metadata — rich metadata for synthbench integration."""

from __future__ import annotations

import json
import time

import pytest

from synth_panel.cost import CostEstimate, TokenUsage, estimate_cost, lookup_pricing
from synth_panel.metadata import (
    PanelTimer,
    build_config_hash,
    build_metadata,
    build_template_vars_fingerprint,
)


class TestBuildConfigHash:
    def test_deterministic(self):
        config = {"model": "haiku", "persona_count": 3}
        h1 = build_config_hash(config)
        h2 = build_config_hash(config)
        assert h1 == h2

    def test_order_independent(self):
        h1 = build_config_hash({"a": 1, "b": 2})
        h2 = build_config_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_configs_different_hashes(self):
        h1 = build_config_hash({"model": "haiku"})
        h2 = build_config_hash({"model": "sonnet"})
        assert h1 != h2

    def test_sha256_length(self):
        h = build_config_hash({"x": 1})
        assert len(h) == 64


class TestBuildTemplateVarsFingerprint:
    def test_none_returns_empty(self):
        assert build_template_vars_fingerprint(None) == {}

    def test_empty_dict_returns_empty(self):
        assert build_template_vars_fingerprint({}) == {}

    def test_keys_preserved_values_hashed(self):
        fp = build_template_vars_fingerprint({"landing_page": "https://example.com"})
        assert list(fp.keys()) == ["landing_page"]
        assert fp["landing_page"] != "https://example.com"
        assert len(fp["landing_page"]) == 16

    def test_deterministic(self):
        a = build_template_vars_fingerprint({"theme": "pricing", "region": "eu"})
        b = build_template_vars_fingerprint({"region": "eu", "theme": "pricing"})
        assert a == b

    def test_different_values_different_hashes(self):
        a = build_template_vars_fingerprint({"landing_page": "v1"})
        b = build_template_vars_fingerprint({"landing_page": "v2"})
        assert a["landing_page"] != b["landing_page"]


class TestPanelTimer:
    def test_total_seconds(self):
        timer = PanelTimer()
        time.sleep(0.05)
        timer.stop()
        assert timer.total_seconds >= 0.04
        assert timer.total_seconds < 1.0

    def test_stop_freezes_time(self):
        timer = PanelTimer()
        timer.stop()
        t1 = timer.total_seconds
        time.sleep(0.05)
        t2 = timer.total_seconds
        assert t1 == t2


class TestBuildMetadata:
    def _make_usage(self, inp: int = 100, out: int = 50) -> TokenUsage:
        return TokenUsage(input_tokens=inp, output_tokens=out)

    def _make_cost(self, usage: TokenUsage, model: str = "haiku") -> CostEstimate:
        pricing, _ = lookup_pricing(model)
        return estimate_cost(usage, pricing)

    def test_basic_structure(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=3,
            question_count=5,
        )
        assert "generation_params" in meta
        assert "models" in meta
        assert "cost" in meta
        assert "timing" in meta
        assert "version" in meta
        assert "config_hash" in meta

    def test_generation_params_defaults(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        gp = meta["generation_params"]
        assert gp["temperature"] is None
        assert gp["top_p"] is None
        assert gp["max_tokens"] == 4096

    def test_models_resolved(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="sonnet",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        models = meta["models"]
        # Aliases are resolved to canonical model names
        assert models["panelist"] != "haiku"
        assert "claude" in models["panelist"]
        assert models["synthesis"] != "sonnet"
        assert "claude" in models["synthesis"]

    def test_cost_breakdown(self):
        p_usage = self._make_usage(100, 50)
        s_usage = self._make_usage(200, 100)
        p_cost = self._make_cost(p_usage, "haiku")
        s_cost = self._make_cost(s_usage, "sonnet")
        total_usage = p_usage + s_usage
        total_cost = p_cost + s_cost
        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="sonnet",
            panelist_usage=p_usage,
            panelist_cost=p_cost,
            synthesis_usage=s_usage,
            synthesis_cost=s_cost,
            total_usage=total_usage,
            total_cost=total_cost,
            persona_count=2,
            question_count=3,
        )
        cost_meta = meta["cost"]
        assert cost_meta["total_tokens"] == total_usage.total_tokens
        assert cost_meta["total_cost_usd"] > 0
        assert len(cost_meta["per_model"]) == 2

    def test_cost_same_model_merged(self):
        p_usage = self._make_usage(100, 50)
        s_usage = self._make_usage(200, 100)
        p_cost = self._make_cost(p_usage, "haiku")
        s_cost = self._make_cost(s_usage, "haiku")
        total_usage = p_usage + s_usage
        total_cost = p_cost + s_cost
        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="haiku",
            panelist_usage=p_usage,
            panelist_cost=p_cost,
            synthesis_usage=s_usage,
            synthesis_cost=s_cost,
            total_usage=total_usage,
            total_cost=total_cost,
            persona_count=2,
            question_count=3,
        )
        # Same model — should be merged into one entry
        assert len(meta["cost"]["per_model"]) == 1

    def test_timing_with_timer(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        timer = PanelTimer()
        time.sleep(0.05)
        timer.stop()
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=3,
            question_count=1,
            timer=timer,
        )
        assert meta["timing"]["total_seconds"] >= 0.04
        assert meta["timing"]["per_panelist_avg"] >= 0.01

    def test_timing_empty_without_timer(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        assert meta["timing"] == {}

    def test_version_present(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        assert "synthpanel" in meta["version"]
        assert "python" in meta["version"]
        assert "." in meta["version"]["python"]

    def test_config_hash_stable(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        kwargs = dict(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=2,
            question_count=3,
        )
        m1 = build_metadata(**kwargs)
        m2 = build_metadata(**kwargs)
        assert m1["config_hash"] == m2["config_hash"]

    def test_template_vars_changes_config_hash(self):
        """sp-ui40: two runs that differ only in --var values must hash differently."""
        usage = self._make_usage()
        cost = self._make_cost(usage)
        base = dict(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=2,
            question_count=3,
        )
        m_a = build_metadata(**base, template_vars={"landing_page": "v1"})
        m_b = build_metadata(**base, template_vars={"landing_page": "v2"})
        assert m_a["config_hash"] != m_b["config_hash"]

    def test_template_vars_different_keys_different_hash(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        base = dict(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        m_a = build_metadata(**base, template_vars={"theme": "pricing"})
        m_b = build_metadata(**base, template_vars={"channel": "pricing"})
        assert m_a["config_hash"] != m_b["config_hash"]

    def test_template_vars_stable_across_key_order(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        base = dict(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        m_a = build_metadata(**base, template_vars={"a": "1", "b": "2"})
        m_b = build_metadata(**base, template_vars={"b": "2", "a": "1"})
        assert m_a["config_hash"] == m_b["config_hash"]

    def test_template_vars_none_preserves_legacy_hash(self):
        """Runs without --var must produce the same hash as before sp-ui40."""
        usage = self._make_usage()
        cost = self._make_cost(usage)
        base = dict(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
        )
        m_none = build_metadata(**base)
        m_empty = build_metadata(**base, template_vars={})
        m_explicit_none = build_metadata(**base, template_vars=None)
        assert m_none["config_hash"] == m_empty["config_hash"]
        assert m_none["config_hash"] == m_explicit_none["config_hash"]
        assert "template_vars_fingerprint" not in m_none
        assert "template_vars_fingerprint" not in m_empty

    def test_template_vars_fingerprint_exposed(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=1,
            question_count=1,
            template_vars={"theme": "pricing", "audience": "smb"},
        )
        fp = meta["template_vars_fingerprint"]
        assert set(fp.keys()) == {"audience", "theme"}
        # Each value is a truncated SHA256 hex digest
        for v in fp.values():
            assert len(v) == 16
            int(v, 16)  # raises if non-hex
        # Keys survive as cleartext; values are hashed
        assert "pricing" not in fp.values()

    def test_json_serializable(self):
        usage = self._make_usage()
        cost = self._make_cost(usage)
        timer = PanelTimer()
        timer.stop()
        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="sonnet",
            panelist_usage=usage,
            panelist_cost=cost,
            total_usage=usage,
            total_cost=cost,
            persona_count=2,
            question_count=3,
            timer=timer,
        )
        # Must be JSON-serializable without errors
        serialized = json.dumps(meta)
        parsed = json.loads(serialized)
        assert parsed["models"]["panelist"] == meta["models"]["panelist"]


class TestBuildMetadataPerModelOverride:
    """sp-atvc: multi-model ensemble runs must surface every model in
    ``metadata.cost.per_model`` with its real tokens and cost — not collapse
    into a single bucket for the default model.
    """

    def _usage(self, inp: int, out: int) -> TokenUsage:
        return TokenUsage(input_tokens=inp, output_tokens=out)

    def _cost(self, usage: TokenUsage, model: str) -> CostEstimate:
        pricing, _ = lookup_pricing(model)
        return estimate_cost(usage, pricing)

    def test_three_model_ensemble_records_all_three(self):
        """The bug repro: 3-model ensemble must produce 3 per_model entries."""
        haiku_u = self._usage(10_000, 5_000)
        gpt_u = self._usage(12_000, 4_000)
        gem_u = self._usage(15_000, 6_000)
        haiku_c = self._cost(haiku_u, "haiku")
        gpt_c = self._cost(gpt_u, "gpt-4o-mini")
        gem_c = self._cost(gem_u, "gemini-2.5-flash")
        total_u = haiku_u + gpt_u + gem_u
        total_c = haiku_c + gpt_c + gem_c

        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=total_u,
            panelist_cost=total_c,
            total_usage=total_u,
            total_cost=total_c,
            persona_count=6,
            question_count=4,
            panelist_per_model={
                "haiku": (haiku_u, haiku_c),
                "gpt-4o-mini": (gpt_u, gpt_c),
                "gemini-2.5-flash": (gem_u, gem_c),
            },
        )

        per_model = meta["cost"]["per_model"]
        # All three providers must appear — regression guard for the
        # audited "only openrouter/anthropic/claude-haiku-4.5 present" bug.
        assert len(per_model) == 3
        # Every bucket carries its own tokens, not the total.
        for entry in per_model.values():
            assert entry["tokens"] < total_u.total_tokens
            assert entry["cost_usd"] > 0
        # Sum of per_model cost ≈ reported total cost.
        summed = sum(e["cost_usd"] for e in per_model.values())
        assert summed == pytest.approx(meta["cost"]["total_cost_usd"])

    def test_override_plus_synthesis_different_model(self):
        """Synthesis model gets its own bucket on top of ensemble panelists."""
        haiku_u = self._usage(100, 50)
        gpt_u = self._usage(100, 50)
        haiku_c = self._cost(haiku_u, "haiku")
        gpt_c = self._cost(gpt_u, "gpt-4o-mini")
        synth_u = self._usage(200, 100)
        synth_c = self._cost(synth_u, "sonnet")

        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="sonnet",
            panelist_usage=haiku_u + gpt_u,
            panelist_cost=haiku_c + gpt_c,
            synthesis_usage=synth_u,
            synthesis_cost=synth_c,
            total_usage=haiku_u + gpt_u + synth_u,
            total_cost=haiku_c + gpt_c + synth_c,
            persona_count=2,
            question_count=1,
            panelist_per_model={
                "haiku": (haiku_u, haiku_c),
                "gpt-4o-mini": (gpt_u, gpt_c),
            },
        )

        per_model = meta["cost"]["per_model"]
        # Three distinct buckets: haiku + gpt-mini + synthesis model.
        assert len(per_model) == 3

    def test_override_merges_when_synthesis_shares_model(self):
        """Synthesis sharing a panelist model merges into that bucket."""
        haiku_u = self._usage(100, 50)
        gpt_u = self._usage(100, 50)
        haiku_c = self._cost(haiku_u, "haiku")
        gpt_c = self._cost(gpt_u, "gpt-4o-mini")
        synth_u = self._usage(200, 100)
        synth_c = self._cost(synth_u, "haiku")

        meta = build_metadata(
            panelist_model="haiku",
            synthesis_model="haiku",
            panelist_usage=haiku_u + gpt_u,
            panelist_cost=haiku_c + gpt_c,
            synthesis_usage=synth_u,
            synthesis_cost=synth_c,
            total_usage=haiku_u + gpt_u + synth_u,
            total_cost=haiku_c + gpt_c + synth_c,
            persona_count=2,
            question_count=1,
            panelist_per_model={
                "haiku": (haiku_u, haiku_c),
                "gpt-4o-mini": (gpt_u, gpt_c),
            },
        )

        per_model = meta["cost"]["per_model"]
        assert len(per_model) == 2
        # haiku bucket should reflect panelist + synthesis tokens.
        haiku_key = next(k for k in per_model if "haiku" in k)
        assert per_model[haiku_key]["tokens"] == (haiku_u + synth_u).total_tokens

    def test_legacy_single_model_path_unchanged(self):
        """Omitting panelist_per_model preserves the original shape."""
        u = self._usage(100, 50)
        c = self._cost(u, "haiku")
        meta = build_metadata(
            panelist_model="haiku",
            panelist_usage=u,
            panelist_cost=c,
            total_usage=u,
            total_cost=c,
            persona_count=1,
            question_count=1,
        )
        per_model = meta["cost"]["per_model"]
        assert len(per_model) == 1
