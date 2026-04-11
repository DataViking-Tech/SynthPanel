"""Tests for synth_panel.metadata — rich metadata for synthbench integration."""

from __future__ import annotations

import json
import time

from synth_panel.cost import CostEstimate, TokenUsage, estimate_cost, lookup_pricing
from synth_panel.metadata import PanelTimer, build_config_hash, build_metadata


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
