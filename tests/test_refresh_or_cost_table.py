"""Tests for scripts/refresh_or_cost_table.py — hq-xq36.

The script's job is to flag drift between OpenRouter's published rates
and the local pricing table. We exercise the comparison logic with a
synthetic models payload so the test stays offline.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "refresh_or_cost_table.py"


@pytest.fixture(scope="module")
def refresh_module():
    import sys

    spec = importlib.util.spec_from_file_location("refresh_or_cost_table", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass forward-refs resolve via sys.modules.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def _model(model_id: str, prompt: float, completion: float, *, cw: float = 0.0, cr: float = 0.0) -> dict:
    """Build an OR /v1/models entry. Pricing fields are per-token (USD)."""
    return {
        "id": model_id,
        "pricing": {
            "prompt": str(prompt),
            "completion": str(completion),
            "input_cache_write": str(cw),
            "input_cache_read": str(cr),
        },
    }


def test_drift_within_tolerance_returns_zero(refresh_module):
    """When every mapped model matches local rates, exit code is 0 and
    output advertises the OK summary."""
    # Use the live mapping but feed rates equal to whatever the local
    # table has for each id, so drift is mechanically zero.
    from synth_panel.cost import lookup_pricing

    models = []
    for or_id in refresh_module.EXPECTED_OR_MAPPING:
        local, _ = lookup_pricing(f"openrouter/{or_id}")
        models.append(
            _model(
                or_id,
                local.input_cost_per_million / 1_000_000,
                local.output_cost_per_million / 1_000_000,
            )
        )

    drifts = refresh_module.compute_drift(models)
    text, failed = refresh_module._format_text(drifts, max_drift=0.25)
    assert failed is False
    assert "OK:" in text


def test_input_drift_above_threshold_fails(refresh_module):
    """A model whose OR rate is 2x the local rate must trip the gate."""
    from synth_panel.cost import lookup_pricing

    or_id = next(iter(refresh_module.EXPECTED_OR_MAPPING))
    local, _ = lookup_pricing(f"openrouter/{or_id}")
    # OR rate inflated 2x → 50% drift, well above the 25% threshold.
    models = [
        _model(
            or_id,
            (local.input_cost_per_million * 2) / 1_000_000,
            local.output_cost_per_million / 1_000_000,
        )
    ]
    # Pad with the rest at parity so we isolate the failure to one model.
    for other in list(refresh_module.EXPECTED_OR_MAPPING)[1:]:
        l2, _ = lookup_pricing(f"openrouter/{other}")
        models.append(
            _model(
                other,
                l2.input_cost_per_million / 1_000_000,
                l2.output_cost_per_million / 1_000_000,
            )
        )

    drifts = refresh_module.compute_drift(models)
    text, failed = refresh_module._format_text(drifts, max_drift=0.25)
    assert failed is True
    assert "FAIL" in text
    # The offender's row should carry the marker.
    offender = next(d for d in drifts if d.model == or_id)
    assert offender.input_drift > 0.25


def test_missing_model_fails(refresh_module):
    """If OR drops a model we map, the script must flag it (not silently
    skip) so the table can be pruned or the mapping updated."""
    # Provide everything except the first mapped model.
    from synth_panel.cost import lookup_pricing

    keys = list(refresh_module.EXPECTED_OR_MAPPING)
    missing = keys[0]
    models = []
    for other in keys[1:]:
        local, _ = lookup_pricing(f"openrouter/{other}")
        models.append(
            _model(
                other,
                local.input_cost_per_million / 1_000_000,
                local.output_cost_per_million / 1_000_000,
            )
        )

    drifts = refresh_module.compute_drift(models)
    text, failed = refresh_module._format_text(drifts, max_drift=0.25)
    assert failed is True
    assert missing in text
    assert "NOT FOUND" in text
    drift_for_missing = next(d for d in drifts if d.model == missing)
    assert drift_for_missing.found is False


def test_relative_drift_handles_zero_upstream(refresh_module):
    """When upstream rate is 0 (e.g. provider stops billing for cache
    reads), the drift function must not divide by zero."""
    # Both zero → 0 drift (trivially aligned).
    assert refresh_module._relative_drift(0.0, 0.0) == 0.0
    # Local non-zero, upstream zero → max-of(0, local) > 0, returns
    # the full ratio without raising.
    drift = refresh_module._relative_drift(1.0, 0.0)
    assert drift == 1.0
