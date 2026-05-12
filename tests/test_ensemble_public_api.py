"""Public-surface contract tests for ``synth_panel.ensemble``.

Pins the v1.1.0 promise (sy-0gy) that external callers can import the
full deliberation engine — runner, blender, judge, map-reduce, seed
pin — from one module path. These tests intentionally exercise the
*re-exports* and ``__all__`` rather than the underlying implementations
(those are covered by ``test_ensemble.py``, ``test_synthesis*.py``).

If a future refactor splits ``synth_panel.ensemble`` into a sub-package
or moves the judge primitives elsewhere, this test should be the first
thing that breaks — that breakage is a deliberate signal that the
public contract is changing and consumers (e.g. boardroom) need a
heads-up.
"""

from __future__ import annotations

import inspect

import pytest


def test_ensemble_module_importable() -> None:
    import synth_panel.ensemble as ens

    assert hasattr(ens, "__all__"), "synth_panel.ensemble must declare __all__"
    assert isinstance(ens.__all__, list)
    assert ens.__all__, "__all__ must not be empty"


# ---------------------------------------------------------------------------
# Surface: ensemble runner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "ensemble_run",
        "EnsembleResult",
        "ModelRunResult",
        "build_ensemble_output",
        "build_mixed_model_rollup",
        "collect_ensemble_incidents",
        "build_ensemble_incident_warnings",
    ],
)
def test_runner_surface_exposed(name: str) -> None:
    import synth_panel.ensemble as ens

    assert hasattr(ens, name), f"public runner surface missing: {name}"
    assert name in ens.__all__, f"{name} must be listed in __all__"


# ---------------------------------------------------------------------------
# Surface: blender
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["blend_distributions", "BlendedResult", "BlendedQuestion"],
)
def test_blender_surface_exposed(name: str) -> None:
    import synth_panel.ensemble as ens

    assert hasattr(ens, name), f"public blender surface missing: {name}"
    assert name in ens.__all__, f"{name} must be listed in __all__"


def test_blend_distributions_supports_model_weights() -> None:
    """Model-weighted scoring is part of the public contract."""
    from synth_panel.ensemble import blend_distributions

    sig = inspect.signature(blend_distributions)
    assert "weights" in sig.parameters, (
        "blend_distributions(..., weights=) is a public contract — weights mapping enables model-weighted scoring"
    )


# ---------------------------------------------------------------------------
# Surface: judge (re-exported from synth_panel.synthesis)
# ---------------------------------------------------------------------------


def test_judge_reexported_under_ensemble() -> None:
    """The single-judge synthesis primitive is reachable via the ensemble namespace."""
    from synth_panel.ensemble import SynthesisResult, synthesize_panel
    from synth_panel.synthesis import SynthesisResult as _SR
    from synth_panel.synthesis import synthesize_panel as _sp

    # Re-exports must be the same object — not a wrapper.
    assert SynthesisResult is _SR
    assert synthesize_panel is _sp


# ---------------------------------------------------------------------------
# Surface: map-reduce (re-exported from synth_panel.synthesis)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "synthesize_panel_mapreduce",
        "select_strategy",
        "resolve_context_window",
        "estimate_single_pass_tokens",
        "MapPhaseFailure",
        "MapChunkOverflowError",
        "STRATEGY_SINGLE",
        "STRATEGY_MAP_REDUCE",
        "STRATEGY_AUTO",
        "SYNTHESIS_STRATEGIES",
    ],
)
def test_mapreduce_surface_exposed(name: str) -> None:
    import synth_panel.ensemble as ens

    assert hasattr(ens, name), f"public map-reduce surface missing: {name}"
    assert name in ens.__all__, f"{name} must be listed in __all__"


def test_mapreduce_reexports_match_synthesis_module() -> None:
    from synth_panel import ensemble as ens
    from synth_panel import synthesis as syn

    for name in (
        "synthesize_panel_mapreduce",
        "select_strategy",
        "resolve_context_window",
        "MapPhaseFailure",
        "MapChunkOverflowError",
        "STRATEGY_SINGLE",
        "STRATEGY_MAP_REDUCE",
        "STRATEGY_AUTO",
        "SYNTHESIS_STRATEGIES",
    ):
        assert getattr(ens, name) is getattr(syn, name), f"{name} re-export drifted from source"


# ---------------------------------------------------------------------------
# Surface: seed pinning
# ---------------------------------------------------------------------------


def test_ensemble_run_accepts_seed() -> None:
    """Seed pinning is part of the public ensemble contract."""
    from synth_panel.ensemble import ensemble_run

    sig = inspect.signature(ensemble_run)
    assert "seed" in sig.parameters, (
        "ensemble_run(..., seed=N) is a public contract — deterministic reproducibility on providers that support it"
    )


def test_completion_request_carries_seed() -> None:
    """The underlying request shape — what the seed pins onto — is stable."""
    from synth_panel.llm.models import CompletionRequest

    # `CompletionRequest` is the surface the seed pin lives on. If the
    # field name changes, the ensemble seed contract breaks silently;
    # this assertion makes that break loud.
    assert "seed" in {f for f in CompletionRequest.__dataclass_fields__}


# ---------------------------------------------------------------------------
# __all__ hygiene
# ---------------------------------------------------------------------------


def test_all_entries_are_actually_exported() -> None:
    """Every name in __all__ must resolve to a real attribute."""
    import synth_panel.ensemble as ens

    missing = [name for name in ens.__all__ if not hasattr(ens, name)]
    assert not missing, f"__all__ lists names that don't exist on the module: {missing}"


def test_no_private_names_in_all() -> None:
    import synth_panel.ensemble as ens

    private = [name for name in ens.__all__ if name.startswith("_")]
    assert not private, f"__all__ must not list private names: {private}"
