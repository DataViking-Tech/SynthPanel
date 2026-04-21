"""Rich metadata for synthpanel JSON output (synthbench integration).

Builds the top-level ``metadata`` key that synthbench consumes for
reproducibility tracking, cost attribution, and run provenance.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from typing import Any

from synth_panel.cost import CostEstimate, TokenUsage, lookup_pricing
from synth_panel.llm.aliases import resolve_alias


def _get_synthpanel_version() -> str:
    """Return the installed synthpanel version."""
    try:
        from importlib.metadata import version

        return version("synth-panel")
    except Exception:
        from synth_panel import __version__

        return __version__


def _get_python_version() -> str:
    return platform.python_version()


def build_config_hash(config: dict[str, Any]) -> str:
    """SHA256 of deterministically serialized config for reproducibility."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


# Truncation length for per-value var hashes. 16 hex chars = 64 bits, enough
# to distinguish var sets in practice while keeping the fingerprint compact.
_VAR_VALUE_HASH_LEN = 16


def build_template_vars_fingerprint(
    template_vars: dict[str, str] | None,
) -> dict[str, str]:
    """Return ``{key: sha256(value)[:16]}`` for the resolved ``--var`` set.

    Keys from ``template_vars`` are preserved (they are the *names* authors
    reference in instrument placeholders and must be visible for audit).
    Values are one-way hashed so the fingerprint can be persisted in
    metadata without leaking the raw substitution (some vars carry
    candidate names, URLs, or prices that callers may treat as sensitive).

    Returns an empty dict when ``template_vars`` is ``None`` or empty so
    callers can unconditionally splice the result into a config-hash input.
    """
    if not template_vars:
        return {}
    return {
        key: hashlib.sha256(str(template_vars[key]).encode()).hexdigest()[:_VAR_VALUE_HASH_LEN]
        for key in sorted(template_vars)
    }


class PanelTimer:
    """Wall-clock timer for panel execution."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._end: float | None = None

    def stop(self) -> None:
        self._end = time.monotonic()

    @property
    def total_seconds(self) -> float:
        end = self._end if self._end is not None else time.monotonic()
        return round(end - self._start, 2)


def build_metadata(
    *,
    panelist_model: str,
    synthesis_model: str | None = None,
    panelist_usage: TokenUsage,
    panelist_cost: CostEstimate,
    synthesis_usage: TokenUsage | None = None,
    synthesis_cost: CostEstimate | None = None,
    total_usage: TokenUsage,
    total_cost: CostEstimate,
    persona_count: int,
    question_count: int,
    timer: PanelTimer | None = None,
    template_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the ``metadata`` dict for synthbench integration.

    Parameters correspond to data already available at the call site in
    both CLI and MCP output paths — no new tracking infrastructure is
    required.
    """
    resolved_panelist = resolve_alias(panelist_model)
    resolved_synthesis = resolve_alias(synthesis_model) if synthesis_model else resolved_panelist

    # -- generation_params --
    generation_params: dict[str, Any] = {
        "temperature": None,
        "top_p": None,
        "max_tokens": 4096,
    }

    # -- models --
    models: dict[str, str] = {
        "panelist": resolved_panelist,
        "synthesis": resolved_synthesis,
    }

    # -- cost --
    _panelist_pricing, _ = lookup_pricing(panelist_model)
    per_model: dict[str, Any] = {
        resolved_panelist: {
            "tokens": panelist_usage.total_tokens,
            "cost_usd": round(panelist_cost.total_cost, 6),
        },
    }
    if synthesis_usage is not None and synthesis_cost is not None and resolved_synthesis != resolved_panelist:
        per_model[resolved_synthesis] = {
            "tokens": synthesis_usage.total_tokens,
            "cost_usd": round(synthesis_cost.total_cost, 6),
        }
    elif synthesis_usage is not None and synthesis_cost is not None:
        # Same model — merge into existing entry
        existing = per_model[resolved_panelist]
        existing["tokens"] += synthesis_usage.total_tokens
        existing["cost_usd"] = round(existing["cost_usd"] + synthesis_cost.total_cost, 6)

    cost: dict[str, Any] = {
        "total_tokens": total_usage.total_tokens,
        "total_cost_usd": round(total_cost.total_cost, 6),
        "per_model": per_model,
    }

    # -- timing --
    timing: dict[str, float] = {}
    if timer is not None:
        total_secs = timer.total_seconds
        timing = {
            "total_seconds": total_secs,
            "per_panelist_avg": round(total_secs / max(persona_count, 1), 2),
        }

    # -- version --
    version: dict[str, str] = {
        "synthpanel": _get_synthpanel_version(),
        "python": _get_python_version(),
    }

    # -- config_hash --
    # sp-ui40: fold resolved --var keys + hashed values into the hash so
    # runs that differ only in their var substitution no longer collide.
    # The fingerprint is only spliced in when a non-empty var set was
    # provided so runs with no --var keep the same hash they had before.
    vars_fingerprint = build_template_vars_fingerprint(template_vars)
    config_for_hash: dict[str, Any] = {
        "panelist_model": resolved_panelist,
        "synthesis_model": resolved_synthesis,
        "persona_count": persona_count,
        "question_count": question_count,
        "generation_params": generation_params,
    }
    if vars_fingerprint:
        config_for_hash["template_vars"] = vars_fingerprint

    meta: dict[str, Any] = {
        "generation_params": generation_params,
        "models": models,
        "cost": cost,
        "timing": timing,
        "version": version,
        "config_hash": build_config_hash(config_for_hash),
    }
    if vars_fingerprint:
        meta["template_vars_fingerprint"] = vars_fingerprint
    return meta
