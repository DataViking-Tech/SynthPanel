"""v1/v2/v3 backward-compatibility test matrix (F3-F).

Runs the same orchestrator path against v1 (flat questions), v2 (linear
rounds), and v3 (branching) instruments to guarantee 0.5.0 does not
regress 0.4.0 behavior. The LLM client and panelist runner are stubbed
out — we only assert the structural invariants of
``run_multi_round_panel`` (path shape, terminal round, warnings, no
crashes), not real model behavior. End-to-end exercise against a live
API lives in ``test_acceptance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from synth_panel.cost import ZERO_USAGE, TokenUsage
from synth_panel.instrument import parse_instrument
from synth_panel import orchestrator
from synth_panel.orchestrator import (
    MultiRoundResult,
    PanelistResult,
    run_multi_round_panel,
)


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

@dataclass
class _StubSynthesis:
    """Minimal SynthesisResult-shaped object for routing."""

    themes: list[str] = field(default_factory=lambda: ["pricing pain"])
    summary: str = "stub summary"
    agreements: list[str] = field(default_factory=list)
    disagreements: list[str] = field(default_factory=list)
    surprises: list[str] = field(default_factory=list)
    recommendation: str = "stub recommendation"
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "themes": self.themes,
            "summary": self.summary,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "surprises": self.surprises,
            "recommendation": self.recommendation,
        }


def _stub_synthesize_round(client, panelist_results, questions, model):
    return _StubSynthesis()


def _stub_synthesize_final(client, panelist_results, questions, model):
    return _StubSynthesis(summary="final")


@pytest.fixture
def patched_panel(monkeypatch):
    """Stub run_panel_parallel so the matrix needs no real LLM client."""

    def fake_run_panel_parallel(
        client, personas, questions, model, system_prompt_fn,
        question_prompt_fn, max_workers=None, response_schema=None,
        sessions=None,
    ):
        results = [
            PanelistResult(
                persona_name=p.get("name", "anon"),
                responses=[{"q": q.get("text", ""), "a": "ok"} for q in questions],
                usage=ZERO_USAGE,
            )
            for p in personas
        ]
        return results, object(), {}

    monkeypatch.setattr(orchestrator, "run_panel_parallel", fake_run_panel_parallel)


# ---------------------------------------------------------------------------
# Instrument fixtures
# ---------------------------------------------------------------------------

V1_INSTRUMENT = {
    "version": 1,
    "questions": [
        {"text": "What frustrates you about your workflow?"},
        {"text": "What would help most?"},
    ],
}

V2_INSTRUMENT = {
    "version": 2,
    "rounds": [
        {"name": "discovery", "questions": [{"text": "What hurts?"}]},
        {
            "name": "deep_dive",
            "depends_on": "discovery",
            "questions": [{"text": "Tell me more."}],
        },
    ],
}

V3_INSTRUMENT = {
    "version": 3,
    "rounds": [
        {
            "name": "intro",
            "questions": [{"text": "What hurts?"}],
            "route_when": [
                {
                    "if": {"field": "themes", "op": "contains", "value": "pricing"},
                    "goto": "probe_pricing",
                },
                {"else": "wrap_up"},
            ],
        },
        {"name": "probe_pricing", "questions": [{"text": "Tell me about pricing."}]},
        {"name": "wrap_up", "questions": [{"text": "Final thoughts?"}]},
    ],
}


PERSONAS = [{"name": "Alice"}, {"name": "Bob"}]


# ---------------------------------------------------------------------------
# Parsing matrix — all three formats parse through the same entry point
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_round_count",
    [
        (V1_INSTRUMENT, 1),
        (V2_INSTRUMENT, 2),
        (V3_INSTRUMENT, 3),
    ],
    ids=["v1", "v2", "v3"],
)
def test_parse_matrix(raw, expected_round_count):
    inst = parse_instrument(raw)
    assert len(inst.rounds) == expected_round_count
    assert inst.warnings == []


# ---------------------------------------------------------------------------
# Orchestrator matrix — all three formats execute through the same loop
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_path_rounds",
    [
        (V1_INSTRUMENT, ["default"]),
        (V2_INSTRUMENT, ["discovery", "deep_dive"]),
        # v3 routes via 'pricing pain' → probe_pricing, then wrap_up via linear fallthrough
        (V3_INSTRUMENT, ["intro", "probe_pricing"]),
    ],
    ids=["v1", "v2", "v3"],
)
def test_orchestrator_matrix(raw, expected_path_rounds, patched_panel):
    inst = parse_instrument(raw)
    result = run_multi_round_panel(
        client=object(),
        personas=PERSONAS,
        instrument=inst,
        model="stub",
        system_prompt_fn=lambda p: "sys",
        question_prompt_fn=lambda q: q.get("text", ""),
        synthesize_round_fn=_stub_synthesize_round,
        synthesize_final_fn=_stub_synthesize_final,
    )

    assert isinstance(result, MultiRoundResult)
    # At minimum the first expected round must execute. v3's probe_pricing
    # is reached via routing on the 'pricing pain' stub theme.
    executed_names = [r.name for r in result.rounds]
    for name in expected_path_rounds:
        assert name in executed_names, (
            f"expected round {name!r} in executed path, got {executed_names}"
        )

    # Path log entries align with executed rounds.
    assert len(result.path) == len(result.rounds)
    for entry, rr in zip(result.path, result.rounds):
        assert entry["round"] == rr.name
        assert "next" in entry

    # Terminal round is the last executed round, regardless of source format.
    assert result.terminal_round == result.rounds[-1].name

    # final_synthesis fired since synthesize_final_fn was supplied
    assert result.final_synthesis is not None


def test_v1_and_v2_share_single_rounds_list_shape(patched_panel):
    """v1 and v2 outputs are structurally identical lists of RoundResults."""
    v1 = run_multi_round_panel(
        client=object(),
        personas=PERSONAS,
        instrument=parse_instrument(V1_INSTRUMENT),
        model="stub",
        system_prompt_fn=lambda p: "",
        question_prompt_fn=lambda q: q.get("text", ""),
        synthesize_round_fn=_stub_synthesize_round,
    )
    v2 = run_multi_round_panel(
        client=object(),
        personas=PERSONAS,
        instrument=parse_instrument(V2_INSTRUMENT),
        model="stub",
        system_prompt_fn=lambda p: "",
        question_prompt_fn=lambda q: q.get("text", ""),
        synthesize_round_fn=_stub_synthesize_round,
    )
    assert isinstance(v1.rounds, list) and isinstance(v2.rounds, list)
    assert all(isinstance(r, type(v1.rounds[0])) for r in v2.rounds)
    # Both formats produce a path log of the same shape — list of dicts
    # carrying round/branch/next.
    for entry in (*v1.path, *v2.path):
        assert set(entry) >= {"round", "branch", "next"}


def test_v3_path_records_routing_decision(patched_panel):
    """v3 output adds path entries that name the firing branch."""
    result = run_multi_round_panel(
        client=object(),
        personas=PERSONAS,
        instrument=parse_instrument(V3_INSTRUMENT),
        model="stub",
        system_prompt_fn=lambda p: "",
        question_prompt_fn=lambda q: q.get("text", ""),
        synthesize_round_fn=_stub_synthesize_round,
    )
    intro_entry = next(p for p in result.path if p["round"] == "intro")
    # Stub synthesis produces themes=['pricing pain'], so the predicate
    # 'themes contains pricing' fires.
    assert "pricing" in intro_entry["branch"]
    assert intro_entry["next"] == "probe_pricing"
