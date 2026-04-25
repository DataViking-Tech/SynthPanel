"""Tests for ``synth_panel.reporting.markdown.render_markdown``.

Covers the 7 tests enumerated in
``specs/sp-viz-layer/structure.md`` §6:

1. small panel contains all sections
2. banner present in header AND footer appears (invariant)
3. provenance surfaces config_hash + timestamps
4. missing provenance degrades to placeholder (no KeyError)
5. map-reduce fixture with ``per_question_synthesis`` renders without exception
6. failed panel reports failure stats
7. pipe characters in persona names are escaped
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synth_panel.analysis.inspect import build_inspect_report
from synth_panel.reporting.markdown import BANNER, FOOTER, render_markdown

FIXTURES = Path(__file__).parent / "fixtures" / "reporting"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _render(name: str, *, source_path: str | None = None) -> tuple[str, dict]:
    raw = _load(name)
    report = build_inspect_report(raw)
    return render_markdown(report, raw, source_path=source_path), raw


# ---------------------------------------------------------------------------
# 1. Small-panel contains all sections
# ---------------------------------------------------------------------------


def test_render_small_panel_contains_all_sections() -> None:
    md, _ = _render("rounds_shape.json")

    # Section headings per structure.md §2.
    assert md.startswith("# Panel Report: ")
    assert "## Provenance" in md
    assert "## Overview" in md
    assert "## Per-Model Rollup" in md
    assert "## Per-Persona Summary" in md
    assert "## Synthesis" in md
    assert "## Failure Stats" in md

    # Populated from the fixture.
    assert "claude-sonnet-4-6" in md
    assert "Sarah Chen" in md
    assert "Marcus Reed" in md
    assert "pricing-discovery" in md


# ---------------------------------------------------------------------------
# 2. Invariant: banner and footer are present verbatim
# ---------------------------------------------------------------------------


def test_render_banner_present_in_header_and_footer() -> None:
    md, _ = _render("rounds_shape.json")

    # The synthetic-panel banner is mandatory and verbatim.
    assert BANNER in md, "synthetic-panel banner must appear verbatim"

    # Banner appears near the top — specifically, before the provenance
    # section. This is the "in header" half of the contract.
    banner_idx = md.find(BANNER)
    provenance_idx = md.find("## Provenance")
    assert 0 < banner_idx < provenance_idx

    # Footer is the last non-empty line of the document.
    assert FOOTER in md
    assert md.rstrip().splitlines()[-1] == FOOTER

    # Same invariants hold on the flat-shape fixture.
    md_flat, _ = _render("flat_shape.json")
    assert BANNER in md_flat
    assert md_flat.rstrip().splitlines()[-1] == FOOTER


# ---------------------------------------------------------------------------
# 3. Provenance surfaces config_hash + timestamps
# ---------------------------------------------------------------------------


def test_render_provenance_surfaces_config_hash_and_timestamps() -> None:
    md, raw = _render("rounds_shape.json", source_path="/tmp/report.json")

    assert raw["metadata"]["config_hash"] == "abc123def456"
    assert "abc123def456" in md
    # Timestamp and versions come through.
    assert "2026-04-22T19:00:00Z" in md
    assert "0.5.0" in md
    assert "3.12.1" in md
    # source_path argument surfaces verbatim.
    assert "/tmp/report.json" in md


# ---------------------------------------------------------------------------
# 4. Missing provenance fields degrade to placeholder (no KeyError)
# ---------------------------------------------------------------------------


def test_render_missing_provenance_degrades_to_placeholder() -> None:
    raw: dict = {
        "id": "sparse-panel",
        "results": [
            {
                "persona": "Anon",
                "model": "claude-sonnet-4-6",
                "responses": [{"question": "q?", "response": "a", "usage": {"input_tokens": 1, "output_tokens": 1}}],
            }
        ],
        "question_count": 1,
        "persona_count": 1,
    }

    report = build_inspect_report(raw)
    md = render_markdown(report, raw)

    # No KeyError, document renders.
    assert "# Panel Report: sparse-panel" in md
    # Missing config_hash → (not recorded)
    assert "(not recorded)" in md
    # Missing version fields → (unknown)
    assert "(unknown)" in md
    # Missing source_path → loaded by ID
    assert "(loaded by ID)" in md
    # Banner + footer invariant still holds.
    assert BANNER in md
    assert md.rstrip().splitlines()[-1] == FOOTER


# ---------------------------------------------------------------------------
# 5. Map-reduce fixture with per_question_synthesis renders without exception
# ---------------------------------------------------------------------------


def test_render_mapreduce_per_question_synthesis() -> None:
    # RISK #1 (plan.md): verify build_inspect_report covers this fixture.
    # If this raises, file a new bug and do NOT patch inspect.py inline.
    md, raw = _render("map_reduce_per_question.json")

    # Basic smoke: summary peek from the reduce step is surfaced.
    assert "Panelists are positive about collaboration" in md
    # Per-question synthesis from the fixture is present (or at least did not crash).
    assert "## Synthesis" in md
    # Themes are emitted as a count in v1, not verbatim; check the count
    # makes it in so we exercise the theme_count branch.
    assert raw["synthesis"]["themes"]  # fixture preflight
    assert "Themes" in md


# ---------------------------------------------------------------------------
# 6. Failed panel reports failure stats
# ---------------------------------------------------------------------------


def test_render_failed_panel_reports_failure_stats() -> None:
    raw: dict = {
        "id": "failed-panel",
        "model": "claude-sonnet-4-6",
        "question_count": 2,
        "persona_count": 2,
        "results": [
            {
                "persona": "Alice",
                "model": "claude-sonnet-4-6",
                "responses": [
                    {"question": "q1", "response": "a", "usage": {"input_tokens": 10, "output_tokens": 4}},
                    {"question": "q2", "error": "timeout"},
                ],
            },
            {
                "persona": "Bob",
                "model": "claude-sonnet-4-6",
                "error": "provider_error: 500",
                "responses": [],
            },
        ],
        "failure_stats": {
            "total_pairs": 4,
            "errored_pairs": 3,
            "failure_rate": 0.75,
            "failed_panelists": 1,
            "errored_personas": ["Alice", "Bob"],
        },
    }

    report = build_inspect_report(raw)
    md = render_markdown(report, raw)

    assert "## Failure Stats" in md
    assert "75.0%" in md
    assert "Failed panelists" in md
    assert "Alice" in md
    assert "Bob" in md
    # The panelist_error on Bob surfaces in the per-persona table.
    assert "provider_error: 500" in md


# ---------------------------------------------------------------------------
# 7. Pipe characters in persona names are escaped
# ---------------------------------------------------------------------------


def test_render_escapes_pipes_in_persona_names() -> None:
    raw: dict = {
        "id": "pipe-panel",
        "model": "claude-sonnet-4-6",
        "question_count": 1,
        "persona_count": 1,
        "results": [
            {
                "persona": "Alice | Evil",
                "model": "claude-sonnet-4-6",
                "responses": [{"question": "q", "response": "<ok>", "usage": {"input_tokens": 1, "output_tokens": 1}}],
            }
        ],
    }

    report = build_inspect_report(raw)
    md = render_markdown(report, raw)

    # Raw pipe must not appear inside the persona name cell — it would
    # split the row into extra columns in GFM.
    assert "Alice \\| Evil" in md
    assert "| Alice | Evil |" not in md


# ---------------------------------------------------------------------------
# 8. Synthesis renders full themes/agreements/disagreements/recommendation
#    (sp-xltd) for string-shape items.
# ---------------------------------------------------------------------------


def test_render_synthesis_full_string_shape() -> None:
    md, raw = _render("rounds_shape.json")

    # All four substantive sections render under ## Synthesis.
    synth_idx = md.index("## Synthesis")
    failure_idx = md.index("## Failure Stats")
    synth_block = md[synth_idx:failure_idx]

    assert "**Themes:**" in synth_block
    assert "**Agreements:**" in synth_block
    assert "**Disagreements:**" in synth_block
    assert "**Recommendation:**" in synth_block

    # Theme bullet content (string shape).
    assert "- pricing" in synth_block
    assert "- onboarding" in synth_block

    # Agreements/disagreements as numbered lists with verbatim text.
    assert "1. Pricing transparency is the single largest blocker" in synth_block
    assert "2. Onboarding friction" in synth_block
    assert "1. Acceptable per-seat price band" in synth_block

    # Recommendation as a blockquote.
    assert "> Publish a tiered pricing page" in synth_block

    # Existing summary peek still renders — the detail sections are
    # additive, not a replacement.
    assert "Panelists agree that pricing opacity" in synth_block


# ---------------------------------------------------------------------------
# 9. Synthesis renders dict-shape themes/agreements (sp-xltd).
# ---------------------------------------------------------------------------


def test_render_synthesis_full_dict_shape() -> None:
    md, _ = _render("synthesis_dict_shape.json")

    synth_idx = md.index("## Synthesis")
    failure_idx = md.index("## Failure Stats")
    synth_block = md[synth_idx:failure_idx]

    # Dict-shape themes render as "**Title** — description".
    assert "- **Tedium** — Repeated manual steps" in synth_block
    assert "- **Speed expectations** — Panelists expect sub-second" in synth_block

    # Dict-shape agreement (title+description) collapses to "title: desc"
    # in numbered-list form.
    assert "1. Manual steps: All panelists call out" in synth_block

    # Dict-shape disagreement keyed only on `text` still surfaces.
    assert "1. Whether to fix UX first" in synth_block

    # Recommendation (string) still rendered.
    assert "> Cut the three redundant steps" in synth_block


# ---------------------------------------------------------------------------
# 10. Synthesis sections truncate runaway items (sp-xltd).
# ---------------------------------------------------------------------------


def test_render_synthesis_truncates_long_items() -> None:
    raw: dict = {
        "id": "long-synth-panel",
        "model": "claude-sonnet-4-6",
        "question_count": 1,
        "persona_count": 1,
        "results": [
            {
                "persona": "Anon",
                "model": "claude-sonnet-4-6",
                "responses": [
                    {
                        "question": "q?",
                        "response": "a",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ],
            }
        ],
        "synthesis": {
            "ran": True,
            "model": "claude-sonnet-4-6",
            "summary": "ok",
            "themes": ["x" * 800],
            "agreements": ["y" * 800],
            "disagreements": [],
            "recommendation": "z" * 800,
        },
    }

    report = build_inspect_report(raw)
    md = render_markdown(report, raw)

    # Truncation marker is the ellipsis; raw 800-char strings must not
    # appear verbatim.
    assert "x" * 800 not in md
    assert "y" * 800 not in md
    assert "z" * 800 not in md
    assert "…" in md


# ---------------------------------------------------------------------------
# 11. Empty synthesis fields don't emit empty section headers (sp-xltd).
# ---------------------------------------------------------------------------


def test_render_synthesis_skips_empty_sections() -> None:
    raw: dict = {
        "id": "minimal-synth-panel",
        "model": "claude-sonnet-4-6",
        "question_count": 1,
        "persona_count": 1,
        "results": [
            {
                "persona": "Anon",
                "model": "claude-sonnet-4-6",
                "responses": [
                    {"question": "q?", "response": "a", "usage": {"input_tokens": 1, "output_tokens": 1}}
                ],
            }
        ],
        "synthesis": {
            "ran": True,
            "model": "claude-sonnet-4-6",
            "summary": "minimal",
            "themes": [],
            "agreements": [],
            "disagreements": [],
            "recommendation": "",
        },
    }

    report = build_inspect_report(raw)
    md = render_markdown(report, raw)

    # No empty bullet sections.
    assert "**Themes:**" not in md
    assert "**Agreements:**" not in md
    assert "**Disagreements:**" not in md
    assert "**Recommendation:**" not in md
    # Status still renders.
    assert "**Status:** ran" in md


if __name__ == "__main__":  # pragma: no cover - convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
