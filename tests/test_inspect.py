"""Tests for the `panel inspect` subcommand and analysis.inspect module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from synth_panel.analysis.inspect import build_inspect_report, format_inspect_text
from synth_panel.llm.aliases import resolve_alias
from synth_panel.main import main


def _flat_result(n_errors: int = 0) -> dict:
    """Shape emitted by save_panel_result (flat, no rounds)."""
    responses_per_persona = []
    for persona_idx in range(3):
        responses = []
        for q in range(2):
            is_err = persona_idx < n_errors
            responses.append(
                {
                    "question": f"Q{q}",
                    "response": "[error]" if is_err else "yes",
                    "follow_up": False,
                    "error": is_err,
                }
            )
        responses_per_persona.append(responses)

    return {
        "id": "result-test",
        "created_at": "2026-04-20T00:00:00+00:00",
        "model": "haiku",
        "persona_count": 3,
        "question_count": 2,
        "total_usage": {
            "input_tokens": 300,
            "output_tokens": 150,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        "total_cost": "$0.01",
        "results": [
            {
                "persona": f"Persona_{i}",
                "responses": responses_per_persona[i],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "cost": "$0.003",
                "error": None,
                "model": "haiku",
            }
            for i in range(3)
        ],
    }


def _rounds_result() -> dict:
    """Shape emitted by --output-format json (rounds-shaped + metadata)."""
    base = _flat_result(n_errors=1)
    base_results = base.pop("results")
    base.update(
        {
            "rounds": [
                {"name": "discovery", "results": base_results, "synthesis": None},
            ],
            "path": [
                {"round": "discovery", "branch": "linear", "next": "probe_pricing"},
                {"round": "probe_pricing", "branch": "themes contains price", "next": "__end__"},
            ],
            "warnings": ["low-n: 3 panelists"],
            "synthesis": {
                "summary": "The panel converged on a single, pricing-led theme.",
                "themes": ["price", "trust"],
                "agreements": [],
                "disagreements": [],
                "surprises": [],
                "recommendation": "ship it",
                "usage": {"input_tokens": 800, "output_tokens": 200},
                "cost": "$0.02",
                "model": "sonnet",
                "prompt_version": 2,
            },
            "panelist_cost": "$0.009",
            "panelist_usage": {
                "input_tokens": 300,
                "output_tokens": 150,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            "failure_stats": {
                "total_pairs": 6,
                "errored_pairs": 2,
                "failure_rate": 0.3333333,
                "failed_panelists": 0,
                "errored_personas": ["Persona_0"],
            },
            "run_invalid": False,
            "metadata": {
                "cost": {
                    "total_tokens": 1450,
                    "total_cost_usd": 0.029,
                    "per_model": {
                        "haiku": {"tokens": 450, "cost_usd": 0.009},
                        "sonnet": {"tokens": 1000, "cost_usd": 0.02},
                    },
                }
            },
            "models": ["haiku", "sonnet"],
            "instrument_name": "pricing-discovery",
        }
    )
    return base


def test_inspect_flat_shape_basics():
    report = build_inspect_report(_flat_result(n_errors=1))

    assert report.model == "haiku"
    assert report.persona_count == 3
    assert report.question_count == 2
    assert report.round_names == ["default"]
    assert report.has_rounds_shape is False
    assert [p.name for p in report.personas] == [
        "Persona_0",
        "Persona_1",
        "Persona_2",
    ]
    assert report.personas[0].error_count == 2
    assert report.personas[1].error_count == 0
    assert report.failure_stats.total_pairs == 6
    assert report.failure_stats.errored_pairs == 2
    assert report.failure_stats.errored_personas == ["Persona_0"]
    assert report.synthesis.ran is False


def test_inspect_rounds_shape_full():
    report = build_inspect_report(_rounds_result())

    assert report.has_rounds_shape is True
    assert report.round_names == ["discovery"]
    assert report.path == [
        "discovery->probe_pricing",
        "probe_pricing[themes contains price]->__end__",
    ]
    assert report.synthesis.ran is True
    assert report.synthesis.model == "sonnet"
    assert report.synthesis.theme_count == 2
    assert report.synthesis.summary_peek and report.synthesis.summary_peek.startswith("The panel converged")
    per_model = {m.model: m for m in report.model_rollup}
    # Rollup keys are canonical model ids (aliases resolved).
    assert per_model[resolve_alias("haiku")].cost_usd == 0.009
    assert per_model[resolve_alias("sonnet")].cost_usd == 0.02
    assert report.warnings == ["low-n: 3 panelists"]
    assert report.instrument_name == "pricing-discovery"


def test_inspect_failure_stats_reconstructed_for_flat_shape():
    """Flat shape has no persisted failure_stats — inspect reconstructs."""
    data = _flat_result(n_errors=2)
    assert "failure_stats" not in data
    report = build_inspect_report(data)
    assert report.failure_stats.total_pairs == 6
    assert report.failure_stats.errored_pairs == 4
    assert set(report.failure_stats.errored_personas) == {"Persona_0", "Persona_1"}


def test_inspect_handles_panelist_level_error():
    data = _flat_result(n_errors=0)
    data["results"][0]["error"] = "TimeoutError: upstream 503"
    data["results"][0]["responses"] = []
    report = build_inspect_report(data)
    assert report.personas[0].panelist_error == "TimeoutError: upstream 503"
    assert report.failure_stats.failed_panelists == 1
    assert report.failure_stats.errored_pairs >= 2  # all questions counted errored


def test_inspect_summary_peek_truncates_long_summary():
    data = _rounds_result()
    data["synthesis"]["summary"] = "x" * 1000
    report = build_inspect_report(data)
    assert report.synthesis.summary_peek is not None
    assert len(report.synthesis.summary_peek) == 301  # 300 + ellipsis


def test_inspect_model_rollup_merges_alias_and_canonical(monkeypatch):
    """Regression for sp-f9jg: panelist keyed by alias must bucket with
    metadata.cost.per_model keyed by canonical id (no duplicate rows)."""
    # Pin a deterministic alias map so the test doesn't depend on the
    # user's ~/.synthpanel/aliases.yaml or env overrides.
    monkeypatch.setenv(
        "SYNTHPANEL_MODEL_ALIASES",
        json.dumps({"haiku": "claude-haiku-4-5-20251001"}),
    )
    from synth_panel.llm import aliases as aliases_mod

    aliases_mod._reset_cache()
    try:
        data = {
            "id": "result-merge",
            "created_at": "2026-04-25T00:00:00+00:00",
            "model": "haiku",
            "persona_count": 1,
            "question_count": 1,
            "results": [
                {
                    "persona": "P0",
                    "responses": [{"question": "q", "response": "a", "follow_up": False}],
                    "usage": {"input_tokens": 30, "output_tokens": 20},
                    "model": "haiku",  # alias
                }
            ],
            "metadata": {
                "cost": {
                    "per_model": {
                        # Canonical id, as written by metadata.py:135.
                        "claude-haiku-4-5-20251001": {"tokens": 50, "cost_usd": 0.001},
                    }
                }
            },
        }

        report = build_inspect_report(data)

        assert len(report.model_rollup) == 1, (
            f"alias and canonical must merge into one row, got {[m.model for m in report.model_rollup]}"
        )
        row = report.model_rollup[0]
        assert row.model == "claude-haiku-4-5-20251001"
        assert row.input_tokens == 30
        assert row.output_tokens == 20
        assert row.total_tokens == 50
        assert row.cost_usd == 0.001
        assert row.personas == 1
    finally:
        aliases_mod._reset_cache()


def test_format_inspect_text_includes_all_sections():
    text = format_inspect_text(build_inspect_report(_rounds_result()))
    for section in (
        "Panel Result:",
        "Per-Persona Summary",
        "Per-Model Rollup",
        "Synthesis",
        "Failure Stats",
    ):
        assert section in text
    assert "pricing-discovery" in text
    assert "discovery[themes contains price]" not in text  # path formatting correctness
    assert "themes contains price" in text


def test_per_persona_summary_aligns_non_ascii_names():
    """Persona names with CJK / accents / emoji must align in the summary.

    Regression: SP#298 — naive ``f"{name:<30}"`` counted code points, so
    a row with ``"王芳"`` shifted ``model=...`` left of an ASCII row.
    """
    from synth_panel.text_width import display_width

    data = _flat_result(n_errors=0)
    data["results"][0]["persona"] = "Naoko 🌸"
    data["results"][1]["persona"] = "José Martínez"
    data["results"][2]["persona"] = "王芳"

    text = format_inspect_text(build_inspect_report(data))

    # Each per-persona line is "  <name padded>  model=...  responses=...  errors=...".
    # The "model=" token should appear at the same display offset on every row.
    rows = [ln for ln in text.splitlines() if "model=haiku" in ln]
    assert len(rows) == 3
    offsets = {display_width(row[: row.index("model=")]) for row in rows}
    assert len(offsets) == 1, f"per-persona rows misaligned: {offsets}"


def test_cli_inspect_with_path(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved.json"
        path.write_text(json.dumps(_rounds_result()), encoding="utf-8")

        rc = main(["panel", "inspect", str(path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Panel Result:" in out
        assert "Per-Model Rollup" in out
        assert "sonnet" in out


def test_cli_inspect_json_output(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved.json"
        path.write_text(json.dumps(_flat_result()), encoding="utf-8")

        rc = main(["--output-format", "json", "panel", "inspect", str(path)])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["message"] == "Panel inspect"
        assert payload["inspect"]["model"] == "haiku"
        assert payload["inspect"]["persona_count"] == 3
        assert payload["inspect"]["has_rounds_shape"] is False


def test_cli_inspect_not_found(capsys):
    rc = main(["panel", "inspect", "result-does-not-exist-xxx"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_cli_inspect_invalid_json(capsys):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.json"
        path.write_text("{not-json", encoding="utf-8")
        rc = main(["panel", "inspect", str(path)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "not valid JSON" in err
