"""Saved-result discoverability + report provenance (#525).

Covers the v1.5.3 dogfood gaps: a ``--save`` artifact must be
rediscoverable through a CLI list command (not only a filesystem search),
resolvable by its stable ID, and carry enough provenance that ``report``
no longer degrades every field to ``(unknown)``.
"""

from __future__ import annotations

import json

import pytest

from synth_panel.cost import PRICING_SNAPSHOT_DATE, CostEstimate, TokenUsage
from synth_panel.main import main
from synth_panel.metadata import build_metadata


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))


def _save_with_metadata() -> str:
    """Persist a minimal result the way the CLI ``--save`` path now does."""
    from synth_panel.mcp.data import save_panel_result

    usage = TokenUsage(input_tokens=10, output_tokens=20)
    cost = CostEstimate(input_cost=0.0005, output_cost=0.0005)
    metadata = build_metadata(
        panelist_model="gemini-2.5-flash",
        panelist_usage=usage,
        panelist_cost=cost,
        total_usage=usage,
        total_cost=cost,
        persona_count=1,
        question_count=1,
    )
    return save_panel_result(
        results=[{"persona": "Alice", "responses": [{"response": "ok"}]}],
        model="gemini-2.5-flash",
        total_usage=usage.to_dict(),
        total_cost=cost.format_usd(),
        persona_count=1,
        question_count=1,
        instrument_name="probe",
        metadata=metadata,
    )


class TestProvenancePersisted:
    def test_metadata_block_round_trips(self):
        from synth_panel.mcp.data import get_panel_result

        rid = _save_with_metadata()
        data = get_panel_result(rid)

        assert "metadata" in data
        assert data["metadata"]["version"]["synthpanel"]
        assert data["metadata"]["version"]["python"]
        assert data["metadata"]["config_hash"]
        assert data["metadata"]["cost"]["pricing_snapshot_date"] == PRICING_SNAPSHOT_DATE

    def test_report_provenance_is_populated(self):
        from synth_panel.analysis.inspect import build_inspect_report
        from synth_panel.mcp.data import get_panel_result
        from synth_panel.reporting import render_markdown

        rid = _save_with_metadata()
        data = get_panel_result(rid)
        md = render_markdown(build_inspect_report(data), data, source_path=None)

        # The dogfood report showed all four of these as (unknown)/(not recorded).
        assert "(not recorded)" not in md
        assert "synthpanel_version | (unknown)" not in md
        assert "python_version | (unknown)" not in md
        assert "pricing_snapshot_date | (unknown)" not in md
        assert PRICING_SNAPSHOT_DATE in md


class TestResultsList:
    def test_lists_saved_result_by_id(self, capsys):
        rid = _save_with_metadata()
        code = main(["results", "list"])
        out = capsys.readouterr().out
        assert code == 0
        assert rid in out

    def test_json_mode_emits_results_array(self, capsys):
        rid = _save_with_metadata()
        code = main(["--output-format", "json", "results", "list"])
        out = capsys.readouterr().out
        assert code == 0
        payload = json.loads(out)
        assert payload["count"] == 1
        assert payload["results"][0]["id"] == rid

    def test_empty_store_reports_cleanly(self, capsys):
        code = main(["results", "list"])
        out = capsys.readouterr().out
        assert code == 0
        assert "No saved results found" in out


class TestResultsShow:
    def test_show_by_id_surfaces_path_and_provenance(self, capsys):
        rid = _save_with_metadata()
        code = main(["results", "show", rid])
        out = capsys.readouterr().out
        assert code == 0
        assert rid in out
        assert "saved_path:" in out
        assert f"{rid}.json" in out
        assert PRICING_SNAPSHOT_DATE in out
        # The canonical follow-up command is advertised.
        assert f"synthpanel report {rid}" in out

    def test_json_mode_includes_saved_path(self, capsys):
        rid = _save_with_metadata()
        code = main(["--output-format", "json", "results", "show", rid])
        out = capsys.readouterr().out
        assert code == 0
        payload = json.loads(out)
        assert payload["saved_path"].endswith(f"{rid}.json")
        assert payload["metadata"]["config_hash"]

    def test_missing_id_errors(self, capsys):
        code = main(["results", "show", "result-does-not-exist"])
        err = capsys.readouterr().err
        assert code == 1
        assert "not found" in err


class TestRunsListPointsAtResults:
    def test_empty_runs_list_hints_results_store(self, capsys, tmp_path):
        # Distinct checkpoint root with no runs → user who ran --save should
        # be pointed at the results store rather than concluding it vanished.
        code = main(["runs", "list", "--root", str(tmp_path / "ckpt")])
        out = capsys.readouterr().out
        assert code == 0
        assert "synthpanel results list" in out
