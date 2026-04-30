"""Tests for synth_panel.cost_summary aggregation (sy-kmw1)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from synth_panel.cost_summary import (
    RunInfo,
    SummaryReport,
    aggregate_runs,
    default_runs_dir,
    discover_run_files,
    format_text,
    parse_run,
    parse_since,
    summarize,
    to_json_payload,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_run(
    runs_dir: Path,
    *,
    name: str,
    created_at: str = "2026-04-15T10:00:00+00:00",
    model: str = "claude-sonnet-4-6",
    models: list[str] | None = None,
    total_cost: str = "$0.0150",
    results: list[dict] | None = None,
    persona_count: int | None = None,
    question_count: int = 3,
) -> Path:
    """Write a result-shaped JSON file to *runs_dir* and return its path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    rs = (
        results
        if results is not None
        else [
            {
                "persona": "Alice",
                "responses": [{"q": "1"}, {"q": "2"}, {"q": "3"}],
                "cost": "$0.0075",
                "model": model,
            },
            {
                "persona": "Bob",
                "responses": [{"q": "1"}, {"q": "2"}, {"q": "3"}],
                "cost": "$0.0075",
                "model": model,
            },
        ]
    )
    body: dict = {
        "created_at": created_at,
        "model": model,
        "persona_count": persona_count if persona_count is not None else len(rs),
        "question_count": question_count,
        "total_cost": total_cost,
        "results": rs,
    }
    if models is not None:
        body["models"] = models
    p = runs_dir / f"{name}.json"
    p.write_text(json.dumps(body), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# parse_since
# ---------------------------------------------------------------------------


class TestParseSince:
    def test_date_only(self):
        dt = parse_since("2026-04-01")
        assert dt.year == 2026 and dt.month == 4 and dt.day == 1

    def test_iso_z_suffix(self):
        dt = parse_since("2026-04-01T12:00:00Z")
        assert dt.tzinfo is not None
        assert dt.hour == 12

    def test_iso_offset(self):
        dt = parse_since("2026-04-01T12:00:00+00:00")
        assert dt.tzinfo is not None

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_since("not-a-date")


# ---------------------------------------------------------------------------
# discover_run_files
# ---------------------------------------------------------------------------


class TestDiscoverRunFiles:
    def test_empty_dir(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        assert discover_run_files(d) == []

    def test_missing_dir_returns_empty(self, tmp_path):
        d = tmp_path / "nope"
        assert discover_run_files(d) == []

    def test_excludes_pre_extend(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        (d / "result-aaa.json").write_text("{}", encoding="utf-8")
        (d / "result-aaa.pre-extend.json").write_text("{}", encoding="utf-8")
        files = discover_run_files(d)
        assert [p.name for p in files] == ["result-aaa.json"]

    def test_excludes_synthesis_sidecars(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        (d / "result-aaa.json").write_text("{}", encoding="utf-8")
        (d / "result-aaa.synthesis-2026-04-16T01-00-00.json").write_text("{}", encoding="utf-8")
        files = discover_run_files(d)
        assert [p.name for p in files] == ["result-aaa.json"]


# ---------------------------------------------------------------------------
# parse_run
# ---------------------------------------------------------------------------


class TestParseRun:
    def test_minimal_valid_run(self, tmp_path):
        p = _write_run(tmp_path / "results", name="result-x")
        info = parse_run(p)
        assert info is not None
        assert info.run_id == "result-x"
        assert info.total_cost_usd == pytest.approx(0.0150)
        assert info.persona_count == 2
        assert info.turn_count == 6  # 2 panelists * 3 responses
        assert info.is_partial is False

    def test_missing_total_cost_skipped(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        p = d / "result-no-cost.json"
        p.write_text(json.dumps({"results": []}), encoding="utf-8")
        assert parse_run(p) is None

    def test_missing_results_skipped(self, tmp_path):
        d = tmp_path / "results"
        d.mkdir()
        p = d / "result-no-results.json"
        p.write_text(json.dumps({"total_cost": "$0.01"}), encoding="utf-8")
        assert parse_run(p) is None

    def test_malformed_json_returns_none(self, tmp_path, caplog):
        d = tmp_path / "results"
        d.mkdir()
        p = d / "result-broken.json"
        p.write_text("{not valid", encoding="utf-8")
        info = parse_run(p)
        assert info is None

    def test_partial_when_results_lt_persona_count(self, tmp_path):
        p = _write_run(
            tmp_path / "results",
            name="r",
            persona_count=5,
            results=[
                {"persona": "A", "responses": [{}], "cost": "$0.0010"},
            ],
        )
        info = parse_run(p)
        assert info is not None
        assert info.is_partial is True

    def test_partial_when_all_panelists_zero_turns(self, tmp_path):
        p = _write_run(
            tmp_path / "results",
            name="r",
            results=[
                {"persona": "A", "responses": [], "cost": "$0.0000"},
                {"persona": "B", "responses": [], "cost": "$0.0000"},
            ],
            total_cost="$0.0000",
        )
        info = parse_run(p)
        assert info is not None
        assert info.is_partial is True

    def test_per_panelist_model_falls_back_to_top_level(self, tmp_path):
        p = _write_run(
            tmp_path / "results",
            name="r",
            model="haiku",
            results=[
                {"persona": "A", "responses": [{}], "cost": "$0.0010"},
            ],
        )
        info = parse_run(p)
        assert info is not None
        assert info.panelists[0].model == "haiku"

    def test_cost_str_with_no_dollar_sign(self, tmp_path):
        p = _write_run(tmp_path / "results", name="r", total_cost="0.05")
        info = parse_run(p)
        assert info is not None
        assert info.total_cost_usd == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# aggregate_runs
# ---------------------------------------------------------------------------


def _make_run(
    *,
    run_id: str,
    created_at: datetime | None = None,
    model: str = "sonnet",
    total_cost: float = 0.01,
    panelists: list[tuple[str, float, int]] | None = None,
    persona_count: int | None = None,
    is_partial: bool = False,
) -> RunInfo:
    """Hand-rolled RunInfo for aggregator tests."""
    from synth_panel.cost_summary import PanelistCost

    pl = [PanelistCost(model=m, cost_usd=c, turns=t) for m, c, t in (panelists or [])]
    return RunInfo(
        path=Path(f"/tmp/{run_id}.json"),
        run_id=run_id,
        created_at=created_at,
        model=model,
        models=[model],
        total_cost_usd=total_cost,
        panelists=pl,
        persona_count=persona_count if persona_count is not None else len(pl),
        question_count=3,
        is_partial=is_partial,
    )


class TestAggregateRuns:
    def test_empty(self):
        report = aggregate_runs([])
        assert report.run_count == 0
        assert report.total_cost_usd == 0.0

    def test_total_sums_all_runs(self):
        r1 = _make_run(run_id="a", total_cost=0.01)
        r2 = _make_run(run_id="b", total_cost=0.02)
        r3 = _make_run(run_id="c", total_cost=0.04)
        report = aggregate_runs([r1, r2, r3])
        assert report.run_count == 3
        assert report.total_cost_usd == pytest.approx(0.07)

    def test_by_model_attributes_panelist_costs(self):
        r1 = _make_run(
            run_id="a",
            total_cost=0.01,
            panelists=[("sonnet", 0.005, 3), ("sonnet", 0.005, 3)],
        )
        report = aggregate_runs([r1])
        assert "sonnet" in report.by_model
        assert report.by_model["sonnet"].cost_usd == pytest.approx(0.01)
        assert report.by_model["sonnet"].runs == 1
        assert report.by_model["sonnet"].turns == 6

    def test_by_model_synthesis_residue_lands_on_top_level_model(self):
        # Two panelists each $0.001, total $0.005 → $0.003 residue to top.
        r1 = _make_run(
            run_id="a",
            model="opus",
            total_cost=0.005,
            panelists=[("haiku", 0.001, 2), ("haiku", 0.001, 2)],
        )
        report = aggregate_runs([r1])
        assert report.by_model["haiku"].cost_usd == pytest.approx(0.002)
        assert report.by_model["opus"].cost_usd == pytest.approx(0.003)

    def test_by_model_run_count_per_model_run_pair(self):
        r1 = _make_run(
            run_id="a",
            total_cost=0.01,
            panelists=[("sonnet", 0.005, 3), ("sonnet", 0.005, 3)],
        )
        r2 = _make_run(
            run_id="b",
            total_cost=0.02,
            panelists=[("sonnet", 0.01, 4), ("haiku", 0.01, 4)],
        )
        report = aggregate_runs([r1, r2])
        # Sonnet appears in 2 runs, haiku in 1.
        assert report.by_model["sonnet"].runs == 2
        assert report.by_model["haiku"].runs == 1

    def test_by_month_groups_by_yyyy_mm(self):
        r1 = _make_run(
            run_id="a",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            total_cost=0.05,
        )
        r2 = _make_run(
            run_id="b",
            created_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
            total_cost=0.10,
        )
        r3 = _make_run(
            run_id="c",
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            total_cost=0.20,
        )
        report = aggregate_runs([r1, r2, r3])
        assert report.by_month == {
            "2026-04": pytest.approx(0.15),
            "2026-05": pytest.approx(0.20),
        }

    def test_since_filter_excludes_older_runs(self):
        r1 = _make_run(
            run_id="old",
            created_at=datetime(2026, 3, 15, tzinfo=timezone.utc),
            total_cost=0.05,
        )
        r2 = _make_run(
            run_id="new",
            created_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
            total_cost=0.10,
        )
        report = aggregate_runs([r1, r2], since=datetime(2026, 4, 1, tzinfo=timezone.utc))
        assert report.run_count == 1
        assert report.total_cost_usd == pytest.approx(0.10)

    def test_since_filter_skips_runs_with_no_timestamp(self):
        r1 = _make_run(run_id="ts-less", created_at=None, total_cost=0.05)
        r2 = _make_run(
            run_id="dated",
            created_at=datetime(2026, 4, 20, tzinfo=timezone.utc),
            total_cost=0.10,
        )
        report = aggregate_runs([r1, r2], since=datetime(2026, 4, 1, tzinfo=timezone.utc))
        assert report.run_count == 1
        assert report.skipped_count == 1

    def test_partial_count(self):
        r1 = _make_run(run_id="ok", total_cost=0.01)
        r2 = _make_run(run_id="part", total_cost=0.005, is_partial=True)
        report = aggregate_runs([r1, r2])
        assert report.partial_count == 1


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


class TestFormatText:
    def test_empty_report(self):
        out = format_text(SummaryReport())
        assert "No runs found" in out

    def test_total_line_uses_singular_for_one_run(self):
        r1 = _make_run(run_id="a", total_cost=0.05)
        out = format_text(aggregate_runs([r1]))
        assert "across 1 run" in out
        assert "across 1 runs" not in out

    def test_partial_count_in_total_line(self):
        r1 = _make_run(run_id="a", total_cost=0.05)
        r2 = _make_run(run_id="b", total_cost=0.01, is_partial=True)
        out = format_text(aggregate_runs([r1, r2]))
        assert "(1 partial)" in out

    def test_by_model_section(self):
        r1 = _make_run(
            run_id="a",
            total_cost=0.01,
            panelists=[("sonnet", 0.005, 3), ("sonnet", 0.005, 3)],
        )
        out = format_text(aggregate_runs([r1]))
        assert "By model:" in out
        assert "sonnet" in out

    def test_by_run_section(self):
        r1 = _make_run(run_id="run-aaa", total_cost=0.02)
        out = format_text(aggregate_runs([r1]), group_by="run")
        assert "By run:" in out
        assert "run-aaa" in out


class TestToJsonPayload:
    def test_top_level_keys(self):
        r1 = _make_run(
            run_id="a",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            total_cost=0.01,
            panelists=[("sonnet", 0.005, 3), ("sonnet", 0.005, 3)],
        )
        payload = to_json_payload(aggregate_runs([r1]))
        assert payload["run_count"] == 1
        assert payload["total_cost_usd"] == pytest.approx(0.01)
        assert "sonnet" in payload["by_model"]
        assert payload["by_month"] == {"2026-04": pytest.approx(0.01)}

    def test_by_run_payload_includes_run_list(self):
        r1 = _make_run(
            run_id="run-aaa",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            total_cost=0.02,
        )
        payload = to_json_payload(aggregate_runs([r1]), group_by="run")
        assert "by_run" in payload
        assert payload["by_run"][0]["run_id"] == "run-aaa"

    def test_payload_is_json_serializable(self):
        r1 = _make_run(
            run_id="a",
            created_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
            total_cost=0.01,
        )
        payload = to_json_payload(aggregate_runs([r1]), group_by="run")
        # Round-trip through json.dumps; must not raise.
        text = json.dumps(payload)
        assert isinstance(json.loads(text), dict)


# ---------------------------------------------------------------------------
# Integration: summarize() + default_runs_dir() + CLI
# ---------------------------------------------------------------------------


class TestSummarizeIntegration:
    def test_summarize_three_fake_runs(self, tmp_path):
        runs_dir = tmp_path / "results"
        _write_run(
            runs_dir,
            name="result-a",
            created_at="2026-04-01T10:00:00+00:00",
            model="sonnet",
            total_cost="$0.10",
            results=[
                {"persona": "A", "responses": [{}, {}, {}], "cost": "$0.05", "model": "sonnet"},
                {"persona": "B", "responses": [{}, {}, {}], "cost": "$0.05", "model": "sonnet"},
            ],
        )
        _write_run(
            runs_dir,
            name="result-b",
            created_at="2026-04-15T10:00:00+00:00",
            model="haiku",
            total_cost="$0.02",
            results=[
                {"persona": "C", "responses": [{}, {}], "cost": "$0.02", "model": "haiku"},
            ],
        )
        _write_run(
            runs_dir,
            name="result-c",
            created_at="2026-05-01T10:00:00+00:00",
            model="haiku",
            total_cost="$0.04",
            results=[
                {"persona": "D", "responses": [{}, {}], "cost": "$0.02", "model": "haiku"},
                {"persona": "E", "responses": [{}, {}], "cost": "$0.02", "model": "haiku"},
            ],
        )
        report = summarize(runs_dir)
        assert report.run_count == 3
        assert report.total_cost_usd == pytest.approx(0.16)
        assert report.by_model["sonnet"].cost_usd == pytest.approx(0.10)
        assert report.by_model["haiku"].cost_usd == pytest.approx(0.06)
        assert report.by_month["2026-04"] == pytest.approx(0.12)
        assert report.by_month["2026-05"] == pytest.approx(0.04)

    def test_since_filter_via_summarize(self, tmp_path):
        runs_dir = tmp_path / "results"
        _write_run(runs_dir, name="result-a", created_at="2026-03-01T10:00:00+00:00", total_cost="$0.10")
        _write_run(runs_dir, name="result-b", created_at="2026-04-15T10:00:00+00:00", total_cost="$0.05")
        report = summarize(runs_dir, since=datetime(2026, 4, 1, tzinfo=timezone.utc))
        assert report.run_count == 1
        assert report.total_cost_usd == pytest.approx(0.05)


class TestDefaultRunsDir:
    def test_respects_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
        assert default_runs_dir() == tmp_path / "results"

    def test_default_under_home(self, monkeypatch):
        monkeypatch.delenv("SYNTH_PANEL_DATA_DIR", raising=False)
        d = default_runs_dir()
        assert d.name == "results"
        assert d.parent.name == ".synthpanel"


# ---------------------------------------------------------------------------
# CLI smoke tests (sub-process to exercise dispatcher + parser)
# ---------------------------------------------------------------------------


class TestCliSmoke:
    def _run(self, env_dir: Path, *args: str) -> subprocess.CompletedProcess:
        env = {"PATH": "/usr/bin:/bin", "SYNTH_PANEL_DATA_DIR": str(env_dir)}
        return subprocess.run(
            [sys.executable, "-m", "synth_panel", "cost", "summary", *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_no_runs_text(self, tmp_path):
        result = self._run(tmp_path)
        assert result.returncode == 0
        assert "No runs found" in result.stdout

    def test_no_runs_json(self, tmp_path):
        result = self._run(tmp_path, "--format", "json")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["run_count"] == 0
        assert payload["total_cost_usd"] == 0.0

    def test_full_pipeline_json(self, tmp_path):
        runs_dir = tmp_path / "results"
        _write_run(runs_dir, name="result-a", total_cost="$0.05")
        _write_run(runs_dir, name="result-b", total_cost="$0.10")
        result = self._run(tmp_path, "--format", "json")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["run_count"] == 2
        assert payload["total_cost_usd"] == pytest.approx(0.15)

    def test_invalid_since_returns_error(self, tmp_path):
        result = self._run(tmp_path, "--since", "definitely-not-a-date")
        assert result.returncode == 1
        assert "invalid --since" in result.stderr

    def test_runs_dir_override(self, tmp_path):
        custom = tmp_path / "elsewhere"
        _write_run(custom, name="result-x", total_cost="$0.07")
        # Point SYNTH_PANEL_DATA_DIR somewhere empty; --runs-dir should win.
        result = self._run(tmp_path, "--runs-dir", str(custom), "--format", "json")
        assert result.returncode == 0
        payload = json.loads(result.stdout)
        assert payload["run_count"] == 1
        assert payload["total_cost_usd"] == pytest.approx(0.07)
