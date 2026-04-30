"""Tests for checkpoint run listing and pruning (sy-67b / GH-314).

Covers:
- parse_duration: valid and invalid formats
- list_runs: empty root, normal runs, malformed entries
- prune_runs: --older-than, --keep, --dry-run, in-progress protection
- CLI wiring: synthpanel runs prune / list
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from synth_panel.checkpoint import (
    PanelCheckpoint,
    _is_in_progress,
    checkpoint_dir_for,
    list_runs,
    parse_duration,
    prune_runs,
)
from synth_panel.main import main

# ---------------------------------------------------------------------------
# parse_duration
# ---------------------------------------------------------------------------


class TestParseDuration:
    def test_days(self) -> None:
        assert parse_duration("7d") == timedelta(days=7)

    def test_hours(self) -> None:
        assert parse_duration("24h") == timedelta(hours=24)

    def test_weeks(self) -> None:
        assert parse_duration("2w") == timedelta(weeks=2)

    def test_minutes(self) -> None:
        assert parse_duration("90m") == timedelta(minutes=90)

    def test_fractional(self) -> None:
        assert parse_duration("1.5d") == timedelta(days=1.5)

    def test_invalid_unit(self) -> None:
        with pytest.raises(ValueError, match="invalid duration"):
            parse_duration("7x")

    def test_invalid_no_unit(self) -> None:
        with pytest.raises(ValueError, match="invalid duration"):
            parse_duration("7")

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError, match="invalid duration"):
            parse_duration("")


# ---------------------------------------------------------------------------
# _is_in_progress
# ---------------------------------------------------------------------------


class TestIsInProgress:
    def _make(self, remaining: list[str], abort_reason: str | None) -> PanelCheckpoint:
        now = datetime.now(timezone.utc).isoformat()
        return PanelCheckpoint(
            run_id="run-test",
            created_at=now,
            updated_at=now,
            config_fingerprint="abc",
            config={},
            remaining=remaining,
            abort_reason=abort_reason,
        )

    def test_completed_run_not_in_progress(self) -> None:
        ckpt = self._make(remaining=[], abort_reason=None)
        assert not _is_in_progress(ckpt)

    def test_aborted_run_not_in_progress(self) -> None:
        ckpt = self._make(remaining=["alice"], abort_reason="signal:SIGINT")
        assert not _is_in_progress(ckpt)

    def test_running_or_crashed_is_in_progress(self) -> None:
        ckpt = self._make(remaining=["alice"], abort_reason=None)
        assert _is_in_progress(ckpt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_run(
    root: Path,
    run_id: str,
    *,
    remaining: list[str] | None = None,
    abort_reason: str | None = None,
    updated_at: datetime | None = None,
) -> None:
    """Write a checkpoint run directly (bypasses save_checkpoint's now() stamp)."""
    now = datetime.now(timezone.utc)
    if updated_at is None:
        updated_at = now
    directory = checkpoint_dir_for(run_id, root)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "run_id": run_id,
        "created_at": now.isoformat(),
        "updated_at": updated_at.isoformat(),
        "config_fingerprint": "deadbeef",
        "config": {},
        "completed": [],
        "remaining": remaining or [],
        "usage": {},
        "abort_reason": abort_reason,
    }
    (directory / "state.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path) == []

    def test_missing_root_returns_empty(self, tmp_path: Path) -> None:
        missing = tmp_path / "no-such-dir"
        assert list_runs(missing) == []

    def test_single_completed_run(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "run-20240101T000000Z-abc123")
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-20240101T000000Z-abc123"
        assert not runs[0]["in_progress"]
        assert runs[0]["remaining"] == 0

    def test_in_progress_run_flagged(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "run-a", remaining=["alice", "bob"])
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0]["in_progress"] is True
        assert runs[0]["remaining"] == 2

    def test_malformed_entry_reported(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "bad-run"
        bad_dir.mkdir()
        (bad_dir / "state.json").write_text("{not valid json", encoding="utf-8")
        runs = list_runs(tmp_path)
        assert len(runs) == 1
        assert runs[0].get("malformed") is True

    def test_directories_without_state_json_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "some-dir").mkdir()
        assert list_runs(tmp_path) == []


# ---------------------------------------------------------------------------
# prune_runs
# ---------------------------------------------------------------------------


class TestPruneRuns:
    def test_requires_at_least_one_flag(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            prune_runs(root=tmp_path)

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        assert prune_runs(root=tmp_path, keep=5) == []

    def test_keep_n_prunes_oldest(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc)
        for i, age_days in enumerate([10, 5, 1]):
            _write_run(tmp_path, f"run-{i:03d}", updated_at=now - timedelta(days=age_days))

        pruned = prune_runs(root=tmp_path, keep=2)
        assert len(pruned) == 1
        assert "run-000" in pruned  # oldest (10 days ago)

    def test_older_than_prunes_by_age(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=10))
        _write_run(tmp_path, "run-new", updated_at=now - timedelta(days=1))

        pruned = prune_runs(root=tmp_path, older_than=timedelta(days=7))
        assert pruned == ["run-old"]
        assert not (tmp_path / "run-old").exists()
        assert (tmp_path / "run-new").exists()

    def test_dry_run_does_not_delete(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=10))

        pruned = prune_runs(root=tmp_path, older_than=timedelta(days=7), dry_run=True)
        assert pruned == ["run-old"]
        assert (tmp_path / "run-old").exists()

    def test_in_progress_never_pruned(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-active", remaining=["alice"], updated_at=now - timedelta(days=30))
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=30))

        pruned = prune_runs(root=tmp_path, older_than=timedelta(days=7))
        assert "run-old" in pruned
        assert "run-active" not in pruned
        assert (tmp_path / "run-active").exists()

    def test_aborted_run_is_prunable(self, tmp_path: Path) -> None:
        now = datetime.now(timezone.utc)
        _write_run(
            tmp_path,
            "run-aborted",
            remaining=["bob"],
            abort_reason="signal:SIGINT",
            updated_at=now - timedelta(days=10),
        )

        pruned = prune_runs(root=tmp_path, older_than=timedelta(days=7))
        assert "run-aborted" in pruned

    def test_both_flags_are_union(self, tmp_path: Path) -> None:
        """--older-than OR --keep: prune if either condition is met."""
        now = datetime.now(timezone.utc)
        # 5 fresh runs and 1 old run
        for i in range(5):
            _write_run(tmp_path, f"run-new-{i}", updated_at=now - timedelta(hours=1))
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=30))

        # --keep=5 keeps newest 5, --older-than=7d prunes old one too
        pruned = prune_runs(root=tmp_path, keep=5, older_than=timedelta(days=7))
        assert "run-old" in pruned

    def test_keep_n_no_delete_when_fewer(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "run-a")
        _write_run(tmp_path, "run-b")
        pruned = prune_runs(root=tmp_path, keep=10)
        assert pruned == []


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCliRunsPrune:
    def test_prune_requires_flag(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--output-format", "text", "runs", "prune", "--root", str(tmp_path)])
        assert rc != 0
        captured = capsys.readouterr()
        assert "older-than" in captured.err or "keep" in captured.err

    def test_prune_dry_run_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-stale", updated_at=now - timedelta(days=10))

        rc = main(["runs", "prune", "--older-than", "7d", "--dry-run", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "run-stale" in out
        assert "dry-run" in out.lower() or "would prune" in out.lower()

    def test_prune_deletes_run(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=30))

        rc = main(["runs", "prune", "--older-than", "7d", "--root", str(tmp_path)])
        assert rc == 0
        assert not (tmp_path / "run-old").exists()

    def test_prune_nothing_to_prune(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["runs", "prune", "--keep", "10", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Nothing to prune" in out

    def test_prune_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        now = datetime.now(timezone.utc)
        _write_run(tmp_path, "run-old", updated_at=now - timedelta(days=10))

        rc = main(["--output-format", "json", "runs", "prune", "--older-than", "7d", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["count"] == 1
        assert "run-old" in data["pruned"]

    def test_list_shows_runs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_run(tmp_path, "run-abc")
        rc = main(["runs", "list", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "run-abc" in out

    def test_list_empty(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["runs", "list", "--root", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No runs" in out

    def test_invalid_duration_returns_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["runs", "prune", "--older-than", "badval", "--root", str(tmp_path)])
        assert rc != 0
        err = capsys.readouterr().err
        assert "invalid duration" in err
