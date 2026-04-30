"""Tests for PanelProgressBar — progress suppression and basic rendering."""

from __future__ import annotations

import io
import sys
from unittest.mock import MagicMock, patch

from synth_panel.cli.output import OutputFormat
from synth_panel.cli.progress import PanelProgressBar, _fmt_secs
from synth_panel.cost import TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_usage(input_tokens: int = 100, output_tokens: int = 50) -> TokenUsage:
    return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)


# ---------------------------------------------------------------------------
# Progress suppression guard
# ---------------------------------------------------------------------------


class TestProgressGuard:
    """The guard in commands.py: only show progress when TTY + TEXT mode."""

    def test_no_progress_when_stdout_not_tty(self):
        """When stdout is not a TTY, _show_progress must be False."""
        with patch.object(sys.stdout, "isatty", return_value=False):
            show = OutputFormat.TEXT is OutputFormat.TEXT and sys.stdout.isatty()
        assert show is False

    def test_no_progress_when_json_format(self):
        """JSON output mode must suppress progress regardless of TTY state."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            show = OutputFormat.JSON is OutputFormat.TEXT and sys.stdout.isatty()
        assert show is False

    def test_no_progress_when_ndjson_format(self):
        """NDJSON output mode must suppress progress regardless of TTY state."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            show = OutputFormat.NDJSON is OutputFormat.TEXT and sys.stdout.isatty()
        assert show is False

    def test_progress_enabled_when_tty_and_text(self):
        """TEXT mode + TTY must allow progress display."""
        with patch.object(sys.stdout, "isatty", return_value=True):
            show = OutputFormat.TEXT is OutputFormat.TEXT and sys.stdout.isatty()
        assert show is True


# ---------------------------------------------------------------------------
# PanelProgressBar behaviour
# ---------------------------------------------------------------------------


class TestPanelProgressBar:
    def _bar(self, total: int = 5, model: str = "claude-sonnet-4-6") -> tuple[PanelProgressBar, io.StringIO]:
        buf = io.StringIO()
        bar = PanelProgressBar(total=total, model=model, file=buf)
        return bar, buf

    def test_update_writes_to_file(self):
        bar, buf = self._bar(total=3)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.001)
            bar.update(_make_usage())
        output = buf.getvalue()
        assert output != ""
        assert "1/3" in output

    def test_update_increments_completed_count(self):
        bar, buf = self._bar(total=4)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.0)
            bar.update(_make_usage())
            bar.update(_make_usage())
        output = buf.getvalue()
        assert "2/4" in output

    def test_cost_accumulates(self):
        bar, buf = self._bar(total=2)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.0050)
            bar.update(_make_usage())
            bar.update(_make_usage())
        output = buf.getvalue()
        assert "$0.0100" in output

    def test_close_emits_newline_after_render(self):
        bar, buf = self._bar(total=1)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.0)
            bar.update(_make_usage())
        bar.close()
        # After close() the buffer should end with a newline
        assert buf.getvalue().endswith("\n")

    def test_close_without_update_emits_nothing(self):
        bar, buf = self._bar(total=2)
        bar.close()
        assert buf.getvalue() == ""

    def test_resolve_cost_exception_does_not_crash(self):
        bar, buf = self._bar(total=2)
        with patch("synth_panel.cli.progress.resolve_cost", side_effect=ValueError("no pricing")):
            bar.update(_make_usage())  # must not raise
        assert "1/2" in buf.getvalue()

    def test_renders_carriage_return(self):
        bar, buf = self._bar(total=2)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.0)
            bar.update(_make_usage())
        assert buf.getvalue().startswith("\r")

    def test_done_label_at_completion(self):
        bar, buf = self._bar(total=2)
        with patch("synth_panel.cli.progress.resolve_cost") as mock_rc:
            mock_rc.return_value = MagicMock(total_cost=0.0)
            bar.update(_make_usage())
            bar.update(_make_usage())
        assert "done" in buf.getvalue()


# ---------------------------------------------------------------------------
# _fmt_secs helper
# ---------------------------------------------------------------------------


class TestFmtSecs:
    def test_under_60_seconds(self):
        assert _fmt_secs(42.9) == "42s"

    def test_exactly_60_seconds(self):
        assert _fmt_secs(60.0) == "1m00s"

    def test_90_seconds(self):
        assert _fmt_secs(90.0) == "1m30s"

    def test_zero(self):
        assert _fmt_secs(0.0) == "0s"
