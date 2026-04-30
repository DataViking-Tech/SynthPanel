"""Live terminal progress bar for panel runs (CLI text mode only).

Renders to stderr via carriage-return overwrites; never touches stdout.
Caller is responsible for the guard: only instantiate when
sys.stdout.isatty() is True and output format is TEXT.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Any

from synth_panel.cost import resolve_cost


class PanelProgressBar:
    """Thread-safe in-place progress display written to stderr.

    Instantiate before run_panel_parallel, call update() from the
    on_panelist_complete callback, and call close() when the run ends.
    """

    def __init__(self, total: int, model: str, *, file: Any = None) -> None:
        self._total = total
        self._model = model
        self._completed = 0
        self._cost = 0.0
        self._start = time.monotonic()
        self._lock = threading.Lock()
        self._out = file if file is not None else sys.stderr
        self._bar_width = 20
        self._rendered = False

    def update(self, usage: Any) -> None:
        """Record one completed panelist and redraw."""
        try:
            priced = resolve_cost(usage, self._model)
            cost_delta = priced.total_cost
        except Exception:
            cost_delta = 0.0

        with self._lock:
            self._completed += 1
            self._cost += cost_delta
            self._render()

    def close(self) -> None:
        """Emit a trailing newline so the next stdout line is not overwritten."""
        with self._lock:
            if self._rendered:
                print("", file=self._out, flush=True)

    def _render(self) -> None:
        """Redraw the bar (caller must hold self._lock)."""
        self._rendered = True
        pct = self._completed / self._total if self._total else 1.0
        filled = int(self._bar_width * pct)
        if filled >= self._bar_width:
            bar = "=" * self._bar_width
        else:
            bar = "=" * filled + ">" + " " * (self._bar_width - filled - 1)

        elapsed = time.monotonic() - self._start
        elapsed_str = _fmt_secs(elapsed)

        if 0 < self._completed < self._total:
            eta_secs = elapsed / self._completed * (self._total - self._completed)
            eta_str = f"ETA {_fmt_secs(eta_secs)}"
        elif self._completed >= self._total:
            eta_str = "done"
        else:
            eta_str = "ETA ..."

        line = f"\r  [{bar}] {self._completed}/{self._total}  {elapsed_str} elapsed  {eta_str}  ${self._cost:.4f}"

        cols = shutil.get_terminal_size(fallback=(80, 24)).columns
        if len(line) > cols:
            line = line[:cols]
        else:
            # Pad with spaces to erase any characters left from a longer prior render.
            line = line.ljust(cols)

        print(line, end="", file=self._out, flush=True)


def _fmt_secs(s: float) -> str:
    s = int(s)
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"
