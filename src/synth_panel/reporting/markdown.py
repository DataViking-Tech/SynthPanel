"""Markdown renderer stub for post-hoc panel reporting.

Scaffold for sp-viz-layer T1 — body raises :class:`NotImplementedError`.
T3 fills in the 10-section renderer, provenance table, and synthetic-panel
banner/footer per structure.md §2 and §7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_panel.analysis.inspect import InspectReport


def render_markdown(
    report: InspectReport,
    raw: dict[str, Any],
    *,
    source_path: str | None = None,
) -> str:
    """Render a complete Markdown report for a saved panel result.

    Pure function — no I/O, no mutation of ``raw``. The caller is
    responsible for writing the returned string to stdout or disk.

    Sections (in order) are defined in structure.md §2; the mandatory
    synthetic-panel banner and footer are defined in §7 and are not
    configurable.
    """

    raise NotImplementedError("sp-viz-layer T3: implement render_markdown")
