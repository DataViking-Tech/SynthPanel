"""Post-hoc reporting for saved panel results.

Public API (stubs — see sp-viz-layer T2/T3 for implementations):

* :func:`load_panel_json` — resolve a RESULT_ID-or-path to a parsed dict.
* :func:`render_markdown` — render an :class:`InspectReport` + raw panel dict
  to a complete Markdown document.
* :exc:`ReportLoadError` — raised by :func:`load_panel_json` on missing,
  malformed, or mis-shaped inputs.
"""

from __future__ import annotations

from synth_panel.reporting.loader import ReportLoadError, load_panel_json
from synth_panel.reporting.markdown import render_markdown

__all__ = ["ReportLoadError", "load_panel_json", "render_markdown"]
