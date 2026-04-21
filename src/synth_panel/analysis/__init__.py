"""Panel-result analysis utilities (inspection, schema walking).

This package intentionally stays narrow: heavy statistical analysis
lives in :mod:`synth_panel.analyze`. Here we collect lightweight,
no-LLM result introspection helpers that downstream tooling (CLI,
MCP, CI gates) can compose.
"""

from synth_panel.analysis.inspect import (
    InspectReport,
    build_inspect_report,
    format_inspect_text,
)

__all__ = [
    "InspectReport",
    "build_inspect_report",
    "format_inspect_text",
]
