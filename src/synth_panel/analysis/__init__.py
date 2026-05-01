"""Panel-result analysis utilities.

Two layers live here:

* Inspection helpers (:mod:`synth_panel.analysis.inspect`) — schema
  walkers and no-LLM result summaries.
* Structured-response aggregation (sp-2hpi scaffolding) —
  :mod:`synth_panel.analysis.distribution` computes per-question
  distributions from ``response_schema``-typed responses, and
  :mod:`synth_panel.analysis.subgroup` splits those distributions by
  persona field.

Heavy statistical analysis that's entangled with the legacy
narrative-synthesis pipeline continues to live in
:mod:`synth_panel.analyze`.
"""

from synth_panel.analysis.distribution import (
    InvalidResponseSchemaError,
    coerce_enum_value,
    coerce_scale_value,
    coerce_tagged_themes,
    distribution_for_question,
)
from synth_panel.analysis.inspect import (
    InspectReport,
    build_inspect_report,
    format_inspect_text,
)
from synth_panel.analysis.subgroup import (
    AUTO_BIN_FIELDS,
    UnknownPersonaFieldError,
    analyze_subgroup,
    auto_bin_value,
    format_subgroup_text,
    subgroup_breakdown,
)

__all__ = [
    "AUTO_BIN_FIELDS",
    "InspectReport",
    "InvalidResponseSchemaError",
    "UnknownPersonaFieldError",
    "analyze_subgroup",
    "auto_bin_value",
    "build_inspect_report",
    "coerce_enum_value",
    "coerce_scale_value",
    "coerce_tagged_themes",
    "distribution_for_question",
    "format_inspect_text",
    "format_subgroup_text",
    "subgroup_breakdown",
]
