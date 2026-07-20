"""Panel-result analysis utilities.

Two layers live here:

* Inspection helpers (:mod:`althing.analysis.inspect`) — schema
  walkers and no-LLM result summaries.
* Structured-response aggregation (sp-2hpi scaffolding) —
  :mod:`althing.analysis.distribution` computes per-question
  distributions from ``response_schema``-typed responses, and
  :mod:`althing.analysis.subgroup` splits those distributions by
  persona field.

Heavy statistical analysis that's entangled with the legacy
narrative-synthesis pipeline continues to live in
:mod:`althing.analyze`.
"""

from althing.analysis.distribution import (
    InvalidResponseSchemaError,
    coerce_enum_value,
    coerce_scale_value,
    coerce_tagged_themes,
    distribution_for_question,
)
from althing.analysis.inspect import (
    InspectReport,
    build_inspect_report,
    format_inspect_text,
)
from althing.analysis.subgroup import (
    UnknownPersonaFieldError,
    subgroup_breakdown,
)

__all__ = [
    "InspectReport",
    "InvalidResponseSchemaError",
    "UnknownPersonaFieldError",
    "build_inspect_report",
    "coerce_enum_value",
    "coerce_scale_value",
    "coerce_tagged_themes",
    "distribution_for_question",
    "format_inspect_text",
    "subgroup_breakdown",
]
