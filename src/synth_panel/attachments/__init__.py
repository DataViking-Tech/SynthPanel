"""Question/panel attachments — multimodal stimuli (hq-pojo).

Two pieces compose attachment support:

* :mod:`synth_panel.attachments.filter` — per-persona stratification
  (hq-iczd). Decides which attachments each panelist sees based on
  predicate filters against persona traits.
* :mod:`synth_panel.attachments.pdf` — PDF ingest decision tree
  (hq-glz6 / D-phase hq-31t8). Cheap local probes pick between native
  PDF submission, text extraction, and page-as-image rendering before
  any LLM cost is incurred.
"""

from synth_panel.attachments.filter import count_strata, filter_attachments
from synth_panel.attachments.pdf import (
    DEFAULT_OPTIONS,
    PdfEncryptedError,
    PdfError,
    PdfMissingDependencyError,
    PdfOptions,
    PdfOversizeScannedError,
    PdfPlan,
    PdfProbe,
    PdfStrategy,
    SubmissionMode,
    decide_from_probe,
    extract_text_chunks,
    plan_pdf,
    probe_pdf,
    render_pages_as_png,
)

__all__ = [
    "DEFAULT_OPTIONS",
    "PdfEncryptedError",
    "PdfError",
    "PdfMissingDependencyError",
    "PdfOptions",
    "PdfOversizeScannedError",
    "PdfPlan",
    "PdfProbe",
    "PdfStrategy",
    "SubmissionMode",
    "count_strata",
    "decide_from_probe",
    "extract_text_chunks",
    "filter_attachments",
    "plan_pdf",
    "probe_pdf",
    "render_pages_as_png",
]
