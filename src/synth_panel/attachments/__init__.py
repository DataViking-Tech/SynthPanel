"""Question/panel attachments — multimodal stimuli (hq-pojo).

Four pieces compose attachment support:

* :mod:`synth_panel.attachments.models` — typed records for
  attachment kinds and refs (hq-qd7r).
* :mod:`synth_panel.attachments.store` — content-addressable
  persistence (hq-qd7r). A global CAS at
  ``$SYNTH_PANEL_ATTACHMENT_DIR`` (or
  ``$SYNTH_PANEL_DATA_DIR/attachments``, defaulting to
  ``~/.synthpanel/attachments``) shards files under a 2-char sha256
  prefix; per-result ``refs.json`` indexes live in a
  ``<result_id>.attachments/`` sidecar. Result JSON holds only
  attachment ids; bytes are loaded on demand via
  :func:`store.read_blob`.
* :mod:`synth_panel.attachments.filter` — per-persona stratification
  (hq-iczd). Decides which attachments each panelist sees based on
  predicate filters against persona traits.
* :mod:`synth_panel.attachments.pdf` — PDF ingest decision tree
  (hq-glz6 / D-phase hq-31t8). Cheap local probes pick between native
  PDF submission, text extraction, and page-as-image rendering before
  any LLM cost is incurred.
"""

from __future__ import annotations

from synth_panel.attachments.filter import count_strata, filter_attachments
from synth_panel.attachments.models import AttachmentKind, AttachmentRef
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
from synth_panel.attachments.store import (
    attachments_dir,
    read_blob,
    refs_path,
    write_blob,
)

__all__ = [
    "DEFAULT_OPTIONS",
    "AttachmentKind",
    "AttachmentRef",
    "PdfEncryptedError",
    "PdfError",
    "PdfMissingDependencyError",
    "PdfOptions",
    "PdfOversizeScannedError",
    "PdfPlan",
    "PdfProbe",
    "PdfStrategy",
    "SubmissionMode",
    "attachments_dir",
    "count_strata",
    "decide_from_probe",
    "extract_text_chunks",
    "filter_attachments",
    "plan_pdf",
    "probe_pdf",
    "read_blob",
    "refs_path",
    "render_pages_as_png",
    "write_blob",
]
