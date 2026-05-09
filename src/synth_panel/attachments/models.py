"""Typed shapes for attachment refs.

The ref is the *only* thing that lives in result JSON. Bytes always
sit in the CAS; the ref carries enough metadata for readback layers to
resolve, render, and audit without loading bytes.
"""

from __future__ import annotations

from typing import Literal, TypedDict

AttachmentKind = Literal["image", "pdf", "url", "url_screenshot"]


class _AttachmentRefRequired(TypedDict):
    id: str
    kind: AttachmentKind
    sha256: str
    content_type: str
    byte_size: int


class AttachmentRef(_AttachmentRefRequired, total=False):
    """Per-attachment ref persisted in ``<result_id>.attachments/refs.json``.

    The fields in :class:`_AttachmentRefRequired` are sufficient to
    resolve the blob in CAS. Optional fields below carry provenance
    (``source_uri``, ``fetched_at``, ``etag``, ``final_uri``), display
    metadata (``alt_text``, ``dims``, ``thumb_sha256``), and audit notes
    (``redaction_note``). Multi-base TypedDict + ``total=False`` is used
    in place of ``NotRequired`` so we stay compatible with Python 3.10.
    """

    source_uri: str
    final_uri: str
    fetched_at: str
    etag: str
    alt_text: str
    dims: tuple[int, int]
    thumb_sha256: str
    redaction_note: str
