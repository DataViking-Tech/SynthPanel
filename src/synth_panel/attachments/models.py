"""Typed shapes for attachment refs.

The ref is the *only* thing that lives in result JSON. Bytes always
sit in the CAS; the ref carries enough metadata for readback layers to
resolve, render, and audit without loading bytes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AttachmentKind = Literal["image", "pdf", "url", "url_screenshot"]


class AttachmentRef(BaseModel):
    """Per-attachment ref persisted in ``<result_id>.attachments/refs.json``.

    Promoted from ``TypedDict`` to :class:`pydantic.BaseModel` in v1.0.4
    (Pydantic Phase 2). The required fields (``id``, ``kind``, ``sha256``,
    ``content_type``, ``byte_size``) are sufficient to resolve the blob
    in CAS. Optional fields carry provenance (``source_uri``,
    ``fetched_at``, ``etag``, ``final_uri``), display metadata
    (``alt_text``, ``dims``, ``thumb_sha256``), and audit notes
    (``redaction_note``).

    ``extra='forbid'`` so drift in synthpanel-self-written refs.json
    surfaces as :class:`pydantic.ValidationError` at read time instead of
    a silent dict-key-missing failure downstream. Any new optional field
    must be added here at the same release the writer starts emitting it
    — old installs reading new refs.json otherwise hit ValidationError.
    See ``CONTRIBUTING.md`` for the contract.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(..., min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    kind: AttachmentKind
    sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    content_type: str = Field(..., min_length=1)
    byte_size: int = Field(..., ge=0)
    source_uri: str | None = None
    final_uri: str | None = None
    fetched_at: str | None = None
    etag: str | None = None
    alt_text: str | None = None
    dims: tuple[int, int] | None = None
    thumb_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    redaction_note: str | None = None
