"""Typed Pydantic models for the bundled extraction schemas.

These mirror the JSON Schema dicts in :mod:`synth_panel.structured.schemas`
and exist so callers can pass a Pydantic class to ``extract_schema=`` and
get typed parsing of LLM responses (with field-path-aware
:class:`pydantic.ValidationError` on schema violations).

The static JSON Schema dicts in ``schemas.py`` remain the wire format —
they are the frozen MCP contract. The Pydantic models here may be
strictly tighter than the wire schema (e.g. ``Likert.rating`` constrains
to 1..5) since they apply post-extraction, after the LLM has already
produced data conforming to the wire schema.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, Field, field_validator


class RankedItem(BaseModel):
    """One entry inside :class:`Ranking.ranked`."""

    name: str
    rank: int
    reasoning: str = ""


class Ranking(BaseModel):
    """Ordered ranking response. Mirrors ``RANKING_SCHEMA``."""

    ranked: list[RankedItem] = Field(..., min_length=1)


class Likert(BaseModel):
    """5-point Likert response. Mirrors ``LIKERT_SCHEMA``.

    The ``rating`` field is constrained to 1..5 so the post-extraction
    validation surfaces a usable field-path error when the model
    over-shoots the scale (e.g. emits ``rating: 7``).
    """

    rating: Annotated[int, Field(ge=1, le=5)]
    reasoning: str = ""


class YesNo(BaseModel):
    """Binary yes/no response. Mirrors ``YES_NO_SCHEMA``."""

    answer: bool
    reasoning: str = ""


class PickOne(BaseModel):
    """Single-choice pick. Mirrors ``PICK_ONE_SCHEMA``."""

    choice: str = Field(..., min_length=1)
    reasoning: str = ""


class AnnotatedChoice(BaseModel):
    """``PickOne`` + optional ``attachment_id`` (hq-cqt5 v1.0.0 surface).

    ``attachment_id`` is optional (soft validation), so a model that
    doesn't reference an attachment passes the same as under
    :class:`PickOne` — the schemas diverge only when the model *does*
    cite a specific attachment.

    The wire contract (``ANNOTATED_CHOICE_SCHEMA``) types
    ``attachment_id`` as a plain ``string`` and conveys "no attachment"
    via *absence* from the payload, not via JSON ``null``. We therefore
    type it as ``str`` with an empty-string default so
    ``model_json_schema()`` emits ``{"type": "string"}`` (matching the
    wire) instead of Pydantic's default ``anyOf`` for
    ``Optional[str]``. Empty string is the in-Python "no attachment"
    sentinel — callers should test with truthiness, not ``is None``.
    """

    choice: str = Field(..., min_length=1)
    reasoning: str = ""
    attachment_id: str = ""


# Leading bullet / enumeration markers stripped when coercing a
# newline-bulleted string into list items: "-", "*", "•", "–", "1.", "1)".
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")


class PartialSummary(BaseModel):
    """Per-question map-phase synthesis partial (v1.0.3 P2).

    Mirrors the content fields of :class:`synth_panel.synthesis.SynthesisResult`
    so the map-reduce strategy can validate each partial at the map
    boundary before feeding it to the reduce stage. Schema drift in a
    single map call surfaces as a :class:`pydantic.ValidationError`
    instead of silently propagating empty themes through to the final
    synthesis.

    List fields tolerate a string value by splitting it into items
    (newline-bulleted lists like ``"\\n- A\\n- B"`` are the observed
    provider drift). The single-pass synthesis path never re-validates
    these fields, so a string slips through it unchanged — the map
    boundary must not be stricter than single-pass for the same drift.
    """

    summary: str
    themes: list[str]
    agreements: list[str]
    disagreements: list[str]
    surprises: list[str]
    recommendation: str

    @field_validator("themes", "agreements", "disagreements", "surprises", mode="before")
    @classmethod
    def _coerce_string_to_list(cls, v: Any) -> Any:
        if not isinstance(v, str):
            return v
        items = [_BULLET_PREFIX_RE.sub("", line).strip() for line in v.splitlines()]
        return [item for item in items if item]


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "ranking": Ranking,
    "likert": Likert,
    "yes_no": YesNo,
    "pick_one": PickOne,
    "annotated_choice": AnnotatedChoice,
}
"""Maps the same names used by ``schemas._REGISTRY`` to typed models.

Lookup is case-sensitive and unknown names return ``None`` from
``MODEL_REGISTRY.get(name)`` — callers that need a hard failure should
go through :func:`synth_panel.structured.schemas.get_schema` (which
raises :class:`SchemaNotFoundError`) and then look the name up here.
"""


__all__ = [
    "MODEL_REGISTRY",
    "AnnotatedChoice",
    "Likert",
    "PartialSummary",
    "PickOne",
    "RankedItem",
    "Ranking",
    "YesNo",
]
