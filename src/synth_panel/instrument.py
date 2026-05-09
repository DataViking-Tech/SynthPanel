"""Instrument parser — supports v1 (flat), v2 (linear rounds), v3 (branching).

v1 format (flat questions):
    instrument:
      version: 1
      questions:
        - text: "..."

v2 format (multi-round, linear via depends_on):
    instrument:
      version: 2
      rounds:
        - name: discovery
          questions: [...]
        - name: deep_dive
          depends_on: discovery
          questions: [...]

v3 format (branching, route_when):
    instrument:
      version: 3
      rounds:
        - name: intro
          questions: [...]
          route_when:
            - if: "<condition>"
              goto: probe_pricing
            - else: wrap_up
        - name: probe_pricing
          questions: [...]
          route_when:
            - else: wrap_up
        - name: wrap_up
          questions: [...]

v1 and v2 stay valid as degenerate v3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from synth_panel.conditions import ConditionError, validate_condition_string
from synth_panel.structured.schemas import is_known_schema

END_SENTINEL = "__end__"

# Authored attachment IDs (the keys in Instrument.attachments) must match this
# pattern so they're safe to interpolate into log lines, file paths, and
# downstream URIs without escaping.
_ATTACHMENT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")

_ATTACHMENT_TYPES: frozenset[str] = frozenset({"image", "document", "url", "html"})

_IMAGE_MEDIA_TYPES: frozenset[str] = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_DOCUMENT_MEDIA_TYPES: frozenset[str] = frozenset({"application/pdf"})

_FETCH_MODES: frozenset[str] = frozenset({"auto", "html_text", "screenshot", "markdown"})


@dataclass
class Round:
    """A single round in an instrument."""

    name: str
    questions: list[dict[str, Any]]
    depends_on: str | None = None
    route_when: list[dict[str, Any]] | None = None


@dataclass
class Instrument:
    """Parsed instrument with round definitions.

    v1, v2, and v3 YAML formats normalize to this structure.
    v1 instruments become a single round named "default".

    The optional ``system_prompt_template`` field carries a Jinja2 or
    legacy ``{name}``-style template string embedded in the instrument
    YAML.  When present it overrides the default persona system prompt;
    Jinja2 syntax is validated at parse time via
    :func:`parse_instrument`.
    """

    version: int
    rounds: list[Round] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    system_prompt_template: str | None = None
    attachments: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def questions(self) -> list[dict[str, Any]]:
        """Flat question list — convenience for single-round instruments."""
        if len(self.rounds) == 1:
            return self.rounds[0].questions
        result: list[dict[str, Any]] = []
        for r in self.rounds:
            result.extend(r.questions)
        return result

    @property
    def is_multi_round(self) -> bool:
        return len(self.rounds) > 1


class InstrumentError(ValueError):
    """Raised when an instrument definition is invalid."""


_RESPONSE_SCHEMA_TYPES: frozenset[str] = frozenset({"text", "scale", "enum", "tagged_themes"})

_ATTACHMENT_FILTER_OPS: frozenset[str] = frozenset({"contains", "equals", "matches", "gte", "lte", "in"})


def _validate_attachment_filter(flt: Any, *, context: str, q_index: int, a_index: int) -> None:
    """Validate an attachment-level ``filter: list[predicate]`` clause.

    Each predicate must be a dict with ``field`` (str), ``op`` (one of
    the six routing/attachment ops), and ``value`` (any). The list may
    be empty — semantically equivalent to no filter — but must be a
    list when present. Numeric ops (``gte``/``lte``) and ``in`` are
    not coerced here; they fail at evaluation time if the persona
    trait can't satisfy them.
    """
    loc = f"{context} question[{q_index}] attachments[{a_index}] filter"
    if not isinstance(flt, list):
        raise InstrumentError(f"{loc}: must be a list of predicates, got {type(flt).__name__}")
    for k, p in enumerate(flt):
        if not isinstance(p, dict):
            raise InstrumentError(f"{loc}[{k}]: must be a mapping, got {type(p).__name__}")
        for key in ("field", "op", "value"):
            if key not in p:
                raise InstrumentError(f"{loc}[{k}]: missing required key '{key}'")
        if not isinstance(p["field"], str) or not p["field"]:
            raise InstrumentError(f"{loc}[{k}]: 'field' must be a non-empty string")
        if not isinstance(p["op"], str) or p["op"] not in _ATTACHMENT_FILTER_OPS:
            raise InstrumentError(f"{loc}[{k}]: 'op' must be one of {sorted(_ATTACHMENT_FILTER_OPS)}, got {p['op']!r}")
        if p["op"] == "in" and not isinstance(p["value"], list):
            raise InstrumentError(f"{loc}[{k}]: 'in' op requires a list 'value'")


def _validate_response_schema(rs: Any, *, context: str, q_index: int) -> None:
    """Validate a question-level ``response_schema`` entry.

    Recognized shapes (sp-2hpi):

    - ``{"type": "text", "max_tokens": N?}`` — free text (default)
    - ``{"type": "scale", "min": M, "max": N}`` — numeric scale with integer bounds (min < max)
    - ``{"type": "enum", "options": [...]}`` — categorical choice, non-empty list of strings
    - ``{"type": "tagged_themes", "taxonomy": [...], "multi": bool?}`` — structured tags from a fixed taxonomy

    Unknown shapes are rejected so instruments fail fast. A plain dict
    without a recognized ``type`` (legacy use of ``response_schema`` as a
    free JSON Schema) is accepted unchanged for backward compatibility.
    """
    if not isinstance(rs, dict):
        raise InstrumentError(
            f"{context} question[{q_index}]: response_schema must be a mapping, got {type(rs).__name__}"
        )
    t = rs.get("type")
    if not isinstance(t, str) or t not in _RESPONSE_SCHEMA_TYPES:
        # Legacy / inline JSON Schema — accept without semantic checks.
        return

    loc = f"{context} question[{q_index}] response_schema"
    if t == "scale":
        lo = rs.get("min")
        hi = rs.get("max")
        if not isinstance(lo, int) or isinstance(lo, bool):
            raise InstrumentError(f"{loc}: 'min' must be an integer, got {type(lo).__name__}")
        if not isinstance(hi, int) or isinstance(hi, bool):
            raise InstrumentError(f"{loc}: 'max' must be an integer, got {type(hi).__name__}")
        if lo >= hi:
            raise InstrumentError(f"{loc}: 'min' ({lo}) must be strictly less than 'max' ({hi})")
    elif t == "enum":
        opts = rs.get("options")
        if not isinstance(opts, list) or not opts:
            raise InstrumentError(f"{loc}: 'options' must be a non-empty list of strings")
        if not all(isinstance(o, str) and o for o in opts):
            raise InstrumentError(f"{loc}: 'options' entries must be non-empty strings")
        if len(set(opts)) != len(opts):
            raise InstrumentError(f"{loc}: 'options' must be unique")
    elif t == "tagged_themes":
        taxonomy = rs.get("taxonomy")
        if not isinstance(taxonomy, list) or not taxonomy:
            raise InstrumentError(f"{loc}: 'taxonomy' must be a non-empty list of strings")
        if not all(isinstance(tag, str) and tag for tag in taxonomy):
            raise InstrumentError(f"{loc}: 'taxonomy' entries must be non-empty strings")
        if len(set(taxonomy)) != len(taxonomy):
            raise InstrumentError(f"{loc}: 'taxonomy' must be unique")
        multi = rs.get("multi", False)
        if not isinstance(multi, bool):
            raise InstrumentError(f"{loc}: 'multi' must be a boolean, got {type(multi).__name__}")
    elif t == "text":
        mt = rs.get("max_tokens")
        if mt is not None and (not isinstance(mt, int) or isinstance(mt, bool) or mt <= 0):
            raise InstrumentError(f"{loc}: 'max_tokens' must be a positive integer when provided")


def _validate_attachment_source(source: Any, *, attachment_type: str, context: str) -> None:
    """Shape-check an attachment ``source`` mapping (tagged-union variant).

    Recognized variants:
      * ``{"type": "base64", "data": "..."}`` — inline base64 payload.
      * ``{"type": "url", "url": "..."}`` — externally-hosted resource.
      * ``{"type": "file", "file_id": "..."}`` — provider Files API ref.
    """
    if not isinstance(source, dict):
        raise InstrumentError(f"{context}: '{attachment_type}' source must be a mapping, got {type(source).__name__}")
    stype = source.get("type")
    if stype == "base64":
        data = source.get("data")
        if not isinstance(data, str) or not data:
            raise InstrumentError(f"{context}: source.data must be a non-empty string for base64 source")
    elif stype == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url:
            raise InstrumentError(f"{context}: source.url must be a non-empty string for url source")
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise InstrumentError(f"{context}: source.url '{url}' is not a syntactically valid URL")
    elif stype == "file":
        file_id = source.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            raise InstrumentError(f"{context}: source.file_id must be a non-empty string for file source")
    else:
        raise InstrumentError(f"{context}: source.type must be one of 'base64', 'url', 'file' (got {stype!r})")


def _validate_attachment(att_id: str, raw: Any, *, context: str) -> None:
    """Type-check a single attachment definition.

    The bank value is a dict with at least a ``type`` discriminator. Per-type
    shape checks land payload errors at parse time, before the run starts.
    """
    if not _ATTACHMENT_ID_RE.match(att_id):
        raise InstrumentError(f"{context}: attachment id {att_id!r} must match ^[a-z][a-z0-9_-]{{0,63}}$")
    if not isinstance(raw, dict):
        raise InstrumentError(f"{context}: attachment '{att_id}' must be a mapping, got {type(raw).__name__}")
    atype = raw.get("type")
    if atype not in _ATTACHMENT_TYPES:
        raise InstrumentError(
            f"{context}: attachment '{att_id}' has unknown type {atype!r}; expected one of {sorted(_ATTACHMENT_TYPES)}"
        )

    loc = f"{context} attachment '{att_id}'"

    if atype == "image":
        media_type = raw.get("media_type")
        if media_type is not None and media_type not in _IMAGE_MEDIA_TYPES:
            raise InstrumentError(
                f"{loc}: media_type {media_type!r} not allowed; expected one of {sorted(_IMAGE_MEDIA_TYPES)}"
            )
        _validate_attachment_source(raw.get("source"), attachment_type="image", context=loc)
    elif atype == "document":
        media_type = raw.get("media_type", "application/pdf")
        if media_type not in _DOCUMENT_MEDIA_TYPES:
            raise InstrumentError(
                f"{loc}: media_type {media_type!r} not allowed; expected one of {sorted(_DOCUMENT_MEDIA_TYPES)}"
            )
        _validate_attachment_source(raw.get("source"), attachment_type="document", context=loc)
    elif atype == "url":
        url = raw.get("url")
        if not isinstance(url, str) or not url:
            raise InstrumentError(f"{loc}: 'url' must be a non-empty string")
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise InstrumentError(f"{loc}: url {url!r} is not a syntactically valid URL")
        fetch_mode = raw.get("fetch_mode", "auto")
        if fetch_mode not in _FETCH_MODES:
            raise InstrumentError(
                f"{loc}: fetch_mode {fetch_mode!r} not recognized; expected one of {sorted(_FETCH_MODES)}"
            )
    elif atype == "html":
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise InstrumentError(f"{loc}: 'text' must be a non-empty string")

    cache_control = raw.get("cache_control")
    if cache_control is not None and cache_control != "ephemeral":
        raise InstrumentError(f"{loc}: cache_control must be 'ephemeral' or absent (got {cache_control!r})")


def _validate_attachments(
    bank: Any,
    questions_by_context: list[tuple[list[dict[str, Any]], str]],
) -> dict[str, dict[str, Any]]:
    """Validate the top-level attachment bank and per-question references.

    Args:
        bank: Raw value for ``Instrument.attachments``. Either ``None`` or
            a dict-keyed-by-id mapping.
        questions_by_context: Each tuple is ``(question_list, context_label)``
            so reachability errors mention the round.

    Returns:
        The validated bank as a dict (empty if ``bank`` is None).

    Validation:
        * Bank ID format: ``^[a-z][a-z0-9_-]{0,63}$``.
        * Per-attachment type/shape checks (image/document/url/html).
        * Per-question ``attachments`` list: every string resolves to a bank id.
        * Cache invariant: at most one block per question may carry
          ``cache_control: ephemeral`` (across the resolved bank refs and
          inline blocks), and it must be the last entry of the shared
          (bank-referenced) prefix. Protects the prompt-caching contract
          (hq-bqrw §8) at parse time.
    """
    if bank is None:
        bank = {}
    if not isinstance(bank, dict):
        raise InstrumentError(f"'attachments' must be a mapping keyed by id, got {type(bank).__name__}")
    for att_id, raw in bank.items():
        if not isinstance(att_id, str):
            raise InstrumentError(f"attachment ids must be strings, got {type(att_id).__name__}")
        _validate_attachment(att_id, raw, context="attachments")

    for questions, context in questions_by_context:
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                continue
            refs = q.get("attachments")
            if refs is not None:
                if not isinstance(refs, list):
                    raise InstrumentError(
                        f"{context} question[{i}]: attachments must be a list of ids, got {type(refs).__name__}"
                    )
                for ref in refs:
                    if not isinstance(ref, str):
                        raise InstrumentError(
                            f"{context} question[{i}]: attachment ref must be a string, got {type(ref).__name__}"
                        )
                    if ref not in bank:
                        raise InstrumentError(
                            f"{context} question[{i}]: attachment ref '{ref}' "
                            f"does not resolve to a top-level attachment"
                        )

            inline = q.get("inline_attachments")
            if inline is not None:
                if not isinstance(inline, list):
                    raise InstrumentError(
                        f"{context} question[{i}]: inline_attachments must be a list, got {type(inline).__name__}"
                    )
                for j, block in enumerate(inline):
                    if not isinstance(block, dict):
                        raise InstrumentError(f"{context} question[{i}] inline_attachments[{j}] must be a mapping")
                    btype = block.get("type")
                    if btype not in _ATTACHMENT_TYPES and btype != "text":
                        raise InstrumentError(
                            f"{context} question[{i}] inline_attachments[{j}] has unknown "
                            f"type {btype!r}; expected one of {sorted(_ATTACHMENT_TYPES | {'text'})}"
                        )
                    # text blocks are simple; reuse attachment shape checks for the rest
                    if btype != "text":
                        _validate_attachment(
                            f"inline_{i}_{j}",
                            {**block, "type": btype},  # already-shape; ID purely for error context
                            context=f"{context} question[{i}] inline_attachments[{j}]",
                        )

            # Cache-control invariant: at most one ephemeral marker per question,
            # and it must sit at the end of the shared (bank-referenced) prefix.
            ephemeral_idx: list[int] = []
            position = 0
            shared_prefix_end = len(refs) if isinstance(refs, list) else 0
            if isinstance(refs, list):
                for ref in refs:
                    if bank.get(ref, {}).get("cache_control") == "ephemeral":
                        ephemeral_idx.append(position)
                    position += 1
            if isinstance(inline, list):
                for block in inline:
                    if isinstance(block, dict) and block.get("cache_control") == "ephemeral":
                        ephemeral_idx.append(position)
                    position += 1
            if len(ephemeral_idx) > 1:
                raise InstrumentError(
                    f"{context} question[{i}]: at most one attachment may carry "
                    f"cache_control=ephemeral (got {len(ephemeral_idx)})"
                )
            if ephemeral_idx and ephemeral_idx[0] != shared_prefix_end - 1:
                raise InstrumentError(
                    f"{context} question[{i}]: cache_control=ephemeral must mark "
                    f"the LAST shared (bank-referenced) attachment, not an inline "
                    f"or earlier-shared block"
                )

    return bank


def _validate_questions(questions: list[dict[str, Any]], context: str) -> None:
    """Validate ``extraction_schema`` and ``response_schema`` on questions.

    For ``extraction_schema``: string values are checked against the bundled
    schema registry; dict values are accepted as inline JSON Schemas without
    further validation. Other types raise :class:`InstrumentError`.

    For ``response_schema``: dicts with a recognized ``type`` (text, scale,
    enum, tagged_themes) are shape-checked; legacy inline JSON Schemas
    (dicts without a recognized type) pass through untouched.

    Args:
        questions: List of question dicts to validate.
        context: Human-readable location for error messages
            (e.g. ``"v1 instrument"`` or ``"round 'discovery'"``).
    """
    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            continue
        es = q.get("extraction_schema")
        if es is not None:
            if isinstance(es, str):
                if not is_known_schema(es):
                    from synth_panel.structured.schemas import SchemaNotFoundError

                    raise InstrumentError(str(SchemaNotFoundError(es)))
            elif not isinstance(es, dict):
                raise InstrumentError(
                    f"{context} question[{i}]: extraction_schema must be a string (schema name) or mapping, "
                    f"got {type(es).__name__}"
                )
        rs = q.get("response_schema")
        if rs is not None:
            _validate_response_schema(rs, context=context, q_index=i)

        # hq-iczd: validate optional per-attachment stratification filters.
        # Attachment shape itself is hq-pojo's contract; here we only check
        # the ``filter`` predicate list so authoring typos fail at parse
        # time rather than silently mismatching at runtime.
        attachments = q.get("attachments")
        if attachments is not None:
            if not isinstance(attachments, list):
                raise InstrumentError(
                    f"{context} question[{i}]: attachments must be a list, got {type(attachments).__name__}"
                )
            for ai, att in enumerate(attachments):
                if not isinstance(att, dict):
                    continue
                flt = att.get("filter")
                if flt is not None:
                    _validate_attachment_filter(flt, context=context, q_index=i, a_index=ai)

        follow_ups = q.get("follow_ups")
        if isinstance(follow_ups, list):
            for j, fu in enumerate(follow_ups):
                # Plain-string follow-ups default to "always" at eval time and
                # need no validation. Only dict-form follow-ups can carry an
                # explicit condition that might be a typo (sp-t5ok).
                if not isinstance(fu, dict):
                    continue
                cond = fu.get("condition")
                if cond is None:
                    continue
                try:
                    validate_condition_string(
                        cond,
                        context=f"{context} question[{i}] follow_ups[{j}]",
                    )
                except ConditionError as e:
                    raise InstrumentError(str(e)) from e


def parse_instrument(data: dict[str, Any]) -> Instrument:
    """Parse a raw instrument dict into a validated Instrument.

    Accepts v1 (``questions``) or v2/v3 (``rounds``). Runs the full
    DAG validation ladder before returning. Raises
    :class:`InstrumentError` on validation failure.

    The optional ``system_prompt_template`` top-level key embeds a persona
    system prompt template in the instrument.  Jinja2 syntax is validated
    (compiled) here so errors surface at load time rather than per-turn.
    """
    version = data.get("version", 1)

    spt = data.get("system_prompt_template")
    if spt is not None:
        if not isinstance(spt, str):
            raise InstrumentError("'system_prompt_template' must be a string")
        _validate_system_prompt_template(spt)

    if "rounds" in data:
        instrument = _parse_rounds(data, version)
    elif "questions" in data:
        instrument = _parse_v1(data, version)
    else:
        raise InstrumentError("Instrument must have either 'questions' (v1) or 'rounds' (v2) key")

    instrument.system_prompt_template = spt

    # Attachments are top-level (not per-round) so the bank shows up once on
    # the parsed Instrument regardless of v1/v2/v3 shape.
    raw_attachments = data.get("attachments")
    instrument.attachments = _validate_attachments(
        raw_attachments,
        [(r.questions, f"round '{r.name}'") for r in instrument.rounds],
    )
    return instrument


def _validate_system_prompt_template(template_str: str) -> None:
    """Compile a Jinja2 system_prompt_template to validate its syntax.

    Only runs for templates containing Jinja2 markers; legacy ``{name}``
    templates are accepted without further checks.  Raises
    :class:`InstrumentError` on Jinja2 syntax errors.
    """
    _JINJA2_MARKERS = ("{{", "{%", "{#")
    if not any(m in template_str for m in _JINJA2_MARKERS):
        return
    try:
        from synth_panel.prompts import compile_jinja2_template

        compile_jinja2_template(template_str)
    except Exception as exc:
        raise InstrumentError(f"system_prompt_template Jinja2 syntax error: {exc}") from exc


def _parse_v1(data: dict[str, Any], version: int) -> Instrument:
    """Parse v1 flat-questions format into a single-round instrument."""
    questions = data["questions"]
    if not isinstance(questions, list) or not questions:
        raise InstrumentError("'questions' must be a non-empty list")
    _validate_questions(questions, "v1 instrument")
    return Instrument(
        version=version,
        rounds=[Round(name="default", questions=questions)],
    )


def _parse_rounds(data: dict[str, Any], version: int) -> Instrument:
    """Two-pass parse for v2/v3 multi-round instruments."""
    raw_rounds = data["rounds"]
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise InstrumentError("'rounds' must be a non-empty list")

    # ---- Rung 1: Structural pass — collect names, build Round objects ----
    rounds: list[Round] = []
    name_set: set[str] = set()

    for i, raw in enumerate(raw_rounds):
        if not isinstance(raw, dict):
            raise InstrumentError(f"Round {i} must be a mapping, got {type(raw).__name__}")

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise InstrumentError(f"Round {i} must have a 'name' string")

        if name in name_set:
            raise InstrumentError(f"Duplicate round name: '{name}'")

        questions = raw.get("questions")
        if not isinstance(questions, list) or not questions:
            raise InstrumentError(f"Round '{name}' must have a non-empty 'questions' list")
        _validate_questions(questions, f"round '{name}'")

        depends_on = raw.get("depends_on")
        if depends_on is not None and not isinstance(depends_on, str):
            raise InstrumentError(f"Round '{name}': 'depends_on' must be a string, got {type(depends_on).__name__}")

        route_when = raw.get("route_when")
        if route_when is not None:
            if not isinstance(route_when, list) or not route_when:
                raise InstrumentError(f"Round '{name}': 'route_when' must be a non-empty list")
            for j, entry in enumerate(route_when):
                if not isinstance(entry, dict):
                    raise InstrumentError(f"Round '{name}': route_when[{j}] must be a mapping")

        name_set.add(name)
        rounds.append(
            Round(
                name=name,
                questions=questions,
                depends_on=depends_on,
                route_when=route_when,
            )
        )

    # ---- Rung 2: Goto resolution (forward refs allowed) ----
    for r in rounds:
        if r.depends_on is not None and r.depends_on not in name_set:
            raise InstrumentError(f"Round '{r.name}': depends_on '{r.depends_on}' does not exist")
        if r.route_when:
            _validate_route_when_targets(r, name_set)

    # ---- Rung 4: Else completeness (must be checked before reachability) ----
    for r in rounds:
        if r.route_when and "else" not in r.route_when[-1]:
            raise InstrumentError(
                f"round '{r.name}' has no else clause; add 'else: <round_name>' or 'else: {END_SENTINEL}'"
            )

    # ---- Rung 3: Acyclicity (topo sort) ----
    edges = _build_edges(rounds)
    cycle = _find_cycle(rounds, edges)
    if cycle is not None:
        raise InstrumentError(f"Cycle detected in instrument DAG: {' -> '.join(cycle)}")

    # ---- Rung 5: Reachability (warning, not error) ----
    warnings = _reachability_warnings(rounds, edges)

    return Instrument(version=version, rounds=rounds, warnings=warnings)


def _validate_route_when_targets(r: Round, name_set: set[str]) -> None:
    assert r.route_when is not None
    for j, entry in enumerate(r.route_when):
        if "else" in entry:
            target = entry["else"]
            if not isinstance(target, str):
                raise InstrumentError(f"round '{r.name}' route_when[{j}] else must be a string")
            if target != END_SENTINEL and target not in name_set:
                raise InstrumentError(f"round '{r.name}' goto '{target}' does not exist")
        elif "goto" in entry:
            target = entry["goto"]
            if not isinstance(target, str):
                raise InstrumentError(f"round '{r.name}' route_when[{j}] goto must be a string")
            if target != END_SENTINEL and target not in name_set:
                raise InstrumentError(f"round '{r.name}' goto '{target}' does not exist")
        else:
            raise InstrumentError(f"round '{r.name}' route_when[{j}] must have 'goto' or 'else'")


def _build_edges(rounds: list[Round]) -> dict[str, list[str]]:
    """Build directed flow edges (parent -> child) from depends_on and route_when."""
    edges: dict[str, list[str]] = {r.name: [] for r in rounds}
    for r in rounds:
        if r.depends_on:
            edges[r.depends_on].append(r.name)
        if r.route_when:
            for entry in r.route_when:
                target = entry.get("goto") or entry.get("else")
                if target and target != END_SENTINEL:
                    edges[r.name].append(target)
    return edges


def _find_cycle(rounds: list[Round], edges: dict[str, list[str]]) -> list[str] | None:
    """Return a cycle path as a list of round names, or None if acyclic."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {r.name: WHITE for r in rounds}
    parent: dict[str, str | None] = {r.name: None for r in rounds}
    cycle_path: list[str] | None = None

    def dfs(node: str) -> bool:
        nonlocal cycle_path
        color[node] = GRAY
        for nxt in edges.get(node, []):
            if color[nxt] == GRAY:
                # Reconstruct cycle: nxt ... node -> nxt
                path = [nxt]
                cur: str | None = node
                while cur is not None and cur != nxt:
                    path.append(cur)
                    cur = parent[cur]
                path.append(nxt)
                path.reverse()
                cycle_path = path
                return True
            if color[nxt] == WHITE:
                parent[nxt] = node
                if dfs(nxt):
                    return True
        color[node] = BLACK
        return False

    for r in rounds:
        if color[r.name] == WHITE and dfs(r.name):
            return cycle_path
    return None


def _reachability_warnings(rounds: list[Round], edges: dict[str, list[str]]) -> list[str]:
    """Return warnings for unreachable rounds.

    Entry round is rounds[0]. Traversal follows route_when/depends_on edges,
    plus an implicit linear edge from round_i to round_{i+1} when round_i has
    no route_when (preserving v2 linear semantics).
    """
    if not rounds:
        return []

    # Build traversal edges: explicit edges + implicit linear next.
    traverse: dict[str, list[str]] = {r.name: list(edges.get(r.name, [])) for r in rounds}
    for i, r in enumerate(rounds):
        if r.route_when is None and i + 1 < len(rounds):
            traverse[r.name].append(rounds[i + 1].name)

    seen: set[str] = set()
    stack = [rounds[0].name]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in traverse.get(cur, []):
            if nxt not in seen:
                stack.append(nxt)

    return [f"unreachable round: '{r.name}'" for r in rounds if r.name not in seen]
