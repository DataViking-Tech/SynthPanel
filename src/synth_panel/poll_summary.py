"""Automatic poll summary reporting for structured panel results (sy-4yd).

Walks a saved panel result (PollResult / PanelResult / loaded JSON) and
emits a deterministic, machine-readable summary of structured responses:
first-choice vote counts, second-choice vote counts (when the schema
captures one), weighted scores, metric averages, segment splits by
persona attribute, top objections, and the synthesis-derived recommended
next test.

Design constraints (GH #477):

* **Deterministic.** No LLM calls — the summary is a pure function of
  the result JSON. Two runs over the same input produce byte-identical
  output. This is what lets agents quote the summary back to a user
  without a follow-up tool call.
* **Structured-first, free-text-tolerant.** When a panelist response
  has a structured ``extraction`` (or a numeric / enum ``response``),
  it counts toward the typed aggregates. Free-text responses contribute
  to objections / qualitative notes but never warp the counts.
* **Shape-agnostic.** Works on PollResult.responses (one question),
  PanelResult.rounds[].results (many questions), and the raw JSON
  envelope persisted to disk — the same shapes that
  :func:`synth_panel.analysis.inspect.build_inspect_report` consumes.

The summary lives at :class:`PollSummary` and is computed via
:func:`build_poll_summary`. CLI + SDK surfaces are wired up in
``cli/commands.py`` and ``sdk.py`` respectively.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

# Common field names used by the structured-output engine to surface a
# panelist's primary choice. Ordered by how strongly each name signals
# "this is THE choice" — exact matches beat substring matches in the
# extraction shapes we've seen in the wild.
_FIRST_CHOICE_KEYS: tuple[str, ...] = (
    "first_choice",
    "primary_choice",
    "choice",
    "answer",
    "selection",
    "winner",
    "pick",
    "value",
)

_SECOND_CHOICE_KEYS: tuple[str, ...] = (
    "second_choice",
    "runner_up",
    "alternate",
    "fallback",
)

# Numeric-rating fields. Treated as both a per-response metric (for the
# overall average) and as a "weight" toward the panelist's first_choice
# (so an option with mostly 5/5 backers beats one with mostly 3/5 backers
# even at equal vote counts).
_SCORE_KEYS: tuple[str, ...] = (
    "score",
    "rating",
    "confidence",
    "preference_strength",
    "likelihood",
)

# Free-text objection / concern fields. Aggregated into ``top_objections``.
_OBJECTION_KEYS: tuple[str, ...] = (
    "objections",
    "concerns",
    "trust_concerns",
    "blockers",
    "pushback",
    "worries",
)

# Persona attributes that drive segment splits by default. Authors can
# extend this via ``segment_by`` on :func:`build_poll_summary`. We keep
# the default list narrow so the summary stays signal-dense — splitting
# on every persona attribute produces a noisy long-tail table that
# obscures the headline result.
_DEFAULT_SEGMENT_ATTRS: tuple[str, ...] = (
    "occupation",
    "segment",
    "age_band",
    "tier",
)

# Cap on top objections + the cap on per-segment value rows. Empirically
# anything past ~5 starts dragging the headline out of view in CLI text
# mode and adds little to the agent's decision context.
_TOP_OBJECTIONS_CAP = 5
_PER_SEGMENT_VALUE_CAP = 8


@dataclass
class QuestionSummary:
    """Per-question rollup of structured panelist responses.

    The discriminator is :attr:`kind`:

    * ``"enum"`` — categorical choice. ``first_choice_counts`` is the
      headline; ``winner`` is the option with the highest count
      (ties broken by ordering in ``first_choice_counts`` — Counter's
      first-insertion order in Python 3.7+).
    * ``"scale"`` — numeric. ``metric_average`` is the headline;
      ``metric_distribution`` is the histogram.
    * ``"text"`` — free-form. Counts / averages are ``None`` — the
      response is summarized only via :class:`PollSummary.top_objections`
      and the panel's broader synthesis.

    Attributes:
        question: The question text (truncated to 280 chars for safety).
        question_index: Zero-based index within the run.
        kind: One of ``"enum"``, ``"scale"``, ``"text"``.
        first_choice_counts: For enum questions, ``option -> vote count``.
            Sorted by descending count, then by first-appearance order.
        second_choice_counts: Same shape as ``first_choice_counts``, but
            populated from the panelist's ``second_choice`` extraction
            field when present. ``None`` when no panelist supplied one.
        weighted_scores: For enum questions where each panelist also
            supplied a numeric rating, ``option -> mean rating``.
            ``None`` when no rating field is present.
        winner: Option name with the highest first-choice count, or
            ``None`` when no panelist produced a parseable choice.
        metric_average: For scale questions, the mean of all parseable
            numeric responses.
        metric_distribution: For scale questions, ``stringified_value -> count``.
            Stringified so the field is JSON-serializable without
            collapsing 1 and "1" into the same bucket.
        n: Total panelists who answered this question.
        n_unparseable: Subset of ``n`` whose response could not be
            mapped to an option (enum) or a number (scale). Caller can
            divide to get a parse-rate.
    """

    question: str
    question_index: int
    kind: str  # "enum" | "scale" | "text"
    n: int
    n_unparseable: int = 0
    first_choice_counts: dict[str, int] | None = None
    second_choice_counts: dict[str, int] | None = None
    weighted_scores: dict[str, float] | None = None
    winner: str | None = None
    metric_average: float | None = None
    metric_distribution: dict[str, int] | None = None


@dataclass
class PollSummary:
    """Top-level summary returned by :func:`build_poll_summary`.

    Attributes:
        questions: Per-question summaries in the order the panel ran
            them.
        segment_splits: Three-level dict — ``question_index -> attribute
            -> attribute_value -> {option: count}``. Lets a caller answer
            "how did the 25-34 cohort split on Q3?" without rewalking the
            raw transcript.
        top_objections: Free-text concerns aggregated across every
            ``objections`` / ``concerns`` field, normalised and deduped,
            capped at the most-frequent N. Empty when no objections were
            captured.
        recommended_next_test: Echo of ``synthesis.recommendation`` from
            the panel result when present. The synthesis layer already
            produces this; we surface it here so a caller can read one
            object instead of two.
        persona_count: Total panelists in the result. Useful for
            normalising the counts (% of panel = count / persona_count).
        warnings: Non-fatal parser observations — e.g. "Q2 had 3
            unparseable enum responses". Empty on a clean run.
    """

    persona_count: int
    questions: list[QuestionSummary] = field(default_factory=list)
    segment_splits: dict[str, dict[str, dict[str, dict[str, int]]]] = field(default_factory=dict)
    top_objections: list[str] = field(default_factory=list)
    recommended_next_test: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe rendering. Used by SDK + MCP surfaces."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_poll_summary(
    result: dict[str, Any],
    *,
    segment_by: list[str] | None = None,
    personas: list[dict[str, Any]] | None = None,
    choice_field: str | None = None,
    confidence_field: str | None = None,
) -> PollSummary:
    """Compute a :class:`PollSummary` from a saved panel result.

    Args:
        result: A loaded result JSON dict (from disk or fresh from
            :func:`run_panel` / :func:`quick_poll`). Both flat
            (``results: [...]``) and rounds-shaped (``rounds: [{results:
            [...]}, ...]``) envelopes are accepted.
        segment_by: Persona attributes to split on. Defaults to the
            common set (occupation, segment, age_band, tier). Pass
            ``[]`` to disable segment splits entirely.
        personas: Optional explicit persona list. When provided, the
            persona name in each panelist response is looked up here for
            segment attributes. When omitted, attributes embedded in the
            panelist record itself are used.
        choice_field: Explicit name of the structured field that holds
            each panelist's primary choice (e.g. ``"choice"``,
            ``"vote"``). When provided, overrides the auto-detection
            done from :data:`_FIRST_CHOICE_KEYS` and the response
            schema. Useful when an ``--schema`` panel has multiple
            enum properties and the wrong one wins the deterministic
            tiebreak. sy-bn7.
        confidence_field: Explicit name of the numeric rating field
            (e.g. ``"confidence"``, ``"score"``). Overrides
            :data:`_SCORE_KEYS` auto-detection. sy-bn7.

    Returns:
        A :class:`PollSummary`. Always returns — never raises on a
        well-formed result. Malformed pieces become entries in
        ``summary.warnings``.
    """
    panelists = _extract_panelists(result)
    segments = list(_DEFAULT_SEGMENT_ATTRS) if segment_by is None else list(segment_by)

    # The persona lookup lets us join optional persona attributes that
    # the panelist response doesn't carry inline (the orchestrator only
    # records ``name`` + ``responses`` per panelist).
    persona_index: dict[str, dict[str, Any]] = {}
    if personas:
        for p in personas:
            name = p.get("name")
            if isinstance(name, str):
                persona_index[name] = p

    questions_meta = _collect_question_meta(result, panelists)
    summary = PollSummary(persona_count=len(panelists))

    objection_counter: Counter[str] = Counter()
    seg_table: dict[str, dict[str, dict[str, dict[str, int]]]] = {}

    for qi, qmeta in enumerate(questions_meta):
        per_panelist = _per_panelist_view(panelists, qi)
        # sy-bn7: resolve the choice/confidence field names per question
        # so a schema-enforced ``--schema`` run with a structured
        # response dict is recognized as enum instead of falling through
        # to ``text``. The resolver also picks up ambiguous schemas and
        # records a warning so the caller knows which field won the tie.
        resolved_choice, resolved_confidence = _resolve_field_names(
            qi,
            qmeta,
            per_panelist,
            explicit_choice=choice_field,
            explicit_confidence=confidence_field,
            warnings=summary.warnings,
        )
        kind = _detect_kind(qmeta, per_panelist, choice_field=resolved_choice)
        qsum = _summarise_question(
            qi,
            qmeta,
            per_panelist,
            kind,
            summary.warnings,
            choice_field=resolved_choice,
            confidence_field=resolved_confidence,
        )
        summary.questions.append(qsum)

        # Segments — only meaningful for enum questions where the choice
        # is comparable across personas. Scale questions get their own
        # average; text questions feed the objections bucket.
        if kind == "enum" and segments:
            seg_table[str(qi)] = _segment_split(
                per_panelist,
                persona_index,
                segments,
                choice_field=resolved_choice,
            )

        # Objections / concerns get harvested regardless of kind so a
        # free-text follow-up can still surface trust concerns.
        for entry in per_panelist:
            for obj in _harvest_objections(entry):
                objection_counter[obj] += 1

    summary.segment_splits = seg_table
    summary.top_objections = [obj for obj, _ in objection_counter.most_common(_TOP_OBJECTIONS_CAP)]
    summary.recommended_next_test = _extract_recommendation(result)
    return summary


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _extract_panelists(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the various result envelopes into one list of panelist dicts.

    Handles three shapes, mirroring :func:`analysis.inspect._iter_panelist_entries`:

    * Flat: ``result["results"]`` is the panelist list.
    * Rounds: ``result["rounds"][i]["results"]`` — we collapse all rounds
      into one stream because the summary is cross-round by design.
    * Bare PollResult.responses: top-level list of panelist dicts.
    """
    if isinstance(result.get("results"), list):
        return [p for p in result["results"] if isinstance(p, dict)]
    if isinstance(result.get("rounds"), list):
        out: list[dict[str, Any]] = []
        for rnd in result["rounds"]:
            if isinstance(rnd, dict) and isinstance(rnd.get("results"), list):
                out.extend(p for p in rnd["results"] if isinstance(p, dict))
        return out
    if isinstance(result.get("responses"), list):
        return [p for p in result["responses"] if isinstance(p, dict)]
    return []


def _collect_question_meta(
    result: dict[str, Any],
    panelists: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recover one ``{text, response_schema}`` entry per asked question.

    Prefers the run's ``questions`` field when present (carries
    ``response_schema``). Otherwise falls back to the first panelist's
    response list — the question text lives there too, just without the
    schema metadata. ``response_schema`` may be missing in either case;
    :func:`_detect_kind` is robust to that.
    """
    if isinstance(result.get("questions"), list) and result["questions"]:
        return [q for q in result["questions"] if isinstance(q, dict)]

    if panelists:
        first_responses = panelists[0].get("responses") or []
        if isinstance(first_responses, list):
            return [
                {"text": r.get("question", "")}
                for r in first_responses
                if isinstance(r, dict) and not r.get("follow_up")
            ]
    return []


def _per_panelist_view(
    panelists: list[dict[str, Any]],
    question_index: int,
) -> list[dict[str, Any]]:
    """For a question index, return one ``{persona, response_dict}`` per panelist.

    Skips follow-up responses — those don't carry the schema-validated
    primary answer. Missing responses (panelist failed before reaching
    this question) become ``{"persona": name, "response_dict": None}``
    so segment splits still account for the panelist.
    """
    out: list[dict[str, Any]] = []
    for p in panelists:
        responses = p.get("responses") or []
        primaries = [r for r in responses if isinstance(r, dict) and not r.get("follow_up")]
        rd: dict[str, Any] | None = None
        if 0 <= question_index < len(primaries):
            rd = primaries[question_index]
        out.append(
            {
                "persona": p.get("persona") or p.get("name"),
                "persona_attrs": {k: v for k, v in p.items() if k not in ("responses", "usage", "cost")},
                "response_dict": rd,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-question summarisation
# ---------------------------------------------------------------------------


def _detect_kind(
    qmeta: dict[str, Any],
    per_panelist: list[dict[str, Any]],
    *,
    choice_field: str | None = None,
) -> str:
    """Classify a question as enum / scale / text.

    Schema-driven when ``qmeta.response_schema`` is present. Otherwise
    inferred from the response shape: numeric → scale, enum-like dict
    extractions or short repeating strings → enum, anything else → text.

    sy-bn7: a JSON Schema with ``type: "object"`` and a ``properties.<key>.enum``
    field counts as enum, and any structured-response dict carrying a
    recognized choice key in the ``response`` field (the shape ``--schema``
    runs persist) also counts — previously these landed as ``text``.
    """
    schema = qmeta.get("response_schema")
    if isinstance(schema, dict):
        st = schema.get("type")
        if st == "enum":
            return "enum"
        if st in {"scale", "number", "integer"}:
            return "scale"
        if st == "text":
            # text + extract_schema with a primary choice key still counts
            # as enum for summary purposes. The schema text/extraction
            # split is an implementation detail of the orchestrator —
            # for the consumer, "what did the panel pick" is the question.
            if _has_choice_extraction(per_panelist) or _has_choice_in_response_dict(per_panelist, choice_field):
                return "enum"
            return "text"
        # sy-bn7: a JSON Schema with type=object + an enum sub-property
        # is the canonical ``--schema`` poll shape. Trust the schema and
        # return enum without scanning responses.
        if st == "object" and _json_schema_has_enum_field(schema, choice_field):
            return "enum"

    # Inference fallback. Order matters: numeric responses (scale) take
    # precedence over short strings (which would otherwise look enum-y
    # to the second branch).
    if _looks_numeric(per_panelist):
        return "scale"
    if (
        _has_choice_extraction(per_panelist)
        or _has_choice_in_response_dict(per_panelist, choice_field)
        or _looks_enum(per_panelist)
    ):
        return "enum"
    return "text"


def _resolve_field_names(
    qi: int,
    qmeta: dict[str, Any],
    per_panelist: list[dict[str, Any]],
    *,
    explicit_choice: str | None,
    explicit_confidence: str | None,
    warnings: list[str],
) -> tuple[str | None, str | None]:
    """Pick the choice/confidence field names for one question (sy-bn7).

    Resolution order:

    1. Caller-supplied ``--choice-field`` / ``--confidence-field`` wins
       outright — no warning, even if the schema disagrees.
    2. Otherwise: inspect ``response_schema`` for known choice/score
       keys (JSON Schema ``properties.<key>.enum`` or ``type: number``).
    3. Otherwise: scan the first non-null structured response dict for
       a known key.

    When a schema has multiple enum properties, the choice is the
    first field name in :data:`_FIRST_CHOICE_KEYS` that matches, then
    the first enum property in iteration order; the loser names land
    in ``warnings`` so the caller knows the tiebreak ran. Returning
    ``None`` from either slot means "no override — fall through to the
    default behaviour".
    """
    schema = qmeta.get("response_schema") if isinstance(qmeta.get("response_schema"), dict) else None

    if explicit_choice is not None:
        choice = explicit_choice
    else:
        choice = _pick_choice_field_from_schema(schema, qi, warnings)
        if choice is None:
            choice = _pick_choice_field_from_responses(per_panelist)

    if explicit_confidence is not None:
        confidence = explicit_confidence
    else:
        confidence = _pick_confidence_field_from_schema(schema)
        if confidence is None:
            confidence = _pick_confidence_field_from_responses(per_panelist)

    return choice, confidence


def _json_schema_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Return ``properties`` dict from a JSON Schema, or ``{}``."""
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def _json_schema_has_enum_field(schema: dict[str, Any], choice_field: str | None) -> bool:
    """True iff the JSON Schema has at least one enum-typed sub-property.

    When ``choice_field`` is given, restrict the check to that key —
    a caller-supplied override must actually exist in the schema for
    us to claim enum kind off it.
    """
    for name, spec in _json_schema_properties(schema).items():
        if not isinstance(spec, dict):
            continue
        if choice_field is not None and name != choice_field:
            continue
        if isinstance(spec.get("enum"), list) and spec["enum"]:
            return True
    return False


def _pick_choice_field_from_schema(
    schema: dict[str, Any] | None,
    qi: int,
    warnings: list[str],
) -> str | None:
    """Best choice-field guess from a JSON Schema's enum properties (sy-bn7).

    Deterministic tiebreak: prefer names from :data:`_FIRST_CHOICE_KEYS`
    in declared order, then fall back to the first enum property in
    iteration order. When more than one enum property is present and the
    tiebreak fires, log a warning so the caller knows which field won
    and how to override (``--choice-field``).
    """
    if not schema:
        return None
    enum_fields: list[str] = []
    for name, spec in _json_schema_properties(schema).items():
        if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and spec["enum"]:
            enum_fields.append(name)
    if not enum_fields:
        return None
    if len(enum_fields) == 1:
        return enum_fields[0]

    # Multiple enum properties — deterministic tiebreak + warning.
    for known in _FIRST_CHOICE_KEYS:
        if known in enum_fields:
            warnings.append(
                f"Q{qi}: response_schema has multiple enum fields "
                f"({', '.join(enum_fields)}); using {known!r} "
                "(pass --choice-field to override)."
            )
            return known
    chosen = enum_fields[0]
    warnings.append(
        f"Q{qi}: response_schema has multiple enum fields "
        f"({', '.join(enum_fields)}) and none match the known choice-field "
        f"names ({', '.join(_FIRST_CHOICE_KEYS)}); using {chosen!r} "
        "(pass --choice-field to override)."
    )
    return chosen


def _pick_confidence_field_from_schema(schema: dict[str, Any] | None) -> str | None:
    """Best confidence/score field guess from a JSON Schema (sy-bn7).

    Looks for a property whose name matches :data:`_SCORE_KEYS` AND
    whose type is numeric (``integer`` / ``number``). Falls back to
    the first numeric property when no known name matches.
    """
    if not schema:
        return None
    numeric_fields: list[str] = []
    for name, spec in _json_schema_properties(schema).items():
        if not isinstance(spec, dict):
            continue
        if spec.get("type") in {"integer", "number"}:
            numeric_fields.append(name)
    if not numeric_fields:
        return None
    for known in _SCORE_KEYS:
        if known in numeric_fields:
            return known
    return numeric_fields[0]


def _pick_choice_field_from_responses(per_panelist: list[dict[str, Any]]) -> str | None:
    """Sniff the choice field name from the first dict response we find."""
    for entry in per_panelist:
        rd = entry["response_dict"]
        if not isinstance(rd, dict):
            continue
        raw = rd.get("response")
        if isinstance(raw, dict):
            for key in _FIRST_CHOICE_KEYS:
                val = raw.get(key)
                if isinstance(val, str) and val.strip():
                    return key
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return key
    return None


def _pick_confidence_field_from_responses(per_panelist: list[dict[str, Any]]) -> str | None:
    """Sniff a numeric score field name from the first dict response we find."""
    for entry in per_panelist:
        rd = entry["response_dict"]
        if not isinstance(rd, dict):
            continue
        raw = rd.get("response")
        if isinstance(raw, dict):
            for key in _SCORE_KEYS:
                val = raw.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return key
    return None


def _summarise_question(
    qi: int,
    qmeta: dict[str, Any],
    per_panelist: list[dict[str, Any]],
    kind: str,
    warnings: list[str],
    *,
    choice_field: str | None = None,
    confidence_field: str | None = None,
) -> QuestionSummary:
    """Build a :class:`QuestionSummary` for one question."""
    question_text = str(qmeta.get("text") or "")[:280]
    n = sum(1 for v in per_panelist if v["response_dict"] is not None)

    if kind == "enum":
        return _summarise_enum(
            qi,
            question_text,
            per_panelist,
            n,
            warnings,
            choice_field=choice_field,
            confidence_field=confidence_field,
        )
    if kind == "scale":
        return _summarise_scale(
            qi,
            question_text,
            per_panelist,
            n,
            warnings,
            confidence_field=confidence_field,
        )
    return QuestionSummary(question=question_text, question_index=qi, kind="text", n=n)


def _summarise_enum(
    qi: int,
    question_text: str,
    per_panelist: list[dict[str, Any]],
    n: int,
    warnings: list[str],
    *,
    choice_field: str | None = None,
    confidence_field: str | None = None,
) -> QuestionSummary:
    """Enum rollup: vote counts, optional second-choice + weighted scores.

    sy-bn7: also populates ``metric_average`` (overall average across
    every panelist who supplied a score) so enum polls with a numeric
    confidence field surface a panel-wide confidence reading, not just
    per-option weights.
    """
    first: Counter[str] = Counter()
    second: Counter[str] = Counter()
    scores: dict[str, list[float]] = {}
    all_scores: list[float] = []
    unparseable = 0

    for entry in per_panelist:
        rd = entry["response_dict"]
        if rd is None:
            continue
        choice = _extract_first_choice(rd, choice_field=choice_field)
        if choice is None:
            unparseable += 1
            continue
        first[choice] += 1
        sc = _extract_second_choice(rd)
        if sc is not None and sc != choice:
            second[sc] += 1
        score = _extract_score(rd, score_field=confidence_field)
        if score is not None:
            scores.setdefault(choice, []).append(score)
            all_scores.append(score)

    if unparseable:
        warnings.append(
            f"Q{qi}: {unparseable}/{n} response{'s' if unparseable != 1 else ''} "
            "could not be mapped to an option (response was free-text or empty)."
        )

    first_counts = _sorted_counts(first)
    second_counts = _sorted_counts(second) if second else None
    weighted = {opt: round(sum(vals) / len(vals), 4) for opt, vals in scores.items() if vals} if scores else None
    overall_avg = round(sum(all_scores) / len(all_scores), 4) if all_scores else None
    winner = next(iter(first_counts), None) if first_counts else None

    return QuestionSummary(
        question=question_text,
        question_index=qi,
        kind="enum",
        n=n,
        n_unparseable=unparseable,
        first_choice_counts=first_counts or None,
        second_choice_counts=second_counts,
        weighted_scores=weighted,
        winner=winner,
        metric_average=overall_avg,
    )


def _summarise_scale(
    qi: int,
    question_text: str,
    per_panelist: list[dict[str, Any]],
    n: int,
    warnings: list[str],
    *,
    confidence_field: str | None = None,
) -> QuestionSummary:
    """Scale rollup: mean + histogram."""
    values: list[float] = []
    histogram: Counter[str] = Counter()
    unparseable = 0
    for entry in per_panelist:
        rd = entry["response_dict"]
        if rd is None:
            continue
        v = _extract_score(rd, score_field=confidence_field)
        if v is None:
            unparseable += 1
            continue
        values.append(v)
        # Histogram bucket: render as int when the value is integral
        # (Likert 1..5 produces "1".."5", not "1.0".."5.0").
        bucket = str(int(v)) if v == int(v) else f"{v:g}"
        histogram[bucket] += 1

    if unparseable:
        warnings.append(
            f"Q{qi}: {unparseable}/{n} response{'s' if unparseable != 1 else ''} "
            "could not be mapped to a number (response was non-numeric)."
        )

    avg = round(sum(values) / len(values), 4) if values else None
    # Sort histogram by numeric value of the key for stable, scannable
    # output ("1","2","3","4","5") rather than insertion order.
    sorted_hist: dict[str, int] | None = None
    if histogram:
        sorted_hist = dict(sorted(histogram.items(), key=lambda kv: float(kv[0])))

    return QuestionSummary(
        question=question_text,
        question_index=qi,
        kind="scale",
        n=n,
        n_unparseable=unparseable,
        metric_average=avg,
        metric_distribution=sorted_hist,
    )


# ---------------------------------------------------------------------------
# Field extractors — robust to multiple shapes
# ---------------------------------------------------------------------------


def _extract_first_choice(
    response_dict: dict[str, Any],
    *,
    choice_field: str | None = None,
) -> str | None:
    """Pull the panelist's primary choice from a response record.

    Checks (in order): ``extraction`` dict, then the raw ``response``
    field (string for enum schemas, dict for ``--schema`` structured
    output, sy-bn7). When ``choice_field`` is given, that key is tried
    first and the known-key fallback runs only if it's absent — this
    lets ``--choice-field vote`` win over an unrelated ``answer`` key
    that happens to be in the same dict. Returns ``None`` when nothing
    parseable is found — callers add to ``n_unparseable`` rather than
    guess.
    """
    keys: tuple[str, ...]
    if choice_field:
        keys = (choice_field, *(k for k in _FIRST_CHOICE_KEYS if k != choice_field))
    else:
        keys = _FIRST_CHOICE_KEYS

    # sy-547: the orchestrator coerces enum answers against the declared
    # response_schema and persists the canonical option under
    # ``response_typed``. Trust it first — it's already mapped to a valid
    # option (e.g. ``"Blue."`` → ``"blue"``), so it wins over re-parsing
    # the raw free text.
    typed = response_dict.get("response_typed")
    if isinstance(typed, str) and typed.strip():
        return typed.strip()
    if isinstance(typed, (int, float)) and not isinstance(typed, bool):
        return str(typed)

    # sy-547: when the response carries an enum response_schema but no typed
    # value, the orchestrator already determined the free text could not be
    # mapped to a declared option (``schema_unmapped``). Do NOT fall back to
    # the raw string — counting it as a choice would resurrect exactly the
    # off-schema free-text the coercion layer rejected. Report unparseable.
    resp_schema = response_dict.get("response_schema")
    if isinstance(resp_schema, dict) and resp_schema.get("type") == "enum":
        return None

    extraction = response_dict.get("extraction")
    if isinstance(extraction, dict):
        for key in keys:
            val = extraction.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return str(val)

    raw = response_dict.get("response")
    if isinstance(raw, str) and raw.strip():
        # Enum schemas store the picked option directly as the response
        # string. Truncate aggressively — a free-text fallthrough can be
        # an entire paragraph and would pollute the counts dict.
        text = raw.strip()
        if len(text) <= 120:
            return text
    if isinstance(raw, dict):
        for key in keys:
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return str(val)
    return None


def _extract_second_choice(response_dict: dict[str, Any]) -> str | None:
    extraction = response_dict.get("extraction")
    if isinstance(extraction, dict):
        for key in _SECOND_CHOICE_KEYS:
            val = extraction.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _extract_score(
    response_dict: dict[str, Any],
    *,
    score_field: str | None = None,
) -> float | None:
    """Coerce a numeric rating out of a response record.

    Returns ``None`` when nothing numeric is found. Booleans are NOT
    treated as numbers — ``True/False`` showing up here is a schema
    mismatch the caller should learn about via ``n_unparseable``.

    sy-bn7: also reads ``response`` when it's a dict (the shape
    ``--schema`` runs persist) so a ``{"choice": ..., "confidence": 4}``
    response surfaces the confidence number. When ``score_field`` is
    given, that key is tried first.
    """
    keys: tuple[str, ...]
    if score_field:
        keys = (score_field, *(k for k in _SCORE_KEYS if k != score_field))
    else:
        keys = _SCORE_KEYS

    # sy-547: prefer the schema-coerced integer the orchestrator persisted
    # for scale questions (already validated to be inside [min, max]).
    typed = response_dict.get("response_typed")
    if isinstance(typed, (int, float)) and not isinstance(typed, bool):
        return float(typed)

    raw = response_dict.get("response")
    # Top-level numeric response wins when the whole response is a
    # scalar (free-text scale schemas). For dict responses we look
    # inside for a recognized score key.
    if not isinstance(raw, dict):
        val = _coerce_number(raw)
        if val is not None:
            return val
    else:
        for key in keys:
            v = _coerce_number(raw.get(key))
            if v is not None:
                return v

    extraction = response_dict.get("extraction")
    if isinstance(extraction, dict):
        for key in keys:
            v = _coerce_number(extraction.get(key))
            if v is not None:
                return v
    return None


def _coerce_number(value: Any) -> float | None:
    """Best-effort numeric coercion. Strict on booleans."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        # Pull the first numeric token — "4/5" → 4.0, "rating: 3" → 3.0.
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _harvest_objections(entry: dict[str, Any]) -> list[str]:
    """Pull objections from a per-panelist response view.

    Walks ``extraction`` for any of the objection keys (each may carry
    a string or a list of strings). Free-text responses with explicit
    "concern:" / "problem:" prefixes also count — those are how
    free-text panelists tend to phrase the same signal.
    """
    rd = entry.get("response_dict")
    if not isinstance(rd, dict):
        return []

    out: list[str] = []
    extraction = rd.get("extraction")
    if isinstance(extraction, dict):
        for key in _OBJECTION_KEYS:
            val = extraction.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
            elif isinstance(val, list):
                out.extend(str(v).strip() for v in val if isinstance(v, (str, int, float)) and str(v).strip())
    return out


# ---------------------------------------------------------------------------
# Heuristics for kind detection
# ---------------------------------------------------------------------------


def _looks_numeric(per_panelist: list[dict[str, Any]]) -> bool:
    """True iff a majority of present top-level responses parse as numbers.

    sy-bn7: only the scalar response form (or the extraction-only shape)
    counts. A structured ``--schema`` response like
    ``{"choice": "B", "confidence": 4}`` is NOT a scale answer — the
    response is a dict carrying both a categorical choice and a numeric
    confidence. Letting the confidence drive the "looks numeric" probe
    would misclassify the whole question as ``scale`` and bury the vote.

    sy-oyl: this is a *classification* probe — it decides whether a
    question without a schema is a scale. We must NOT extract numbers
    greedily from arbitrary prose here, or any free-text answer mentioning
    a version string ("SynthPanel v1.5.1…") gets classified as a scale
    response. :func:`_response_is_scale_shaped` only accepts responses
    whose entire string is a numeric scale answer (``"4"``, ``"4/5"``,
    ``"rating: 4"``) — the post-classification scorer
    (:func:`_extract_score`) keeps its lenient regex so legitimately-
    classified scale runs still accept verbose answers.
    """
    numeric = 0
    seen = 0
    for entry in per_panelist:
        rd = entry["response_dict"]
        if rd is None:
            continue
        seen += 1
        # Dict response → structured, not a bare numeric. Skip without
        # incrementing numeric so the question stays available for the
        # enum check downstream.
        if isinstance(rd.get("response"), dict):
            continue
        if _response_is_scale_shaped(rd):
            numeric += 1
    return seen > 0 and numeric / seen > 0.5


# sy-oyl: the classification-time scale gate. A scale answer is the
# *whole* response, not a digit buried inside a sentence. Accepted shapes:
#
#   "4"                  bare integer
#   "4.5"                bare decimal
#   "-4"                 negative scale (rare but valid for delta/Likert)
#   "4/5"                fraction
#   "4 out of 5"         English fraction
#   "rating: 4"          short labelled answer
#   "score = 4 / 5"      labelled fraction with light prose
#
# Rejected (the bug we're fixing):
#
#   "SynthPanel v1.5.1 would absolutely feel agent-ready…"  (text + version)
#   "The fourth one, around 7/10 confidence overall."        (essay)
#
# The 32-character cap on the *raw* response is what enforces "this is
# meant to be a scale, not an essay". Labels like ``score:`` / ``rating:``
# / ``confidence:`` are stripped before the cap so the realistic short
# forms still match.
_SCALE_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:rating|score|confidence|value|answer)\s*[:=]\s*",
    re.IGNORECASE,
)
_SCALE_RESPONSE_RE = re.compile(
    r"^\s*-?\d+(?:\.\d+)?(?:\s*(?:/|out\s+of)\s*\d+)?\s*$",
    re.IGNORECASE,
)


def _response_is_scale_shaped(response_dict: dict[str, Any]) -> bool:
    """True iff the response, taken whole, is a numeric scale answer (sy-oyl).

    See :data:`_SCALE_RESPONSE_RE` for the accepted shapes. Numeric
    scalars and bare ``int``/``float`` values always pass; booleans and
    dicts always fail. String responses must match the strict pattern
    after stripping a leading ``rating: `` / ``score = `` label and must
    fit inside the 32-character "this is a scale answer, not prose" cap.

    Extraction-only scoring (``{"extraction": {"score": 4}}``) also
    counts — that's the canonical structured-scale shape.
    """
    raw = response_dict.get("response")
    if isinstance(raw, bool):
        # Same booleans-aren't-numbers stance as ``_coerce_number``.
        return False
    if isinstance(raw, (int, float)):
        return True
    if isinstance(raw, str):
        stripped = _SCALE_LABEL_PREFIX_RE.sub("", raw).strip()
        return len(stripped) <= 32 and _SCALE_RESPONSE_RE.match(stripped) is not None

    # No bare scalar — look for an extracted score on a free-text response.
    extraction = response_dict.get("extraction")
    if isinstance(extraction, dict):
        for key in _SCORE_KEYS:
            v = extraction.get(key)
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                return True
    return False


def _looks_enum(per_panelist: list[dict[str, Any]]) -> bool:
    """True iff responses repeat heavily across a small option set.

    A 12-panelist run where 5 say "A", 4 say "B", 3 say "C" is enum-shaped
    even without a schema. A 12-panelist run where every response is a
    different sentence is text-shaped.
    """
    values: list[str] = []
    for entry in per_panelist:
        rd = entry["response_dict"]
        if rd is None:
            continue
        v = _extract_first_choice(rd)
        if v is not None:
            values.append(v)
    if len(values) < 3:
        return False
    unique = len(set(values))
    # Enum-shaped: at most a quarter of responses are unique values.
    return unique <= max(2, len(values) // 4)


def _has_choice_extraction(per_panelist: list[dict[str, Any]]) -> bool:
    """True iff any panelist's extraction carries a recognised choice key."""
    for entry in per_panelist:
        rd = entry["response_dict"]
        if not isinstance(rd, dict):
            continue
        extraction = rd.get("extraction")
        if isinstance(extraction, dict):
            for key in _FIRST_CHOICE_KEYS:
                if isinstance(extraction.get(key), (str, int, float)) and not isinstance(extraction.get(key), bool):
                    return True
    return False


def _has_choice_in_response_dict(
    per_panelist: list[dict[str, Any]],
    choice_field: str | None = None,
) -> bool:
    """True iff any panelist's ``response`` is a dict with a choice key (sy-bn7).

    This is the missing detector that was letting ``--schema`` runs fall
    through to ``text``. When ``choice_field`` is given, only that key
    counts; otherwise the function probes :data:`_FIRST_CHOICE_KEYS`.
    """
    keys: tuple[str, ...] = (choice_field,) if choice_field else _FIRST_CHOICE_KEYS
    for entry in per_panelist:
        rd = entry["response_dict"]
        if not isinstance(rd, dict):
            continue
        raw = rd.get("response")
        if not isinstance(raw, dict):
            continue
        for key in keys:
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return True
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return True
    return False


# ---------------------------------------------------------------------------
# Segment splits
# ---------------------------------------------------------------------------


def _segment_split(
    per_panelist: list[dict[str, Any]],
    persona_index: dict[str, dict[str, Any]],
    segment_attrs: list[str],
    *,
    choice_field: str | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    """Return ``attr -> attr_value -> {option: count}`` for one question.

    Pulls attribute values from the per-panelist persona snapshot first,
    falling back to the explicit ``personas`` list when present. Empty
    attributes are dropped; the table contains only attributes where at
    least one panelist had a value AND at least one made a parseable
    choice.
    """
    table: dict[str, dict[str, dict[str, int]]] = {}
    for attr in segment_attrs:
        rows: dict[str, dict[str, int]] = {}
        for entry in per_panelist:
            rd = entry["response_dict"]
            if not isinstance(rd, dict):
                continue
            choice = _extract_first_choice(rd, choice_field=choice_field)
            if choice is None:
                continue

            attrs_inline = entry.get("persona_attrs") or {}
            value = attrs_inline.get(attr)
            if value is None:
                # Look up from the explicit personas index.
                lookup = persona_index.get(entry.get("persona") or "", {})
                value = lookup.get(attr)
            if value is None or value == "":
                continue

            v_str = str(value)
            rows.setdefault(v_str, {})
            rows[v_str][choice] = rows[v_str].get(choice, 0) + 1

        if rows:
            # Cap to keep the table scannable — large persona packs can
            # produce dozens of values per attribute.
            trimmed = dict(
                sorted(
                    rows.items(),
                    key=lambda kv: sum(kv[1].values()),
                    reverse=True,
                )[:_PER_SEGMENT_VALUE_CAP]
            )
            table[attr] = trimmed
    return table


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    """Render a Counter as a count-descending dict (ties keep insertion order)."""
    return dict(counter.most_common())


def _extract_recommendation(result: dict[str, Any]) -> str | None:
    """Pull ``synthesis.recommendation`` from a result envelope.

    The recommendation field is the part of the synthesis layer that most
    closely matches "recommended next test". Looks at the top-level
    ``synthesis``, then the last round's synthesis (the natural
    "terminal" synthesis for v3 branching runs).
    """
    syn = result.get("synthesis")
    if isinstance(syn, dict):
        rec = syn.get("recommendation")
        if isinstance(rec, str) and rec.strip():
            return rec.strip()

    rounds = result.get("rounds")
    if isinstance(rounds, list) and rounds:
        last = rounds[-1]
        if isinstance(last, dict) and isinstance(last.get("synthesis"), dict):
            rec = last["synthesis"].get("recommendation")
            if isinstance(rec, str) and rec.strip():
                return rec.strip()
    return None
