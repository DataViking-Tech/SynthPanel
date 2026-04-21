"""Inspect a saved panel result: schema walker and human summary.

The CLI subcommand ``synthpanel panel inspect <result>`` renders the
report below without making any LLM calls. It works on two result
shapes:

* the flat shape written by :func:`synth_panel.mcp.data.save_panel_result`
  (``results`` at the top level, no ``rounds``, no ``synthesis``), and
* the rounds-shaped payload emitted by ``--output-format json`` which
  carries ``rounds[]``, ``synthesis``, ``failure_stats``, ``metadata``
  and per-model cost breakdowns.

The report is intentionally independent of :mod:`synth_panel.analyze`
(which runs statistical tests and needs SciPy-shaped inputs) — this
module only reads structure, so it stays cheap and importable in
minimal environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_SYNTH_SUMMARY_PEEK = 300


@dataclass
class PersonaSummary:
    """Per-persona rollup for the inspect report."""

    name: str
    model: str | None
    response_count: int
    error_count: int
    panelist_error: str | None


@dataclass
class ModelRollup:
    """Per-model rollup aggregated across panelists."""

    model: str
    personas: int
    questions_answered: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float | None


@dataclass
class SynthesisStatus:
    """Whether synthesis ran and, if so, the gist of its output."""

    ran: bool
    model: str | None
    cost: str | None
    summary_peek: str | None
    error: str | None
    theme_count: int


@dataclass
class FailureStats:
    """Panel-run failure counters."""

    total_pairs: int
    errored_pairs: int
    failure_rate: float
    failed_panelists: int
    errored_personas: list[str]


@dataclass
class InspectReport:
    """Structured output of ``panel inspect``."""

    result_id: str | None
    created_at: str | None
    model: str | None
    models: list[str]
    persona_count: int
    question_count: int
    round_names: list[str]
    path: list[str]
    total_tokens: int
    total_cost: str | None
    panelist_cost: str | None
    instrument_name: str | None
    warnings: list[str]
    personas: list[PersonaSummary]
    model_rollup: list[ModelRollup]
    synthesis: SynthesisStatus
    failure_stats: FailureStats
    has_rounds_shape: bool
    extraction_present: bool
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "created_at": self.created_at,
            "model": self.model,
            "models": self.models,
            "persona_count": self.persona_count,
            "question_count": self.question_count,
            "round_names": self.round_names,
            "path": self.path,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "panelist_cost": self.panelist_cost,
            "instrument_name": self.instrument_name,
            "warnings": list(self.warnings),
            "personas": [
                {
                    "name": p.name,
                    "model": p.model,
                    "response_count": p.response_count,
                    "error_count": p.error_count,
                    "panelist_error": p.panelist_error,
                }
                for p in self.personas
            ],
            "model_rollup": [
                {
                    "model": m.model,
                    "personas": m.personas,
                    "questions_answered": m.questions_answered,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_tokens": m.total_tokens,
                    "cost_usd": m.cost_usd,
                }
                for m in self.model_rollup
            ],
            "synthesis": {
                "ran": self.synthesis.ran,
                "model": self.synthesis.model,
                "cost": self.synthesis.cost,
                "summary_peek": self.synthesis.summary_peek,
                "error": self.synthesis.error,
                "theme_count": self.synthesis.theme_count,
            },
            "failure_stats": {
                "total_pairs": self.failure_stats.total_pairs,
                "errored_pairs": self.failure_stats.errored_pairs,
                "failure_rate": self.failure_stats.failure_rate,
                "failed_panelists": self.failure_stats.failed_panelists,
                "errored_personas": list(self.failure_stats.errored_personas),
            },
            "has_rounds_shape": self.has_rounds_shape,
            "extraction_present": self.extraction_present,
        }


def _is_rounds_shape(data: dict[str, Any]) -> bool:
    rounds = data.get("rounds")
    return isinstance(rounds, list) and bool(rounds)


def _iter_panelist_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the flat list of panelist result dicts across shapes.

    For the rounds-shaped payload we concatenate results from every
    round so per-model rollups reflect the full run. For the flat
    shape we just return ``results``.
    """
    if _is_rounds_shape(data):
        out: list[dict[str, Any]] = []
        for rnd in data.get("rounds") or []:
            for entry in rnd.get("results") or []:
                if isinstance(entry, dict):
                    out.append(entry)
        return out
    results = data.get("results") or []
    return [r for r in results if isinstance(r, dict)]


def _get_round_names(data: dict[str, Any]) -> list[str]:
    if _is_rounds_shape(data):
        return [str(r.get("name", "")) for r in data.get("rounds") or []]
    return ["default"]


def _format_path(path_entries: list[Any]) -> list[str]:
    parts: list[str] = []
    for entry in path_entries or []:
        if not isinstance(entry, dict):
            continue
        rnd = entry.get("round", "")
        branch = entry.get("branch", "")
        nxt = entry.get("next", "")
        if branch and branch != "linear":
            parts.append(f"{rnd}[{branch}]->{nxt}")
        else:
            parts.append(f"{rnd}->{nxt}" if nxt else str(rnd))
    return parts


def _collect_persona_summaries(panelists: list[dict[str, Any]]) -> list[PersonaSummary]:
    out: list[PersonaSummary] = []
    for p in panelists:
        name = str(p.get("persona", "unknown"))
        model = p.get("model")
        responses = p.get("responses") or []
        resp_count = sum(1 for r in responses if isinstance(r, dict))
        err_count = sum(1 for r in responses if isinstance(r, dict) and r.get("error") and not r.get("follow_up"))
        panelist_error = p.get("error")
        out.append(
            PersonaSummary(
                name=name,
                model=model,
                response_count=resp_count,
                error_count=err_count,
                panelist_error=panelist_error if panelist_error else None,
            )
        )
    return out


def _collect_model_rollup(
    panelists: list[dict[str, Any]],
    fallback_model: str | None,
    metadata: dict[str, Any] | None,
) -> list[ModelRollup]:
    """Roll up tokens, cost, personas, and questions-answered by model.

    ``metadata.cost.per_model`` carries authoritative cost/token splits
    when present (it's built in :mod:`synth_panel.metadata`). We fall
    back to summing per-panelist ``usage`` dicts when the metadata
    block is missing (older saves, ``save_panel_result``).
    """
    by_model: dict[str, dict[str, Any]] = {}

    def _bucket(model: str) -> dict[str, Any]:
        if model not in by_model:
            by_model[model] = {
                "model": model,
                "personas": 0,
                "questions_answered": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": None,
            }
        return by_model[model]

    for p in panelists:
        model = p.get("model") or fallback_model or "unknown"
        bucket = _bucket(str(model))
        bucket["personas"] += 1
        usage = p.get("usage") or {}
        if isinstance(usage, dict):
            bucket["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            bucket["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        for resp in p.get("responses") or []:
            if not isinstance(resp, dict):
                continue
            if resp.get("follow_up"):
                continue
            if not resp.get("error"):
                bucket["questions_answered"] += 1

    for bucket in by_model.values():
        bucket["total_tokens"] = bucket["input_tokens"] + bucket["output_tokens"]

    if metadata and isinstance(metadata.get("cost"), dict):
        per_model = metadata["cost"].get("per_model") or {}
        if isinstance(per_model, dict):
            for model, stats in per_model.items():
                if not isinstance(stats, dict):
                    continue
                bucket = _bucket(str(model))
                cost_val = stats.get("cost_usd")
                if isinstance(cost_val, (int, float)):
                    bucket["cost_usd"] = float(cost_val)

    return [
        ModelRollup(
            model=b["model"],
            personas=b["personas"],
            questions_answered=b["questions_answered"],
            input_tokens=b["input_tokens"],
            output_tokens=b["output_tokens"],
            total_tokens=b["total_tokens"],
            cost_usd=b["cost_usd"],
        )
        for b in sorted(by_model.values(), key=lambda x: x["model"])
    ]


def _collect_synthesis(data: dict[str, Any]) -> SynthesisStatus:
    synth = data.get("synthesis")
    if not isinstance(synth, dict):
        return SynthesisStatus(
            ran=False,
            model=None,
            cost=None,
            summary_peek=None,
            error=None,
            theme_count=0,
        )
    error = synth.get("synthesis_error") or synth.get("error")
    summary = synth.get("summary")
    peek: str | None = None
    if isinstance(summary, str) and summary:
        peek = summary[:_SYNTH_SUMMARY_PEEK]
        if len(summary) > _SYNTH_SUMMARY_PEEK:
            peek += "…"
    themes = synth.get("themes") or []
    return SynthesisStatus(
        ran=error is None and summary is not None,
        model=synth.get("model"),
        cost=synth.get("cost"),
        summary_peek=peek,
        error=str(error) if error else None,
        theme_count=len(themes) if isinstance(themes, list) else 0,
    )


def _collect_failure_stats(
    data: dict[str, Any],
    panelists: list[dict[str, Any]],
    question_count: int,
) -> FailureStats:
    """Prefer the persisted ``failure_stats`` block; reconstruct from
    per-panelist response errors otherwise so older / flat saves still
    surface a rate.
    """
    fs = data.get("failure_stats")
    if isinstance(fs, dict):
        return FailureStats(
            total_pairs=int(fs.get("total_pairs", 0) or 0),
            errored_pairs=int(fs.get("errored_pairs", 0) or 0),
            failure_rate=float(fs.get("failure_rate", 0.0) or 0.0),
            failed_panelists=int(fs.get("failed_panelists", 0) or 0),
            errored_personas=list(fs.get("errored_personas") or []),
        )

    total = 0
    errored = 0
    failed_panelists = 0
    bad_personas: set[str] = set()
    for p in panelists:
        name = str(p.get("persona", "unknown"))
        if p.get("error"):
            failed_panelists += 1
            errored += question_count
            total += question_count
            bad_personas.add(name)
            continue
        seen = 0
        bad = 0
        for resp in p.get("responses") or []:
            if not isinstance(resp, dict) or resp.get("follow_up"):
                continue
            seen += 1
            if resp.get("error"):
                bad += 1
        if question_count and seen < question_count:
            shortfall = question_count - seen
            bad += shortfall
            seen += shortfall
        total += seen
        errored += bad
        if bad > 0:
            bad_personas.add(name)
    rate = (errored / total) if total > 0 else 0.0
    return FailureStats(
        total_pairs=total,
        errored_pairs=errored,
        failure_rate=rate,
        failed_panelists=failed_panelists,
        errored_personas=sorted(bad_personas),
    )


def _extraction_present(panelists: list[dict[str, Any]]) -> bool:
    for p in panelists:
        for resp in p.get("responses") or []:
            if isinstance(resp, dict) and resp.get("extraction") is not None:
                return True
    return False


def build_inspect_report(data: dict[str, Any]) -> InspectReport:
    """Walk a loaded panel-result dict and build an :class:`InspectReport`."""
    panelists = _iter_panelist_entries(data)
    metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else None
    models_list = data.get("models")
    if not isinstance(models_list, list):
        models_list = []
    if not models_list:
        seen: list[str] = []
        for p in panelists:
            m = p.get("model")
            if isinstance(m, str) and m not in seen:
                seen.append(m)
        models_list = seen

    total_usage = data.get("total_usage") or {}
    total_tokens = 0
    if isinstance(total_usage, dict):
        total_tokens = int(total_usage.get("input_tokens", 0) or 0) + int(total_usage.get("output_tokens", 0) or 0)

    question_count = int(data.get("question_count") or 0)
    if not question_count and panelists:
        first_responses = panelists[0].get("responses") or []
        question_count = sum(1 for r in first_responses if isinstance(r, dict) and not r.get("follow_up"))

    persona_count = int(data.get("persona_count") or 0) or len(panelists)

    warnings = data.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = []

    return InspectReport(
        result_id=data.get("id"),
        created_at=data.get("created_at"),
        model=data.get("model"),
        models=list(models_list),
        persona_count=persona_count,
        question_count=question_count,
        round_names=_get_round_names(data),
        path=_format_path(data.get("path") or []),
        total_tokens=total_tokens,
        total_cost=data.get("total_cost"),
        panelist_cost=data.get("panelist_cost"),
        instrument_name=data.get("instrument_name"),
        warnings=[str(w) for w in warnings],
        personas=_collect_persona_summaries(panelists),
        model_rollup=_collect_model_rollup(panelists, data.get("model"), metadata),
        synthesis=_collect_synthesis(data),
        failure_stats=_collect_failure_stats(data, panelists, question_count),
        has_rounds_shape=_is_rounds_shape(data),
        extraction_present=_extraction_present(panelists),
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------

_BAR = "=" * 60
_THIN = "-" * 60


def _section(title: str) -> list[str]:
    return [title, _THIN]


def format_inspect_text(report: InspectReport) -> str:
    """Render an :class:`InspectReport` as a human-friendly text block."""
    lines: list[str] = []
    header_title = report.result_id or "(unsaved result)"
    lines.append(_BAR)
    lines.append(f"Panel Result: {header_title}")
    lines.append(_BAR)
    if report.created_at:
        lines.append(f"Created:       {report.created_at}")
    if report.instrument_name:
        lines.append(f"Instrument:    {report.instrument_name}")
    lines.append(f"Model:         {report.model or '(unknown)'}")
    if report.models and report.models != [report.model]:
        lines.append(f"Models:        {', '.join(report.models)}")
    lines.append(f"Personas:      {report.persona_count}")
    lines.append(f"Questions:     {report.question_count}")
    rounds_line = f"{len(report.round_names)}"
    if report.round_names:
        rounds_line += f" ({', '.join(report.round_names)})"
    lines.append(f"Rounds:        {rounds_line}")
    if report.path:
        lines.append(f"Path:          {' '.join(report.path)}")
    lines.append(f"Total tokens:  {report.total_tokens}")
    if report.panelist_cost:
        lines.append(f"Panelist cost: {report.panelist_cost}")
    if report.total_cost:
        lines.append(f"Total cost:    {report.total_cost}")
    if report.extraction_present:
        lines.append("Extraction:    present (post-hoc structured extraction used)")
    if not report.has_rounds_shape:
        lines.append("Shape:         flat (saved via --save; no rounds metadata)")

    lines.append("")
    lines.extend(_section("Per-Persona Summary"))
    if not report.personas:
        lines.append("  (no panelists recorded)")
    else:
        for p in report.personas:
            parts = [f"  {p.name:<30}"]
            if p.model:
                parts.append(f"model={p.model}")
            parts.append(f"responses={p.response_count}")
            parts.append(f"errors={p.error_count}")
            lines.append("  ".join(parts))
            if p.panelist_error:
                lines.append(f"      panelist error: {p.panelist_error}")

    lines.append("")
    lines.extend(_section("Per-Model Rollup"))
    if not report.model_rollup:
        lines.append("  (no model usage recorded)")
    else:
        for m in report.model_rollup:
            lines.append(f"  {m.model}")
            lines.append(f"      personas:           {m.personas}")
            lines.append(f"      questions answered: {m.questions_answered}")
            lines.append(f"      tokens:             {m.total_tokens} (in={m.input_tokens} / out={m.output_tokens})")
            if m.cost_usd is not None:
                lines.append(f"      cost:               ${m.cost_usd:.4f}")

    lines.append("")
    lines.extend(_section("Synthesis"))
    s = report.synthesis
    if not s.ran and s.error is None and s.summary_peek is None:
        lines.append("  Status:  not run")
    elif s.error and not s.ran:
        lines.append(f"  Status:  failed — {s.error}")
    else:
        lines.append("  Status:  ran")
        if s.model:
            lines.append(f"  Model:   {s.model}")
        if s.cost:
            lines.append(f"  Cost:    {s.cost}")
        if s.theme_count:
            lines.append(f"  Themes:  {s.theme_count}")
        if s.summary_peek:
            lines.append("  Summary peek:")
            for chunk in _wrap(s.summary_peek, width=70):
                lines.append(f"    {chunk}")

    lines.append("")
    lines.extend(_section("Failure Stats"))
    fs = report.failure_stats
    lines.append(f"  Total pairs:   {fs.total_pairs}")
    lines.append(f"  Errored pairs: {fs.errored_pairs}")
    lines.append(f"  Failure rate:  {fs.failure_rate:.1%}")
    if fs.failed_panelists:
        lines.append(f"  Failed panelists: {fs.failed_panelists}")
    if fs.errored_personas:
        shown = ", ".join(fs.errored_personas[:6])
        extra = len(fs.errored_personas) - 6
        if extra > 0:
            shown += f", +{extra} more"
        lines.append(f"  Affected personas: {shown}")

    if report.warnings:
        lines.append("")
        lines.extend(_section("Warnings"))
        for w in report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _wrap(text: str, *, width: int) -> list[str]:
    """Tiny word-wrap helper — avoids importing :mod:`textwrap` for one call."""
    words = text.split()
    if not words:
        return [text]
    out: list[str] = []
    cur = ""
    for word in words:
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur = f"{cur} {word}"
        else:
            out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out
