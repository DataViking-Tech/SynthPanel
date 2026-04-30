"""Standalone cost summary aggregation for saved panel runs (sy-kmw1).

Walks the saved-results directory (default ``~/.synthpanel/results/``),
loads each ``result-*.json`` produced by ``save_panel_result``, and
aggregates per-run / per-panelist cost into model and month buckets.

The on-disk schema is the one written by
:func:`synth_panel.mcp.data.save_panel_result`. Sidecars (``*.pre-extend.json``,
``*.synthesis-*.json``) and any non-result JSON are skipped.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Cost strings on disk are produced by ``CostEstimate.format_usd`` ("$X.XXXX"),
# but tolerate plain numbers and stray whitespace so an externally-written
# result still loads.
_COST_RE = re.compile(r"^\s*\$?\s*([-+]?\d+(?:\.\d+)?)\s*$")


def _parse_cost_str(value: Any) -> float | None:
    """Parse a saved cost field (typically ``"$0.0234"``) to a float USD value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _COST_RE.match(value)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def _parse_iso_datetime(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp; return ``None`` on failure."""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_since(value: str) -> datetime:
    """Parse a ``--since`` argument (``YYYY-MM-DD`` or full ISO datetime).

    Raises :class:`ValueError` on unparseable input. Pure date inputs are
    treated as 00:00 UTC of that day so a date-only ``--since`` includes
    everything from midnight onward.
    """
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    return parsed


@dataclass
class PanelistCost:
    """Per-panelist cost record extracted from a saved run."""

    model: str
    cost_usd: float
    turns: int


@dataclass
class RunInfo:
    """Per-run summary distilled from one ``result-*.json``."""

    path: Path
    run_id: str
    created_at: datetime | None
    model: str
    models: list[str]
    total_cost_usd: float
    panelists: list[PanelistCost]
    persona_count: int
    question_count: int
    is_partial: bool

    @property
    def turn_count(self) -> int:
        return sum(p.turns for p in self.panelists)


@dataclass
class ModelStat:
    """Aggregated stats for a single model across runs."""

    cost_usd: float = 0.0
    runs: int = 0
    turns: int = 0


@dataclass
class SummaryReport:
    """Final aggregated cost report."""

    total_cost_usd: float = 0.0
    run_count: int = 0
    partial_count: int = 0
    skipped_count: int = 0
    by_model: dict[str, ModelStat] = field(default_factory=dict)
    by_month: dict[str, float] = field(default_factory=dict)
    by_run: list[RunInfo] = field(default_factory=list)


def default_runs_dir() -> Path:
    """Resolve the default runs directory.

    Mirrors :func:`synth_panel.mcp.data._results_dir` without invoking it
    so this module remains independent of the MCP-data layer's mkdir
    side effect: ``$SYNTH_PANEL_DATA_DIR/results/`` (default
    ``~/.synthpanel/results/``).
    """
    import os

    base = Path(os.environ.get("SYNTH_PANEL_DATA_DIR", "~/.synthpanel")).expanduser()
    return base / "results"


def discover_run_files(runs_dir: Path) -> list[Path]:
    """Return candidate result JSONs under *runs_dir*, in name-sorted order.

    Excludes ``*.pre-extend.json`` and ``*.synthesis-*.json`` sidecars
    so they aren't double-counted alongside their parent result.
    """
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(runs_dir.glob("*.json")):
        name = p.name
        if name.endswith(".pre-extend.json"):
            continue
        # synthesis sidecars: <result-id>.synthesis-<timestamp>.json
        if ".synthesis-" in name:
            continue
        out.append(p)
    return out


def _extract_panelists(
    raw_results: list[dict[str, Any]],
    top_level_model: str,
) -> list[PanelistCost]:
    """Pull (model, cost, turns) for each panelist row in *raw_results*."""
    out: list[PanelistCost] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        cost = _parse_cost_str(row.get("cost"))
        if cost is None:
            cost = 0.0
        model = row.get("model") or top_level_model or "(unknown)"
        responses = row.get("responses") or []
        turns = len(responses) if isinstance(responses, list) else 0
        out.append(PanelistCost(model=str(model), cost_usd=float(cost), turns=int(turns)))
    return out


def parse_run(path: Path) -> RunInfo | None:
    """Parse a single result file. Returns ``None`` for unrecognized shapes."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("cost-summary: cannot read %s: %s", path, exc)
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("cost-summary: %s is not valid JSON: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("cost-summary: %s does not contain a JSON object", path)
        return None

    # Required fields for a recognizable panel-result file. Missing
    # ``total_cost`` or ``results`` => not the shape we know how to summarize.
    if "total_cost" not in data or "results" not in data:
        return None

    total_cost = _parse_cost_str(data.get("total_cost"))
    if total_cost is None:
        return None

    raw_results = data.get("results") or []
    if not isinstance(raw_results, list):
        return None

    top_level_model = str(data.get("model") or "")
    models_field = data.get("models")
    models_list: list[str]
    if isinstance(models_field, list):
        models_list = [str(m) for m in models_field if m]
    elif top_level_model:
        models_list = [top_level_model]
    else:
        models_list = []

    panelists = _extract_panelists(raw_results, top_level_model)

    persona_count = int(data.get("persona_count") or 0)
    question_count = int(data.get("question_count") or 0)

    # A run is treated as partial when fewer panelists landed than
    # ``persona_count`` advertises, or when every recorded panelist
    # produced no responses (cost gate / SIGINT mid-run).
    is_partial = False
    if (persona_count > 0 and len(panelists) < persona_count) or (panelists and all(p.turns == 0 for p in panelists)):
        is_partial = True

    return RunInfo(
        path=path,
        run_id=path.stem,
        created_at=_parse_iso_datetime(data.get("created_at")),
        model=top_level_model,
        models=models_list,
        total_cost_usd=float(total_cost),
        panelists=panelists,
        persona_count=persona_count,
        question_count=question_count,
        is_partial=is_partial,
    )


def _attribute_synthesis_residue(run: RunInfo) -> dict[str, float]:
    """Distribute ``total_cost`` across models, including synthesis residue.

    Per-panelist costs are summed by model. Any positive residue between
    ``total_cost`` and the panelist sum is attributed to the run's
    top-level ``model`` (typically the synthesis model). The residue
    represents synthesis overhead, which ``save_panel_result`` does not
    persist as a distinct field. Negative residues (panelists summing
    higher than total_cost — shouldn't happen, but tolerate floating
    point) are clamped to zero.
    """
    by_model: dict[str, float] = {}
    panelist_sum = 0.0
    for p in run.panelists:
        by_model[p.model] = by_model.get(p.model, 0.0) + p.cost_usd
        panelist_sum += p.cost_usd

    residue = run.total_cost_usd - panelist_sum
    if residue > 1e-9:
        # Synthesis residue lands on the top-level model. If the run had
        # no panelists at all (every persona errored), the whole total
        # still needs a bucket — fall through to ``(unknown)`` instead
        # of being silently dropped.
        bucket = run.model or "(unknown)"
        by_model[bucket] = by_model.get(bucket, 0.0) + residue
    return by_model


def aggregate_runs(
    runs: list[RunInfo],
    *,
    since: datetime | None = None,
) -> SummaryReport:
    """Aggregate per-run records into a ``SummaryReport``.

    Runs without a ``created_at`` are still counted in totals but
    excluded from ``by_month``. ``since`` filters runs whose
    ``created_at`` is strictly older; runs with no timestamp are
    excluded from a ``--since`` filter (we can't prove they qualify).
    """
    report = SummaryReport()

    for run in runs:
        if since is not None:
            if run.created_at is None:
                report.skipped_count += 1
                continue
            # Compare aware-vs-naive safely: if either side is naive,
            # drop tzinfo from the other to compare wall-clock dates.
            ran_at = run.created_at
            cmp_since = since
            if ran_at.tzinfo is None and cmp_since.tzinfo is not None:
                cmp_since = cmp_since.replace(tzinfo=None)
            elif ran_at.tzinfo is not None and cmp_since.tzinfo is None:
                ran_at = ran_at.replace(tzinfo=None)
            if ran_at < cmp_since:
                continue

        report.run_count += 1
        report.total_cost_usd += run.total_cost_usd
        if run.is_partial:
            report.partial_count += 1
        report.by_run.append(run)

        # Per-model attribution (with synthesis residue on the top-level model).
        run_by_model = _attribute_synthesis_residue(run)
        models_seen_in_run: set[str] = set()
        for model, cost in run_by_model.items():
            stat = report.by_model.setdefault(model, ModelStat())
            stat.cost_usd += cost
            models_seen_in_run.add(model)
        # Run count and turn count are accumulated once per (model, run)
        # pair so a single 5-panelist sonnet run shows ``runs=1`` not
        # ``runs=5``.
        for p in run.panelists:
            stat = report.by_model.setdefault(p.model, ModelStat())
            stat.turns += p.turns
        for model in models_seen_in_run:
            report.by_model[model].runs += 1

        # Per-month bucket
        if run.created_at is not None:
            month_key = run.created_at.strftime("%Y-%m")
            report.by_month[month_key] = report.by_month.get(month_key, 0.0) + run.total_cost_usd

    return report


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def format_text(
    report: SummaryReport,
    *,
    group_by: str = "model",
) -> str:
    """Render a human-readable text summary."""
    lines: list[str] = []
    if report.run_count == 0:
        lines.append("No runs found.")
        if report.skipped_count:
            lines.append(f"  ({report.skipped_count} skipped — no created_at)")
        return "\n".join(lines)

    partial_note = f" ({report.partial_count} partial)" if report.partial_count else ""
    lines.append(
        f"Total: ${report.total_cost_usd:.2f} across "
        f"{report.run_count} {_plural(report.run_count, 'run')}{partial_note}"
    )

    if group_by == "run":
        lines.append("By run:")
        # Newest first for human scanning.
        rows = sorted(
            report.by_run,
            key=lambda r: r.created_at.isoformat() if r.created_at else r.run_id,
            reverse=True,
        )
        for run in rows:
            tag = " (partial)" if run.is_partial else ""
            model_label = run.model or "(unknown)"
            cost_str = f"${run.total_cost_usd:.4f}"
            n_pan = len(run.panelists)
            lines.append(
                f"  {run.run_id:<36} {cost_str:>9}  ({model_label}, {n_pan} {_plural(n_pan, 'panelist')}){tag}"
            )
    else:
        lines.append("By model:")
        # Sort highest-spend first.
        for model, stat in sorted(report.by_model.items(), key=lambda kv: kv[1].cost_usd, reverse=True):
            cost_str = f"${stat.cost_usd:.4f}"
            lines.append(
                f"  {model:<32} {cost_str:>9}  "
                f"({stat.runs} {_plural(stat.runs, 'run')}, "
                f"{stat.turns} {_plural(stat.turns, 'turn')})"
            )

    if report.by_month:
        lines.append("By month:")
        for month in sorted(report.by_month.keys()):
            cost_str = f"${report.by_month[month]:.4f}"
            lines.append(f"  {month:<32} {cost_str:>9}")

    if report.skipped_count:
        lines.append(f"Skipped (no/unparseable timestamp): {report.skipped_count}")

    return "\n".join(lines)


def to_json_payload(
    report: SummaryReport,
    *,
    group_by: str = "model",
) -> dict[str, Any]:
    """Render the report as a plain JSON-friendly dict."""
    payload: dict[str, Any] = {
        "total_cost_usd": round(report.total_cost_usd, 6),
        "run_count": report.run_count,
        "partial_count": report.partial_count,
        "skipped_count": report.skipped_count,
        "by_model": {
            m: {
                "cost_usd": round(s.cost_usd, 6),
                "runs": s.runs,
                "turns": s.turns,
            }
            for m, s in sorted(report.by_model.items(), key=lambda kv: kv[1].cost_usd, reverse=True)
        },
        "by_month": {month: round(report.by_month[month], 6) for month in sorted(report.by_month.keys())},
    }
    if group_by == "run":
        payload["by_run"] = [
            {
                "run_id": r.run_id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "model": r.model,
                "models": r.models,
                "total_cost_usd": round(r.total_cost_usd, 6),
                "panelist_count": len(r.panelists),
                "turn_count": r.turn_count,
                "is_partial": r.is_partial,
            }
            for r in sorted(
                report.by_run,
                key=lambda r: r.created_at.isoformat() if r.created_at else r.run_id,
                reverse=True,
            )
        ]
    return payload


def summarize(
    runs_dir: Path | None = None,
    *,
    since: datetime | None = None,
) -> SummaryReport:
    """High-level convenience: discover, parse, aggregate."""
    rd = runs_dir if runs_dir is not None else default_runs_dir()
    parsed: list[RunInfo] = []
    for path in discover_run_files(rd):
        info = parse_run(path)
        if info is not None:
            parsed.append(info)
    return aggregate_runs(parsed, since=since)
