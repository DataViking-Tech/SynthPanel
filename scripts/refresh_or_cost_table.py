#!/usr/bin/env python3
"""Compare ``synth_panel.cost`` pricing against OpenRouter's live feed.

Pulls https://openrouter.ai/api/v1/models (no auth required) and
diffs the input/output/cache rates for every entry in
``EXPECTED_OR_MAPPING`` against what synthpanel's local table would
charge for the same model. Used by hq-xq36 to keep the cost-table
fresh when OR's blended rates shift.

Usage:

    python scripts/refresh_or_cost_table.py
    python scripts/refresh_or_cost_table.py --max-drift 0.10
    python scripts/refresh_or_cost_table.py --json > drift.json

Exits non-zero when any mapped model drifts by more than
``--max-drift`` (default 0.25) on input or output rates, so it can
be wired to a weekly CI cron and fail loudly on stale entries.

Cache rates are reported but not gated — providers commonly omit them
from the feed (returning $0) or report storage-vs-hit semantics that
do not map cleanly onto our flat per-million model.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass

from synth_panel.cost import ModelPricing, lookup_pricing

OR_MODELS_URL = "https://openrouter.ai/api/v1/models"

# OR canonical model id → expected-table-entry hint. Each id is one
# we want priced explicitly; the hint is purely for the human-facing
# diff report (``lookup_pricing`` does the actual resolution by
# substring match against the same key the runtime uses).
EXPECTED_OR_MAPPING: dict[str, str] = {
    "anthropic/claude-haiku-4.5": "HAIKU_PRICING",
    "anthropic/claude-sonnet-4.5": "SONNET_PRICING",
    # Current-generation Anthropic flagships. Opus was previously uncovered
    # here, so its stale Opus-3-era pricing could not self-heal via the drift
    # cron — the exact rot this check exists to prevent. ``claude-sonnet-5``
    # and ``claude-opus-4.8`` are what the ``sonnet``/``opus`` aliases now
    # resolve to; both are checked against the live OR feed so a future
    # price move (or a delisted id → NOT FOUND) fails loudly.
    "anthropic/claude-sonnet-5": "SONNET_PRICING",
    "anthropic/claude-opus-4.8": "OPUS_PRICING",
    "google/gemini-2.5-flash": "GEMINI_FLASH_PRICING",
    "google/gemini-2.5-flash-lite": "GEMINI_FLASH_LITE_PRICING",
    "google/gemini-2.5-pro": "GEMINI_PRO_PRICING",
    "openai/gpt-5-mini": "GPT_5_MINI_PRICING",
    "openai/gpt-4o-mini": "GPT_4O_MINI_PRICING",
    "openai/gpt-4o": "GPT_4O_PRICING",
    "openai/gpt-4.1-mini": "GPT_4_1_MINI_PRICING",
    "deepseek/deepseek-chat-v3.1": "DEEPSEEK_CHAT_PRICING",
    "deepseek/deepseek-v3.2": "DEEPSEEK_V3_2_PRICING",
    "deepseek/deepseek-v3.2-speciale": "DEEPSEEK_V3_2_SPECIALE_PRICING",
    "deepseek/deepseek-v3.2-exp": "DEEPSEEK_V3_2_EXP_PRICING",
    "qwen/qwen3.6-plus": "QWEN3_6_PLUS_PRICING",
    "qwen/qwen3-max": "QWEN3_MAX_PRICING",
    "mistralai/mistral-medium-3": "MISTRAL_MEDIUM_PRICING",
    "meta-llama/llama-3.3-70b-instruct": "LLAMA_3_3_70B_PRICING",
}


@dataclass
class Drift:
    model: str
    expected: str
    found: bool
    or_input: float
    or_output: float
    or_cache_write: float
    or_cache_read: float
    local: ModelPricing | None
    input_drift: float
    output_drift: float


def _fetch_models(url: str = OR_MODELS_URL) -> list[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "synthpanel/refresh_or_cost_table"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.load(resp)
    data = payload.get("data") or []
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected /v1/models response shape: {type(data)!r}")
    return data


def _per_million(value) -> float:
    try:
        return float(value or 0) * 1_000_000
    except (TypeError, ValueError):
        return 0.0


def _relative_drift(local: float, upstream: float) -> float:
    if upstream <= 0 and local <= 0:
        return 0.0
    return abs(upstream - local) / max(upstream, local)


def compute_drift(models: list[dict]) -> list[Drift]:
    by_id = {m.get("id"): m for m in models if isinstance(m, dict)}
    out: list[Drift] = []
    for model_id, expected in EXPECTED_OR_MAPPING.items():
        m = by_id.get(model_id)
        if m is None:
            out.append(
                Drift(
                    model=model_id,
                    expected=expected,
                    found=False,
                    or_input=0.0,
                    or_output=0.0,
                    or_cache_write=0.0,
                    or_cache_read=0.0,
                    local=None,
                    input_drift=0.0,
                    output_drift=0.0,
                )
            )
            continue
        pricing = m.get("pricing") or {}
        or_in = _per_million(pricing.get("prompt"))
        or_out = _per_million(pricing.get("completion"))
        or_cw = _per_million(pricing.get("input_cache_write"))
        or_cr = _per_million(pricing.get("input_cache_read"))
        # Resolve via the same path the runtime uses so we catch
        # substring-routing surprises (e.g. a new ``-mini`` suffix
        # falling through to a less-specific key).
        local, _ = lookup_pricing(f"openrouter/{model_id}")
        out.append(
            Drift(
                model=model_id,
                expected=expected,
                found=True,
                or_input=or_in,
                or_output=or_out,
                or_cache_write=or_cw,
                or_cache_read=or_cr,
                local=local,
                input_drift=_relative_drift(local.input_cost_per_million, or_in),
                output_drift=_relative_drift(local.output_cost_per_million, or_out),
            )
        )
    return out


def _format_text(drifts: list[Drift], max_drift: float) -> tuple[str, bool]:
    lines: list[str] = []
    failed = False
    lines.append(f"OpenRouter cost-table drift report (max_drift={max_drift:.0%})")
    lines.append("=" * 76)
    for d in drifts:
        if not d.found:
            lines.append(f"  ✗ {d.model:48s} NOT FOUND on OpenRouter")
            failed = True
            continue
        assert d.local is not None
        worst = max(d.input_drift, d.output_drift)
        marker = "✗" if worst > max_drift else "·"
        if worst > max_drift:
            failed = True
        lines.append(
            f"  {marker} {d.model:48s} "
            f"in: ${d.local.input_cost_per_million:.4f} → ${d.or_input:.4f} ({d.input_drift:+.1%})  "
            f"out: ${d.local.output_cost_per_million:.4f} → ${d.or_output:.4f} ({d.output_drift:+.1%})"
        )
    if failed:
        lines.append("")
        lines.append("FAIL: one or more models drifted beyond threshold or are missing.")
    else:
        lines.append("")
        lines.append("OK: all mapped models within tolerance.")
    return "\n".join(lines), failed


def _format_json(drifts: list[Drift], max_drift: float) -> tuple[str, bool]:
    failed = False
    payload = {"max_drift": max_drift, "models": []}
    for d in drifts:
        entry: dict = {
            "model": d.model,
            "expected_constant": d.expected,
            "found": d.found,
        }
        if d.found:
            assert d.local is not None
            entry.update(
                {
                    "or_input_per_million": d.or_input,
                    "or_output_per_million": d.or_output,
                    "or_cache_write_per_million": d.or_cache_write,
                    "or_cache_read_per_million": d.or_cache_read,
                    "local_input_per_million": d.local.input_cost_per_million,
                    "local_output_per_million": d.local.output_cost_per_million,
                    "local_cache_write_per_million": d.local.cache_creation_cost_per_million,
                    "local_cache_read_per_million": d.local.cache_read_cost_per_million,
                    "input_drift": d.input_drift,
                    "output_drift": d.output_drift,
                    "exceeds_threshold": max(d.input_drift, d.output_drift) > max_drift,
                }
            )
            if entry["exceeds_threshold"]:
                failed = True
        else:
            failed = True
        payload["models"].append(entry)
    payload["ok"] = not failed
    return json.dumps(payload, indent=2), failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-drift",
        type=float,
        default=0.25,
        help="Fail when input or output rate drift exceeds this fraction (default 0.25).",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--url", default=OR_MODELS_URL, help="Override the OR /v1/models URL.")
    args = parser.parse_args(argv)

    try:
        models = _fetch_models(args.url)
    except Exception as exc:
        print(f"error: failed to fetch {args.url}: {exc}", file=sys.stderr)
        return 2

    drifts = compute_drift(models)
    formatter = _format_json if args.json else _format_text
    text, failed = formatter(drifts, args.max_drift)
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
