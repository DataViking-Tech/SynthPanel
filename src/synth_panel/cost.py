"""Cost and budget tracking for synthpanel.

Implements SPEC.md Section 7: token usage accumulation, model-specific
cost estimation, budget enforcement, and human-readable summaries.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Immutable snapshot of token counts for a single LLM turn.

    ``provider_reported_cost`` is the authoritative USD cost as billed by
    the upstream provider (e.g. OpenRouter's ``usage.cost``). When present,
    ``resolve_cost`` prefers it over the local pricing table; the local
    table becomes a divergence sanity-check rather than a billing source.

    ``reasoning_tokens`` / ``cached_tokens`` are informational sub-counts
    already included in ``output_tokens`` / ``input_tokens`` respectively.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    provider_reported_cost: float | None = None
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        if self.provider_reported_cost is None and other.provider_reported_cost is None:
            summed_cost: float | None = None
        else:
            summed_cost = (self.provider_reported_cost or 0.0) + (other.provider_reported_cost or 0.0)
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(self.cache_creation_input_tokens + other.cache_creation_input_tokens),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
            provider_reported_cost=summed_cost,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }
        if self.provider_reported_cost is not None:
            d["provider_reported_cost"] = self.provider_reported_cost
        if self.reasoning_tokens:
            d["reasoning_tokens"] = self.reasoning_tokens
        if self.cached_tokens:
            d["cached_tokens"] = self.cached_tokens
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TokenUsage:
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
            provider_reported_cost=data.get("provider_reported_cost"),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
        )


ZERO_USAGE = TokenUsage()


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token pricing rates in USD."""

    input_cost_per_million: float
    output_cost_per_million: float
    cache_creation_cost_per_million: float
    cache_read_cost_per_million: float


# Known pricing tiers from SPEC.md §7.
# pricing snapshot_date: 2026-04-21
HAIKU_PRICING = ModelPricing(
    input_cost_per_million=1.00,
    output_cost_per_million=5.00,
    cache_creation_cost_per_million=1.25,
    cache_read_cost_per_million=0.10,
)

# Sonnet 4.5 published rates (per Anthropic price list): $3/M in, $15/M out,
# $3.75/M 5-min cache write, $0.30/M cache read. Prior to sp-cxyb this table
# carried legacy Opus-3 rates ($15/$75/$18.75/$1.50), which also leaked into
# DEFAULT_PRICING and over-billed every unknown model by ~5x.
SONNET_PRICING = ModelPricing(
    input_cost_per_million=3.00,
    output_cost_per_million=15.00,
    cache_creation_cost_per_million=3.75,
    cache_read_cost_per_million=0.30,
)

OPUS_PRICING = ModelPricing(
    input_cost_per_million=15.00,
    output_cost_per_million=75.00,
    cache_creation_cost_per_million=18.75,
    cache_read_cost_per_million=1.50,
)

GEMINI_FLASH_PRICING = ModelPricing(
    input_cost_per_million=0.15,
    output_cost_per_million=0.60,
    cache_creation_cost_per_million=0.19,
    cache_read_cost_per_million=0.04,
)

# Gemini 2.5 Flash-Lite (OpenRouter: ``google/gemini-2.5-flash-lite``).
# Published Google rates: $0.10/M input, $0.40/M output, $0.025/M cache read.
# cache_creation approximated at input rate (Google bills cache writes close to
# input rate). sp-9gcm: previously fell through the ``gemini`` substring to
# GEMINI_FLASH_PRICING, overstating the local-table estimate by ~41%.
GEMINI_FLASH_LITE_PRICING = ModelPricing(
    input_cost_per_million=0.10,
    output_cost_per_million=0.40,
    cache_creation_cost_per_million=0.10,
    cache_read_cost_per_million=0.025,
)

GEMINI_PRO_PRICING = ModelPricing(
    input_cost_per_million=1.25,
    output_cost_per_million=10.00,
    cache_creation_cost_per_million=1.56,
    cache_read_cost_per_million=0.31,
)

# OpenAI gpt-5-mini (also served via OpenRouter as ``openai/gpt-5-mini``).
# Published rates: $0.25/M input, $2.00/M output, $0.025/M cached input.
# cache_creation approximated at input rate (OpenAI/OpenRouter bill cache
# writes as regular input); cache_read matches the published cached rate.
GPT_5_MINI_PRICING = ModelPricing(
    input_cost_per_million=0.25,
    output_cost_per_million=2.00,
    cache_creation_cost_per_million=0.25,
    cache_read_cost_per_million=0.025,
)

# OpenAI gpt-4o-mini (OpenRouter: ``openai/gpt-4o-mini``).
# Published rates: $0.15/M input, $0.60/M output, $0.075/M cached input.
GPT_4O_MINI_PRICING = ModelPricing(
    input_cost_per_million=0.15,
    output_cost_per_million=0.60,
    cache_creation_cost_per_million=0.15,
    cache_read_cost_per_million=0.075,
)

# OpenAI gpt-4o (OpenRouter: ``openai/gpt-4o``).
# Published rates: $2.50/M input, $10.00/M output, $1.25/M cached input.
GPT_4O_PRICING = ModelPricing(
    input_cost_per_million=2.50,
    output_cost_per_million=10.00,
    cache_creation_cost_per_million=2.50,
    cache_read_cost_per_million=1.25,
)

# OpenAI gpt-4.1-mini (OpenRouter: ``openai/gpt-4.1-mini``).
# Published rates: $0.40/M input, $1.60/M output, $0.10/M cached input.
GPT_4_1_MINI_PRICING = ModelPricing(
    input_cost_per_million=0.40,
    output_cost_per_million=1.60,
    cache_creation_cost_per_million=0.40,
    cache_read_cost_per_million=0.10,
)

# DeepSeek Chat V3 (OpenRouter: ``deepseek/deepseek-chat-v3``, ``deepseek-chat-v3.1``).
# OpenRouter rates: $0.27/M input, $1.10/M output, $0.07/M cache read.
# Covers the v3 family via the ``deepseek-chat`` substring; R1 is priced separately
# (not included here — different product tier).
DEEPSEEK_CHAT_PRICING = ModelPricing(
    input_cost_per_million=0.27,
    output_cost_per_million=1.10,
    cache_creation_cost_per_million=0.27,
    cache_read_cost_per_million=0.07,
)

# DeepSeek V3.2 (OpenRouter: ``deepseek/deepseek-v3.2``). Successor family to
# deepseek-chat-v3; OpenRouter drops the ``-chat-`` infix, so the older
# ``deepseek-chat`` substring no longer matches.
# Rates: $0.252/M input, $0.378/M output. Cache not published — input rate.
DEEPSEEK_V3_2_PRICING = ModelPricing(
    input_cost_per_million=0.252,
    output_cost_per_million=0.378,
    cache_creation_cost_per_million=0.252,
    cache_read_cost_per_million=0.252,
)

# DeepSeek V3.2 Speciale (OpenRouter: ``deepseek/deepseek-v3.2-speciale``).
# Higher-tier sibling of v3.2. Rates: $0.40/M input, $1.20/M output.
DEEPSEEK_V3_2_SPECIALE_PRICING = ModelPricing(
    input_cost_per_million=0.40,
    output_cost_per_million=1.20,
    cache_creation_cost_per_million=0.40,
    cache_read_cost_per_million=0.40,
)

# DeepSeek V3.2 Experimental (OpenRouter: ``deepseek/deepseek-v3.2-exp``).
# Rates: $0.27/M input, $0.41/M output.
DEEPSEEK_V3_2_EXP_PRICING = ModelPricing(
    input_cost_per_million=0.27,
    output_cost_per_million=0.41,
    cache_creation_cost_per_million=0.27,
    cache_read_cost_per_million=0.27,
)

# Qwen3 Plus (OpenRouter: ``qwen/qwen3-plus``, Alibaba DashScope rate).
# Published rates: $0.33/M input, $1.95/M output. Cache rates not published —
# defaulted to input rate (no-savings baseline).
QWEN3_PLUS_PRICING = ModelPricing(
    input_cost_per_million=0.33,
    output_cost_per_million=1.95,
    cache_creation_cost_per_million=0.33,
    cache_read_cost_per_million=0.33,
)

# Qwen3.6 Plus (OpenRouter: ``qwen/qwen3.6-plus``). Minor-version bump on
# qwen3-plus; the ``.6`` infix breaks the old ``qwen3-plus`` substring so an
# explicit key is required. Same published rates as qwen3-plus.
QWEN3_6_PLUS_PRICING = ModelPricing(
    input_cost_per_million=0.33,
    output_cost_per_million=1.95,
    cache_creation_cost_per_million=0.33,
    cache_read_cost_per_million=0.33,
)

# Qwen3 Max (OpenRouter: ``qwen/qwen3-max``). Top-tier sibling of qwen3-plus.
# Published rates: $0.78/M input, $3.90/M output.
QWEN3_MAX_PRICING = ModelPricing(
    input_cost_per_million=0.78,
    output_cost_per_million=3.90,
    cache_creation_cost_per_million=0.78,
    cache_read_cost_per_million=0.78,
)

# Mistral Medium 3 (OpenRouter: ``mistralai/mistral-medium-3``).
# Published rates: $0.40/M input, $2.00/M output. Cache rates not published.
MISTRAL_MEDIUM_PRICING = ModelPricing(
    input_cost_per_million=0.40,
    output_cost_per_million=2.00,
    cache_creation_cost_per_million=0.40,
    cache_read_cost_per_million=0.40,
)

# Meta Llama 3.3 70B Instruct (OpenRouter: ``meta-llama/llama-3.3-70b-instruct``).
# Typical OpenRouter rate: $0.23/M input, $0.40/M output. Cache rates not published.
LLAMA_3_3_70B_PRICING = ModelPricing(
    input_cost_per_million=0.23,
    output_cost_per_million=0.40,
    cache_creation_cost_per_million=0.23,
    cache_read_cost_per_million=0.23,
)

DEFAULT_PRICING = SONNET_PRICING

# NOTE: substring match order matters. Put the most specific keys first so
# e.g. ``gpt-4o-mini`` wins over the shorter ``gpt-4o`` key.
_PRICING_TABLE: dict[str, ModelPricing] = {
    "haiku": HAIKU_PRICING,
    "sonnet": SONNET_PRICING,
    "opus": OPUS_PRICING,
    "gemini-2.5-pro": GEMINI_PRO_PRICING,
    # sp-9gcm: ``flash-lite`` (distinctive substring) must precede the bare
    # ``gemini`` key so Lite models don't inherit full-flash rates (~41%
    # overstatement). Matches both bare ``gemini-flash-lite`` and the
    # OpenRouter form ``gemini-2.5-flash-lite``.
    "flash-lite": GEMINI_FLASH_LITE_PRICING,
    "gemini": GEMINI_FLASH_PRICING,
    "gpt-5-mini": GPT_5_MINI_PRICING,
    "gpt-4.1-mini": GPT_4_1_MINI_PRICING,
    "gpt-4o-mini": GPT_4O_MINI_PRICING,
    "gpt-4o": GPT_4O_PRICING,
    # DeepSeek entries — specific variant suffixes must precede the bare
    # ``deepseek-v3.2`` key so ``-speciale`` / ``-exp`` don't get swallowed
    # by the shorter match.
    "deepseek-v3.2-speciale": DEEPSEEK_V3_2_SPECIALE_PRICING,
    "deepseek-v3.2-exp": DEEPSEEK_V3_2_EXP_PRICING,
    "deepseek-v3.2": DEEPSEEK_V3_2_PRICING,
    # ``deepseek-v3`` (sp-9gcm): OpenRouter's short route that resolves to the
    # v3.2 family. Placed after v3.2-* keys so the specific variants still win
    # when present; catches bare ``deepseek-v3`` and any future ``deepseek-v3.x``
    # before they fall through to DEFAULT_PRICING.
    "deepseek-v3": DEEPSEEK_V3_2_PRICING,
    "deepseek-chat": DEEPSEEK_CHAT_PRICING,
    # Qwen entries — ``qwen3.6-plus`` before ``qwen3-plus`` (the former
    # contains the latter only by coincidence, but keep specific-first
    # ordering consistent).
    "qwen3.6-plus": QWEN3_6_PLUS_PRICING,
    "qwen3-max": QWEN3_MAX_PRICING,
    "qwen3-plus": QWEN3_PLUS_PRICING,
    "mistral-medium": MISTRAL_MEDIUM_PRICING,
    "llama-3.3-70b": LLAMA_3_3_70B_PRICING,
}


def lookup_pricing(model: str | None = None) -> tuple[ModelPricing, bool]:
    """Return (pricing, is_estimated) for *model*.

    Checks whether any key in ``_PRICING_TABLE`` appears as a substring of
    the canonical model name.  Falls back to ``DEFAULT_PRICING`` with
    ``is_estimated=True`` when no tier matches.
    """
    if model:
        lower = model.lower()
        for key, pricing in _PRICING_TABLE.items():
            if key in lower:
                return pricing, False
    return DEFAULT_PRICING, True


# sp-nn8k: template used when a model's cost was priced via ``DEFAULT_PRICING``
# fallback. Surfaced into panel output ``warnings[]`` so downstream consumers
# can tell an estimated $X.XX apart from a billed rate, and we also set a
# top-level ``cost_is_estimated: True`` boolean for programmatic gating.
_COST_FALLBACK_WARNING_TEMPLATE = (
    "Cost for model {model!r} computed using DEFAULT_PRICING fallback — real "
    "charges may differ. Consider adding an explicit pricing entry."
)


def collect_estimated_models(models: Iterable[str | None]) -> list[str]:
    """Return the subset of *models* whose pricing falls back to DEFAULT_PRICING.

    Preserves first-seen order and deduplicates. ``None``/empty entries are
    skipped so callers can splat together ``panelist_model``,
    ``synthesis_model``, and per-panelist model keys without pre-filtering.
    """
    seen: set[str] = set()
    estimated: list[str] = []
    for m in models:
        if not m or m in seen:
            continue
        seen.add(m)
        _, is_estimated = lookup_pricing(m)
        if is_estimated:
            estimated.append(m)
    return estimated


def build_cost_fallback_warnings(models: Iterable[str | None]) -> list[str]:
    """Return one warning string per model that was priced via DEFAULT_PRICING.

    Thin wrapper over :func:`collect_estimated_models` that formats each
    offender into the canonical ``"Cost for model 'X' computed using
    DEFAULT_PRICING fallback ..."`` sentence. Empty when every contributing
    model has an explicit pricing tier.
    """
    return [_COST_FALLBACK_WARNING_TEMPLATE.format(model=m) for m in collect_estimated_models(models)]


# Bucket prefixes observed in synthbench `config.provider` strings
# (see synthbench specs/cost-metrics/research/synthbench-runner.md Q5).
_PROVIDER_BUCKETS = ("synthpanel", "openrouter", "raw-anthropic", "raw-openai", "raw-gemini", "ollama")

# Strip a leading bucket prefix and any trailing " t=...", " profile=...", " tpl=..."
# decorators so the remainder is a plain canonical model string suitable for
# ``lookup_pricing``.
_PROVIDER_SPLIT_RE = re.compile(
    r"^(?P<bucket>" + "|".join(re.escape(b) for b in _PROVIDER_BUCKETS) + r")/(?P<inner>.+?)"
    r"(?:\s+(?:t|profile|tpl)=\S+)*\s*$"
)

# Provider strings that intentionally have no priced equivalent: synthetic
# baselines (zero-cost references) and ensemble blends (cost is the sum of
# constituents, computed elsewhere). Callers decide whether to emit null or 0.
_UNPRICED_PROVIDERS = frozenset(
    {
        "random-baseline",
        "majority-baseline",
        "population-average-baseline",
    }
)


def lookup_pricing_by_provider(provider_string: str) -> tuple[ModelPricing | None, bool]:
    """Resolve a synthbench ``config.provider`` string to pricing.

    Parses the bucket prefix (``synthpanel/``, ``openrouter/``,
    ``raw-(anthropic|openai|gemini)/``, ``ollama/``) and trailing
    ``" t=..."``, ``" profile=..."``, ``" tpl=..."`` decorators, then
    delegates to ``lookup_pricing`` on the inner model string.

    Returns ``(pricing, is_estimated)`` matching ``lookup_pricing``'s
    return shape, except: returns ``(None, False)`` for unresolved
    providers — ``ollama/*`` (self-hosted), the named baselines, any
    ``ensemble/*`` provider, and any inner string that ``lookup_pricing``
    can only price via the substring-fallback. Refusing the fallback is
    intentional: callers (e.g. synthbench publish) decide whether to emit
    null or a default rather than silently billing at Sonnet rates.
    """
    if not provider_string:
        return None, False

    stripped = provider_string.strip()

    if stripped in _UNPRICED_PROVIDERS or stripped.startswith("ensemble/"):
        return None, False

    match = _PROVIDER_SPLIT_RE.match(stripped)
    if not match:
        return None, False

    bucket = match.group("bucket")
    inner = match.group("inner").strip()

    if bucket == "ollama":
        return None, False

    pricing, is_estimated = lookup_pricing(inner)
    if is_estimated:
        return None, False
    return pricing, False


@dataclass(frozen=True)
class CostEstimate:
    """Breakdown of estimated USD costs for a token usage snapshot."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    cache_creation_cost: float = 0.0
    cache_read_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost + self.cache_creation_cost + self.cache_read_cost

    def __add__(self, other: CostEstimate) -> CostEstimate:
        return CostEstimate(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            cache_creation_cost=(self.cache_creation_cost + other.cache_creation_cost),
            cache_read_cost=self.cache_read_cost + other.cache_read_cost,
        )

    def format_usd(self) -> str:
        return f"${self.total_cost:.4f}"


def estimate_cost(
    usage: TokenUsage,
    pricing: ModelPricing | None = None,
) -> CostEstimate:
    """Convert a *usage* snapshot to a ``CostEstimate`` using *pricing*."""
    p = pricing or DEFAULT_PRICING
    return CostEstimate(
        input_cost=usage.input_tokens * p.input_cost_per_million / 1_000_000,
        output_cost=usage.output_tokens * p.output_cost_per_million / 1_000_000,
        cache_creation_cost=(usage.cache_creation_input_tokens * p.cache_creation_cost_per_million / 1_000_000),
        cache_read_cost=(usage.cache_read_input_tokens * p.cache_read_cost_per_million / 1_000_000),
    )


# Local-vs-provider divergence threshold for the sanity-check warning.
# Below this ratio we assume the local pricing table is still calibrated;
# above it we flag a likely stale or mis-keyed entry.
_DIVERGENCE_WARN_RATIO = 0.20


def resolve_cost(
    usage: TokenUsage,
    model: str | None = None,
) -> CostEstimate:
    """Return the authoritative ``CostEstimate`` for *usage*.

    Precedence:

    1. If ``usage.provider_reported_cost`` is set (e.g. OpenRouter's
       ``usage.cost``), use it verbatim — this is what was actually
       billed and already reflects BYOK discounts, upstream pricing
       changes, and per-token promos the local table cannot track. The
       amount is deposited on ``CostEstimate.output_cost`` so totals are
       preserved; input/cache buckets are left at zero because the
       provider does not always return a breakdown.
    2. Otherwise fall back to the local pricing table via ``lookup_pricing``.

    When both the provider value and a local estimate are available and
    they diverge by more than ``_DIVERGENCE_WARN_RATIO`` (20%), a
    warning is logged so a stale pricing table can be caught during
    development. The warning is informational only — the provider value
    is always returned.
    """
    pricing, _is_estimated = lookup_pricing(model)
    local = estimate_cost(usage, pricing)

    if usage.provider_reported_cost is None:
        return local

    provider_total = float(usage.provider_reported_cost)
    local_total = local.total_cost

    if local_total > 0 and provider_total > 0:
        ratio = abs(provider_total - local_total) / max(provider_total, local_total)
        if ratio > _DIVERGENCE_WARN_RATIO:
            logger.warning(
                "Local cost estimate diverges from provider-reported for model=%s: "
                "local=$%.6f provider=$%.6f ratio=%.1f%% — pricing table may be stale.",
                model,
                local_total,
                provider_total,
                ratio * 100.0,
            )

    return CostEstimate(
        input_cost=0.0,
        output_cost=provider_total,
        cache_creation_cost=0.0,
        cache_read_cost=0.0,
    )


def aggregate_per_model(
    panelist_results,  # Iterable[PanelistResult]; untyped to avoid circular import
    default_model: str,
) -> tuple[dict[str, TokenUsage], dict[str, CostEstimate]]:
    """Bucket panelist usage/cost by each result's ``.model`` attribute.

    Each panelist's ``.model`` (falling back to *default_model* when unset)
    is used to look up that provider's pricing, so multi-model runs
    (``--models haiku:0.5,gemini:0.5`` routing, ``ensemble_run``, or
    ``--blend`` mode) get per-model cost that reflects the actual rate
    the provider charged — not a uniform rate applied across all tokens.

    Returns ``(per_model_usage, per_model_cost)``. Keys are the model
    identifiers as recorded on panelist results (alias resolution happens
    later in ``build_metadata`` when producing the public payload).
    """
    per_usage: dict[str, TokenUsage] = {}
    for pr in panelist_results:
        m = getattr(pr, "model", None) or default_model
        per_usage[m] = per_usage.get(m, ZERO_USAGE) + pr.usage

    per_cost: dict[str, CostEstimate] = {}
    for m, usage in per_usage.items():
        per_cost[m] = resolve_cost(usage, m)
    return per_usage, per_cost


def format_summary(
    label: str,
    usage: TokenUsage,
    cost: CostEstimate,
    *,
    model: str | None = None,
    is_estimated: bool = False,
) -> str:
    """Produce the two-line summary described in SPEC.md §7."""
    parts = [
        f"{label}:",
        f"total_tokens={usage.total_tokens}",
        f"input={usage.input_tokens}",
        f"output={usage.output_tokens}",
        f"cache_write={usage.cache_creation_input_tokens}",
        f"cache_read={usage.cache_read_input_tokens}",
        f"estimated_cost={cost.format_usd()}",
    ]
    if model:
        parts.append(f"model={model}")
    if is_estimated:
        parts.append("pricing=estimated-default")
    line1 = " ".join(parts)
    line2 = (
        f"  cost breakdown:"
        f" input=${cost.input_cost:.4f}"
        f" output=${cost.output_cost:.4f}"
        f" cache_write=${cost.cache_creation_cost:.4f}"
        f" cache_read=${cost.cache_read_cost:.4f}"
    )
    return f"{line1}\n{line2}"


class UsageTracker:
    """Accumulates token usage across multiple turns.

    Thread-safety is *not* provided — callers must synchronise externally
    if the tracker is shared.  (The spec notes that this is sufficient for
    the synthpanel use case.)
    """

    def __init__(self) -> None:
        self._turns: list[TokenUsage] = []
        self._cumulative: TokenUsage = ZERO_USAGE

    # --- Recording -------------------------------------------------------

    def record_turn(self, usage: TokenUsage) -> None:
        """Append a single turn's usage and update the running total."""
        self._turns.append(usage)
        self._cumulative = self._cumulative + usage
        logger.debug(
            "usage turn %d: in=%d out=%d cumulative=%d",
            len(self._turns),
            usage.input_tokens,
            usage.output_tokens,
            self._cumulative.total_tokens,
        )

    # --- Queries ---------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def current_turn_usage(self) -> TokenUsage:
        """Return the most recent turn's usage, or ``ZERO_USAGE``."""
        if self._turns:
            return self._turns[-1]
        return ZERO_USAGE

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._cumulative

    def get_turn(self, index: int) -> TokenUsage:
        return self._turns[index]

    # --- Reconstruction --------------------------------------------------

    @classmethod
    def from_usages(cls, usages: list[TokenUsage]) -> UsageTracker:
        """Reconstruct a tracker by replaying a list of per-turn usages."""
        tracker = cls()
        for u in usages:
            tracker.record_turn(u)
        return tracker

    # --- Summaries -------------------------------------------------------

    def summarise(
        self,
        label: str = "Cumulative",
        *,
        model: str | None = None,
    ) -> str:
        _, is_estimated = lookup_pricing(model)
        cost = resolve_cost(self._cumulative, model)
        # When the provider billed this usage directly, the ``estimated``
        # tag from the local table is misleading — provider-reported cost
        # is always authoritative.
        if self._cumulative.provider_reported_cost is not None:
            is_estimated = False
        return format_summary(
            label,
            self._cumulative,
            cost,
            model=model,
            is_estimated=is_estimated,
        )


class BudgetError(Exception):
    """Raised when a budget limit would be exceeded."""

    def __init__(self, budget: int, projected: int) -> None:
        self.budget = budget
        self.projected = projected
        super().__init__(f"Budget exceeded: projected {projected} tokens > budget {budget}")


@dataclass
class BudgetGate:
    """Enforces a maximum token budget.

    The gate is checked *before* a turn begins.  If the cumulative usage
    plus a projected turn size would exceed ``max_tokens``, a
    ``BudgetError`` is raised.

    ``max_tokens`` defaults to 2000 as specified in SPEC.md §7 for
    lightweight / query-engine use.
    """

    max_tokens: int = 2000
    _tracker: UsageTracker = field(default_factory=UsageTracker)

    @property
    def tracker(self) -> UsageTracker:
        return self._tracker

    def record_turn(self, usage: TokenUsage) -> None:
        self._tracker.record_turn(usage)

    def check(self, projected_tokens: int = 0) -> None:
        """Raise ``BudgetError`` if the budget would be exceeded."""
        current = self._tracker.cumulative_usage.total_tokens
        if current + projected_tokens > self.max_tokens:
            raise BudgetError(self.max_tokens, current + projected_tokens)

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self._tracker.cumulative_usage.total_tokens)
