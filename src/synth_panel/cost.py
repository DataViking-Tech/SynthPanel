"""Cost and budget tracking for synthpanel.

Implements SPEC.md Section 7: token usage accumulation, model-specific
cost estimation, budget enforcement, and human-readable summaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenUsage:
    """Immutable snapshot of token counts for a single LLM turn."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(self.cache_creation_input_tokens + other.cache_creation_input_tokens),
            cache_read_input_tokens=(self.cache_read_input_tokens + other.cache_read_input_tokens),
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TokenUsage:
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cache_creation_input_tokens=data.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=data.get("cache_read_input_tokens", 0),
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
# pricing snapshot_date: 2026-04-14
HAIKU_PRICING = ModelPricing(
    input_cost_per_million=1.00,
    output_cost_per_million=5.00,
    cache_creation_cost_per_million=1.25,
    cache_read_cost_per_million=0.10,
)

SONNET_PRICING = ModelPricing(
    input_cost_per_million=15.00,
    output_cost_per_million=75.00,
    cache_creation_cost_per_million=18.75,
    cache_read_cost_per_million=1.50,
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

DEFAULT_PRICING = SONNET_PRICING

# NOTE: substring match order matters. Put the most specific keys first so
# e.g. ``gpt-5-mini`` wins over any future shorter ``gpt-5`` key.
_PRICING_TABLE: dict[str, ModelPricing] = {
    "haiku": HAIKU_PRICING,
    "sonnet": SONNET_PRICING,
    "opus": OPUS_PRICING,
    "gemini-2.5-pro": GEMINI_PRO_PRICING,
    "gemini": GEMINI_FLASH_PRICING,
    "gpt-5-mini": GPT_5_MINI_PRICING,
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
        pricing, _ = lookup_pricing(m)
        per_cost[m] = estimate_cost(usage, pricing)
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
        pricing, is_estimated = lookup_pricing(model)
        cost = estimate_cost(self._cumulative, pricing)
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
