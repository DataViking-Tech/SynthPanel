"""Cost and budget tracking for synthpanel.

Implements SPEC.md Section 7: token usage accumulation, model-specific
cost estimation, budget enforcement, and human-readable summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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

DEFAULT_PRICING = SONNET_PRICING

_PRICING_TABLE: dict[str, ModelPricing] = {
    "haiku": HAIKU_PRICING,
    "sonnet": SONNET_PRICING,
    "opus": OPUS_PRICING,
    "gemini-2.5-pro": GEMINI_PRO_PRICING,
    "gemini": GEMINI_FLASH_PRICING,
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
