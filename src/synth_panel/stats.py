"""Pure-Python statistical tests for synthetic panel analysis.

All functions use only ``math``, ``random``, and ``dataclasses`` from stdlib.
No scipy or numpy required.

Implements: bootstrap BCa confidence intervals, chi-squared goodness-of-fit,
Kendall's W concordance, frequency tables, and Borda count ranking.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "BootstrapResult",
    "bootstrap_ci",
    "proportion_stat",
    "ChiSquaredResult",
    "chi_squared_test",
    "KendallWResult",
    "kendall_w",
    "FrequencyRow",
    "FrequencyTable",
    "frequency_table",
    "BordaResult",
    "borda_count",
]

# ---------------------------------------------------------------------------
# Internal helpers: normal CDF / inverse CDF / chi-squared survival
# ---------------------------------------------------------------------------


def _ndtr(x: float) -> float:
    """Standard normal CDF. Uses ``math.erfc``; max error < 7.5e-8."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF (percent-point function).

    Beasley-Springer-Moro rational approximation.
    Max error < 4.5e-4 for p in (0.0001, 0.9999).
    """
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0

    # Coefficients for the central region approximation
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail
        q = math.sqrt(-2.0 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    else:
        # Upper tail
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1 - CDF) of chi-squared distribution.

    Uses the regularized upper incomplete gamma function via series expansion.
    Adequate for df <= 200 and x <= 1000.
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 0.0

    a = df / 2.0
    z = x / 2.0

    # Regularized lower incomplete gamma P(a, z) via series expansion
    # P(a, z) = e^{-z} * z^a * sum_{n=0..inf} z^n / Gamma(a + n + 1)
    log_prefix = -z + a * math.log(z) - math.lgamma(a + 1)

    series_sum = 1.0
    term = 1.0
    for n in range(1, 300):
        term *= z / (a + n)
        series_sum += term
        if abs(term) < 1e-15 * abs(series_sum):
            break

    p_lower = math.exp(log_prefix) * series_sum
    # Clamp to [0, 1] for numerical safety
    p_lower = max(0.0, min(1.0, p_lower))
    return 1.0 - p_lower


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a bootstrap confidence interval computation."""

    estimate: float
    ci_lower: float
    ci_upper: float
    confidence: float
    n_resamples: int
    method: str = "BCa"


def proportion_stat(value) -> callable:
    """Return a stat function that computes the proportion of *value* in data."""

    def _fn(data: list) -> float:
        return sum(1 for x in data if x == value) / len(data)

    return _fn


def bootstrap_ci(
    data: list[float],
    stat_fn: callable,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int | None = None,
) -> BootstrapResult:
    """Compute a BCa bootstrap confidence interval.

    Args:
        data: Observed data points. Minimum length 5.
        stat_fn: Statistic function ``list[float] -> float``.
        confidence: Confidence level in (0, 1). Default 0.95.
        n_resamples: Number of bootstrap resamples. Default 2000.
        seed: RNG seed for reproducibility.

    Returns:
        BootstrapResult with point estimate, CI bounds, metadata.

    Raises:
        ValueError: If ``len(data) < 5`` or confidence not in (0, 1).
    """
    n = len(data)
    if n < 5:
        raise ValueError(f"data must have at least 5 elements, got {n}")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = random.Random(seed)

    # 1. Point estimate
    theta_hat = stat_fn(data)

    # 2-3. Bootstrap resamples
    theta_star = []
    for _ in range(n_resamples):
        resample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        theta_star.append(stat_fn(resample))

    # 4. Bias correction z0
    count_below = sum(1 for t in theta_star if t < theta_hat)
    prop_below = count_below / n_resamples
    # Clamp to avoid _ndtri(0) or _ndtri(1)
    prop_below = max(1 / (n_resamples + 1), min(n_resamples / (n_resamples + 1), prop_below))
    z0 = _ndtri(prop_below)

    # 5. Acceleration via jackknife
    jackknife_vals = []
    for i in range(n):
        jack_sample = data[:i] + data[i + 1 :]
        jackknife_vals.append(stat_fn(jack_sample))

    theta_dot = sum(jackknife_vals) / n
    d = [theta_dot - jv for jv in jackknife_vals]
    sum_d2 = sum(di * di for di in d)
    sum_d3 = sum(di * di * di for di in d)

    denom = 6.0 * sum_d2**1.5
    a_hat = sum_d3 / denom if denom != 0 else 0.0

    # 6. Adjusted percentiles
    alpha = 1.0 - confidence
    z_alpha_lo = _ndtri(alpha / 2.0)
    z_alpha_hi = _ndtri(1.0 - alpha / 2.0)

    def _adj(z_a: float) -> float:
        num = z0 + z_a
        denom_val = 1.0 - a_hat * num
        if denom_val == 0:
            return 0.5
        return _ndtr(z0 + num / denom_val)

    a1 = _adj(z_alpha_lo)
    a2 = _adj(z_alpha_hi)

    # Clamp
    lo_clamp = 1.0 / (n_resamples + 1)
    hi_clamp = n_resamples / (n_resamples + 1)
    a1 = max(lo_clamp, min(hi_clamp, a1))
    a2 = max(lo_clamp, min(hi_clamp, a2))

    # 7. Extract CI from sorted bootstrap distribution
    theta_star.sort()
    ci_lower = theta_star[int(math.floor(a1 * n_resamples))]
    ci_upper = theta_star[int(math.floor(a2 * n_resamples))]

    return BootstrapResult(
        estimate=theta_hat,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        n_resamples=n_resamples,
    )


# ---------------------------------------------------------------------------
# chi_squared_test
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChiSquaredResult:
    """Result of a chi-squared goodness-of-fit test."""

    statistic: float
    df: int
    p_value: float
    expected: dict[str, float]
    observed: dict[str, int]
    cramers_v: float
    warning: str | None


def chi_squared_test(
    observed: dict[str, int],
    expected: dict[str, float] | None = None,
) -> ChiSquaredResult:
    """Chi-squared goodness-of-fit test.

    Args:
        observed: Mapping of category -> count.
        expected: Expected counts per category. If None, assumes uniform.

    Returns:
        ChiSquaredResult with statistic, df, p_value, effect size.

    Raises:
        ValueError: If observed is empty, any count is negative,
                    or expected keys don't match observed keys.
    """
    if not observed:
        raise ValueError("observed must not be empty")

    for k, v in observed.items():
        if v < 0:
            raise ValueError(f"observed count for {k!r} is negative: {v}")

    k_cats = len(observed)
    total = sum(observed.values())

    if expected is None:
        exp_val = total / k_cats
        expected = {k: exp_val for k in observed}
    else:
        if set(expected.keys()) != set(observed.keys()):
            raise ValueError(
                f"expected keys {set(expected.keys())} don't match "
                f"observed keys {set(observed.keys())}"
            )

    # Chi-squared statistic
    chi2 = sum((observed[k] - expected[k]) ** 2 / expected[k] for k in observed)

    df = k_cats - 1
    p_value = _chi2_sf(chi2, df) if df > 0 else 1.0

    # Cramer's V for GOF (single-row): V = sqrt(chi2 / (N * (K - 1)))
    if total > 0 and df > 0:
        v = math.sqrt(chi2 / (total * df))
    else:
        v = 0.0

    # Warning for small expected counts
    min_expected = min(expected.values())
    warning = None
    if min_expected < 5:
        warning = (
            f"Some expected count(s) < 5 (min={min_expected:.1f}). "
            "Chi-squared approximation may be unreliable."
        )

    return ChiSquaredResult(
        statistic=chi2,
        df=df,
        p_value=p_value,
        expected=dict(expected),
        observed=dict(observed),
        cramers_v=v,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# kendall_w
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KendallWResult:
    """Result of Kendall's W concordance test."""

    w: float
    chi_squared: float
    df: int
    p_value: float
    n_raters: int
    n_items: int


def kendall_w(
    rankings: list[list[int]],
) -> KendallWResult:
    """Kendall's W coefficient of concordance.

    Args:
        rankings: List of N rankings. Each is a list of K integers (ranks 1..K).

    Returns:
        KendallWResult with W, chi-squared approximation, p-value.

    Raises:
        ValueError: If rankings is empty, lengths differ, or ranks invalid.
    """
    if not rankings:
        raise ValueError("rankings must not be empty")

    n_raters = len(rankings)
    k_items = len(rankings[0])

    for i, r in enumerate(rankings):
        if len(r) != k_items:
            raise ValueError(
                f"ranking {i} has length {len(r)}, expected {k_items}"
            )

    if k_items < 2:
        raise ValueError("need at least 2 items to compute concordance")

    # R_j = sum of ranks for item j across all raters
    r_sums = [0.0] * k_items
    for ranking in rankings:
        for j in range(k_items):
            r_sums[j] += ranking[j]

    r_bar = n_raters * (k_items + 1) / 2.0
    s = sum((rj - r_bar) ** 2 for rj in r_sums)

    w = 12.0 * s / (n_raters**2 * (k_items**3 - k_items))

    chi2 = n_raters * (k_items - 1) * w
    df = k_items - 1
    p_value = _chi2_sf(chi2, df)

    return KendallWResult(
        w=w,
        chi_squared=chi2,
        df=df,
        p_value=p_value,
        n_raters=n_raters,
        n_items=k_items,
    )


# ---------------------------------------------------------------------------
# frequency_table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyRow:
    """One row in a frequency table."""

    category: str
    count: int
    proportion: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True)
class FrequencyTable:
    """Frequency table with optional bootstrap CIs."""

    rows: list[FrequencyRow]
    total: int
    chi_squared: ChiSquaredResult | None


def frequency_table(
    responses: list[str],
    *,
    categories: list[str] | None = None,
    bootstrap_ci_conf: float | None = 0.95,
    n_resamples: int = 2000,
    seed: int | None = None,
) -> FrequencyTable:
    """Build a frequency table with optional bootstrap CIs and chi-squared test.

    Args:
        responses: List of categorical responses.
        categories: Explicit category list (for ordering / zero-count inclusion).
        bootstrap_ci_conf: Confidence level for bootstrap CIs. None to skip.
        n_resamples: Bootstrap resamples. Default 2000.
        seed: RNG seed for reproducibility.

    Returns:
        FrequencyTable with rows sorted by count (descending), plus
        chi-squared GOF test vs uniform distribution.
    """
    if categories is None:
        cats = sorted(set(responses))
    else:
        cats = list(categories)

    total = len(responses)

    # Count occurrences
    counts: dict[str, int] = {c: 0 for c in cats}
    for r in responses:
        if r in counts:
            counts[r] += 1

    rows = []
    for cat in cats:
        cnt = counts[cat]
        prop = cnt / total if total > 0 else 0.0

        ci_lo = 0.0
        ci_hi = 0.0
        if bootstrap_ci_conf is not None and total >= 5:
            result = bootstrap_ci(
                responses,
                proportion_stat(cat),
                confidence=bootstrap_ci_conf,
                n_resamples=n_resamples,
                seed=seed,
            )
            ci_lo = result.ci_lower
            ci_hi = result.ci_upper
        else:
            ci_lo = prop
            ci_hi = prop

        rows.append(FrequencyRow(
            category=cat,
            count=cnt,
            proportion=prop,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
        ))

    # Sort by count descending
    rows.sort(key=lambda r: r.count, reverse=True)

    # Chi-squared GOF vs uniform
    observed = {cat: counts[cat] for cat in cats}
    chi2_result = chi_squared_test(observed) if total > 0 and len(cats) > 1 else None

    return FrequencyTable(rows=rows, total=total, chi_squared=chi2_result)


# ---------------------------------------------------------------------------
# borda_count
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BordaResult:
    """Borda count ranking result."""

    scores: dict[str, float]
    ranking: list[str]
    n_voters: int


def borda_count(
    rankings: list[dict[str, int]],
) -> BordaResult:
    """Compute Borda count aggregate ranking.

    Args:
        rankings: List of N ranking dicts mapping item name -> rank (1 = best).

    Returns:
        BordaResult with per-item mean Borda scores and aggregate ranking.

    Raises:
        ValueError: If rankings is empty or items are inconsistent.
    """
    if not rankings:
        raise ValueError("rankings must not be empty")

    items = set(rankings[0].keys())
    for i, r in enumerate(rankings):
        if set(r.keys()) != items:
            raise ValueError(
                f"ranking {i} has items {set(r.keys())}, expected {items}"
            )

    k = len(items)
    n_voters = len(rankings)

    # Accumulate Borda scores: rank r -> score (K - r)
    totals: dict[str, float] = {item: 0.0 for item in items}
    for r in rankings:
        for item, rank in r.items():
            totals[item] += k - rank

    # Normalize to mean per voter
    scores = {item: total / n_voters for item, total in totals.items()}

    # Sort by score descending, then alphabetically for ties
    ranking = sorted(scores.keys(), key=lambda x: (-scores[x], x))

    return BordaResult(scores=scores, ranking=ranking, n_voters=n_voters)
