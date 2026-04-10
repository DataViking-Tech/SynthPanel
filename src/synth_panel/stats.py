"""Pure-Python statistical tests for synthetic panel analysis.

All functions use only ``math``, ``random``, and ``dataclasses`` from stdlib.
No scipy or numpy required.

Implements: bootstrap BCa confidence intervals, chi-squared goodness-of-fit,
Kendall's W concordance, frequency tables, Borda count ranking,
persona cluster analysis (agglomerative + silhouette), cross-model
convergence/divergence classification, and Krippendorff's alpha.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "BootstrapResult",
    "BordaResult",
    "ChiSquaredResult",
    "Cluster",
    "ClusterResult",
    "ConvergenceLevel",
    "ConvergenceReport",
    "FindingConvergence",
    "FrequencyRow",
    "FrequencyTable",
    "KendallWResult",
    "KrippendorffResult",
    "ModelDistribution",
    "bootstrap_ci",
    "borda_count",
    "chi_squared_test",
    "cluster_personas",
    "convergence_report",
    "frequency_table",
    "kendall_w",
    "krippendorff_alpha",
    "proportion_stat",
    "robustness_score",
    "silhouette_score",
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
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        # Central region
        q = p - 0.5
        r = q * q
        return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        # Upper tail
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )


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


def proportion_stat(value) -> Callable:
    """Return a stat function that computes the proportion of *value* in data."""

    def _fn(data: list) -> float:
        return sum(1 for x in data if x == value) / len(data)

    return _fn


def bootstrap_ci(
    data: list[float],
    stat_fn: Callable,
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
    ci_lower = theta_star[math.floor(a1 * n_resamples)]
    ci_upper = theta_star[math.floor(a2 * n_resamples)]

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
            raise ValueError(f"expected keys {set(expected.keys())} don't match observed keys {set(observed.keys())}")

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
        warning = f"Some expected count(s) < 5 (min={min_expected:.1f}). Chi-squared approximation may be unreliable."

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
            raise ValueError(f"ranking {i} has length {len(r)}, expected {k_items}")

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

        rows.append(
            FrequencyRow(
                category=cat,
                count=cnt,
                proportion=prop,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
            )
        )

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
            raise ValueError(f"ranking {i} has items {set(r.keys())}, expected {items}")

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


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KrippendorffResult:
    """Result of Krippendorff's alpha computation."""

    alpha: float  # Alpha coefficient [-1, 1], typically [0, 1]
    n_raters: int  # Number of raters (models)
    n_items: int  # Number of items (personas)
    n_categories: int  # Number of distinct values observed
    level: str  # Measurement level used ("nominal", "ordinal", "interval")
    interpretation: str  # Human-readable interpretation


def _nominal_delta(c: object, k: object) -> float:
    """Nominal distance: 0 if equal, 1 if different."""
    return 0.0 if c == k else 1.0


def _interval_delta(c: object, k: object) -> float:
    """Interval distance: squared difference."""
    return (float(c) - float(k)) ** 2


def _ordinal_delta(
    c: object,
    k: object,
    sorted_values: list,
    value_counts: dict[object, float],
) -> float:
    """Ordinal distance per Krippendorff (2011).

    d_ord(c, k) = (sum of n_g for all g where c <= g <= k
                   - (n_c + n_k) / 2) ^ 2
    """
    i_c = sorted_values.index(c)
    i_k = sorted_values.index(k)
    if i_c > i_k:
        i_c, i_k = i_k, i_c
    running_sum = sum(value_counts[sorted_values[g]] for g in range(i_c, i_k + 1))
    running_sum -= (value_counts[c] + value_counts[k]) / 2
    return running_sum**2


def _interpret_alpha(alpha: float) -> str:
    if alpha >= 0.80:
        return "Strong agreement — reliable for drawing conclusions"
    elif alpha >= 0.667:
        return "Moderate agreement — tentative conclusions only"
    elif alpha >= 0.40:
        return "Weak agreement — interpret with caution"
    else:
        return "No meaningful agreement — model choice dominates"


_VALID_LEVELS = frozenset({"nominal", "ordinal", "interval"})


def krippendorff_alpha(
    reliability_data: list[list[str | int | float | None]],
    level_of_measurement: str = "nominal",
) -> KrippendorffResult:
    """Compute Krippendorff's alpha inter-rater reliability coefficient.

    Uses the coincidence matrix method from Krippendorff (2011), which
    handles missing data correctly.

    Args:
        reliability_data: Rater-by-item matrix.  Each inner list represents
            one rater's codings across all items.  ``None`` indicates a
            missing value.  Shape: ``[n_raters][n_items]``.
        level_of_measurement: One of ``"nominal"``, ``"ordinal"``,
            ``"interval"``.

    Returns:
        :class:`KrippendorffResult` with alpha, metadata, interpretation.

    Raises:
        ValueError: If *reliability_data* is empty, all values are ``None``,
            *level_of_measurement* is invalid, or rater lists have different
            lengths.
    """
    # --- Validation ---
    if not reliability_data:
        raise ValueError("reliability_data must not be empty")

    if level_of_measurement not in _VALID_LEVELS:
        raise ValueError(f"level_of_measurement must be one of {sorted(_VALID_LEVELS)}, got {level_of_measurement!r}")

    n_raters = len(reliability_data)
    n_items = len(reliability_data[0])
    if n_items == 0:
        raise ValueError("reliability_data contains empty rater lists")
    for i, row in enumerate(reliability_data):
        if len(row) != n_items:
            raise ValueError(
                f"All rater lists must have the same length; rater 0 has {n_items} items but rater {i} has {len(row)}"
            )

    # --- Collect all non-None values and discover categories ---
    all_values: list[object] = []
    for row in reliability_data:
        for v in row:
            if v is not None:
                all_values.append(v)

    if not all_values:
        raise ValueError("reliability_data contains only None values")

    categories = sorted(set(all_values), key=lambda x: (isinstance(x, str), x))
    n_categories = len(categories)

    # --- Step 1: Build coincidence matrix ---
    o: dict[tuple[object, object], float] = {}
    for c in categories:
        for k_val in categories:
            o[(c, k_val)] = 0.0

    for item_idx in range(n_items):
        values_for_item = [
            reliability_data[r][item_idx] for r in range(n_raters) if reliability_data[r][item_idx] is not None
        ]
        m_u = len(values_for_item)
        if m_u < 2:
            continue

        for i in range(m_u):
            for j in range(m_u):
                if i == j:
                    continue
                o[(values_for_item[i], values_for_item[j])] += 1.0 / (m_u - 1)

    # --- Step 2: Compute marginals ---
    n_c: dict[object, float] = {}
    for c in categories:
        n_c[c] = sum(o[(c, k_val)] for k_val in categories)

    n = sum(n_c.values())

    if n < 2:
        raise ValueError("Fewer than 2 pairable values — cannot compute alpha")

    # --- Build delta function ---
    if level_of_measurement == "nominal":

        def delta(c: object, k_val: object) -> float:
            return _nominal_delta(c, k_val)

    elif level_of_measurement == "interval":

        def delta(c: object, k_val: object) -> float:
            return _interval_delta(c, k_val)

    elif level_of_measurement == "ordinal":
        sorted_values = sorted(categories, key=lambda x: (isinstance(x, str), x))

        def delta(c: object, k_val: object) -> float:
            return _ordinal_delta(c, k_val, sorted_values, n_c)

    # --- Step 3: D_observed ---
    d_observed = 0.0
    for c in categories:
        for k_val in categories:
            if c != k_val:
                d_observed += o[(c, k_val)] * delta(c, k_val)
    d_observed /= n

    # --- Step 4: D_expected ---
    d_expected = 0.0
    for c in categories:
        for k_val in categories:
            if c != k_val:
                d_expected += n_c[c] * n_c[k_val] * delta(c, k_val)
    d_expected /= n * (n - 1)

    # --- Step 5: Alpha ---
    if d_expected == 0.0:
        alpha = 1.0  # All values identical
    else:
        alpha = 1.0 - d_observed / d_expected

    return KrippendorffResult(
        alpha=alpha,
        n_raters=n_raters,
        n_items=n_items,
        n_categories=n_categories,
        level=level_of_measurement,
        interpretation=_interpret_alpha(alpha),
    )


# ---------------------------------------------------------------------------
# Cross-model convergence/divergence classification (sp-5on.12)
# ---------------------------------------------------------------------------


class ConvergenceLevel(Enum):
    """Classification of cross-model agreement on a finding."""

    STRONG = "strong"  # alpha >= 0.80
    MODERATE = "moderate"  # 0.60 <= alpha < 0.80
    WEAK = "weak"  # 0.40 <= alpha < 0.60
    NONE = "none"  # alpha < 0.40


@dataclass(frozen=True)
class ModelDistribution:
    """Per-model response distribution for one finding."""

    model: str
    distribution: dict[str, float]  # category -> proportion
    n: int  # Number of responses from this model
    top_choice: str  # Modal response


@dataclass(frozen=True)
class FindingConvergence:
    """Convergence assessment for a single finding/question."""

    question_index: int
    question_text: str
    alpha: float  # Krippendorff's alpha
    level: ConvergenceLevel
    per_model: list[ModelDistribution]  # One per model
    top_choice_agreement: bool  # Do all models agree on the top choice?
    divergent_models: list[str]  # Models whose top choice differs from majority
    interpretation: str  # Human-readable summary


@dataclass(frozen=True)
class ConvergenceReport:
    """Full convergence analysis across all questions in a multi-model run."""

    findings: list[FindingConvergence]
    overall_alpha: float  # Mean alpha across all questions
    overall_level: ConvergenceLevel  # Classification of overall alpha
    n_convergent: int  # Findings with alpha >= 0.60
    n_divergent: int  # Findings with alpha < 0.40
    n_models: int
    model_names: list[str]


def _classify_alpha(alpha: float) -> ConvergenceLevel:
    """Classify an alpha value into a convergence level."""
    if alpha >= 0.80:
        return ConvergenceLevel.STRONG
    elif alpha >= 0.60:
        return ConvergenceLevel.MODERATE
    elif alpha >= 0.40:
        return ConvergenceLevel.WEAK
    else:
        return ConvergenceLevel.NONE


def convergence_report(
    multi_model_responses: dict[str, list[list[str]]],
    question_texts: list[str],
    *,
    level_of_measurement: str = "nominal",
) -> ConvergenceReport:
    """Classify cross-model convergence for each question in a multi-model run.

    Args:
        multi_model_responses: model_name -> list of lists.
            Outer list: one entry per persona (length N).
            Inner list: one response per question (length Q).
            All models must have the same N and Q.

        question_texts: List of Q question text strings (for reporting).

        level_of_measurement: Passed to krippendorff_alpha.
            "nominal" for pick-one-of-N, "ordinal" for rankings/Likert.

    Returns:
        ConvergenceReport with per-question and overall convergence.

    Raises:
        ValueError: If models have inconsistent N or Q, fewer than 2 models,
                    or question_texts length doesn't match Q.
    """
    model_names = list(multi_model_responses.keys())
    n_models = len(model_names)

    if n_models < 2:
        raise ValueError(f"Need at least 2 models, got {n_models}")

    # Validate consistent shapes
    first_model = model_names[0]
    n_personas = len(multi_model_responses[first_model])

    for model in model_names:
        if len(multi_model_responses[model]) != n_personas:
            raise ValueError(
                f"Model {model!r} has {len(multi_model_responses[model])} personas, "
                f"expected {n_personas} (from {first_model!r})"
            )

    # Determine Q from first persona of first model
    if n_personas == 0:
        raise ValueError("Models must have at least 1 persona")

    n_questions = len(multi_model_responses[first_model][0])

    for model in model_names:
        for p_idx, persona_responses in enumerate(multi_model_responses[model]):
            if len(persona_responses) != n_questions:
                raise ValueError(
                    f"Model {model!r} persona {p_idx} has {len(persona_responses)} questions, expected {n_questions}"
                )

    if len(question_texts) != n_questions:
        raise ValueError(f"question_texts has {len(question_texts)} entries but data has {n_questions} questions")

    findings: list[FindingConvergence] = []

    for q in range(n_questions):
        # Build ratings matrix: [M raters][N items]
        reliability_data: list[list[str]] = []
        for model in model_names:
            rater_row = [multi_model_responses[model][n][q] for n in range(n_personas)]
            reliability_data.append(rater_row)

        # Compute alpha
        alpha_result = krippendorff_alpha(reliability_data, level_of_measurement)
        alpha_q = alpha_result.alpha

        # Classify
        level = _classify_alpha(alpha_q)

        # Per-model distributions
        per_model: list[ModelDistribution] = []
        model_top_choices: dict[str, str] = {}

        for model in model_names:
            responses_m = [multi_model_responses[model][n][q] for n in range(n_personas)]
            counts = Counter(responses_m)
            total = len(responses_m)
            distribution = {cat: cnt / total for cat, cnt in sorted(counts.items())}
            top_choice = counts.most_common(1)[0][0]
            model_top_choices[model] = top_choice

            per_model.append(
                ModelDistribution(
                    model=model,
                    distribution=distribution,
                    n=total,
                    top_choice=top_choice,
                )
            )

        # Divergent model detection: majority top choice
        top_choice_counts = Counter(model_top_choices.values())
        majority_top = top_choice_counts.most_common(1)[0][0]
        divergent = [m for m in model_names if model_top_choices[m] != majority_top]
        top_choice_agreement = len(divergent) == 0

        # Interpretation
        if level == ConvergenceLevel.STRONG:
            interpretation = (
                f"Strong convergence (alpha={alpha_q:.2f}). All models agree: {majority_top} is the top choice."
            )
        elif level == ConvergenceLevel.MODERATE:
            interpretation = (
                f"Moderate convergence (alpha={alpha_q:.2f}). "
                f"Models mostly agree on {majority_top} but with some variation."
            )
        elif level == ConvergenceLevel.WEAK:
            interpretation = (
                f"Weak convergence (alpha={alpha_q:.2f}). "
                f"Interpret with caution. {divergent} disagree with the majority."
            )
        else:
            interpretation = (
                f"No convergence (alpha={alpha_q:.2f}). "
                f"Model choice dominates this finding. {divergent} diverge. Not reliable."
            )

        findings.append(
            FindingConvergence(
                question_index=q,
                question_text=question_texts[q],
                alpha=alpha_q,
                level=level,
                per_model=per_model,
                top_choice_agreement=top_choice_agreement,
                divergent_models=divergent,
                interpretation=interpretation,
            )
        )

    # Overall metrics
    overall_alpha = sum(f.alpha for f in findings) / len(findings)
    overall_level = _classify_alpha(overall_alpha)
    n_convergent = sum(1 for f in findings if f.alpha >= 0.60)
    n_divergent = sum(1 for f in findings if f.alpha < 0.40)

    return ConvergenceReport(
        findings=findings,
        overall_alpha=overall_alpha,
        overall_level=overall_level,
        n_convergent=n_convergent,
        n_divergent=n_divergent,
        n_models=n_models,
        model_names=model_names,
    )


# ---------------------------------------------------------------------------
# Persona cluster analysis (agglomerative clustering + silhouette scoring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cluster:
    """A single cluster of personas."""

    cluster_id: int
    persona_names: list[str]
    size: int
    dominant_responses: dict[int, str]  # question_index -> most common response
    dominant_proportions: dict[int, float]


@dataclass(frozen=True)
class ClusterResult:
    """Result of persona cluster analysis."""

    clusters: list[Cluster]
    n_clusters: int
    silhouette_score: float
    persona_assignments: dict[str, int]  # persona_name -> cluster_id
    k_range_tested: tuple[int, int]


def _encode_responses(
    persona_responses: dict[str, list[str]],
    categories: list[str],
) -> dict[str, list[float]]:
    """Encode categorical responses as one-hot feature vectors.

    For Q questions and C categories, vector length = Q * C.
    Each question contributes a one-hot block of length C.
    """
    cat_index = {c: i for i, c in enumerate(categories)}
    c = len(categories)
    result: dict[str, list[float]] = {}
    for name, responses in persona_responses.items():
        vec: list[float] = []
        for resp in responses:
            block = [0.0] * c
            if resp in cat_index:
                block[cat_index[resp]] = 1.0
            vec.extend(block)
        result[name] = vec
    return result


def _sq_euclidean(a: list[float], b: list[float]) -> float:
    """Squared Euclidean distance between two vectors."""
    return sum((ai - bi) ** 2 for ai, bi in zip(a, b))


def _agglomerative_ward(
    names: list[str],
    vectors: list[list[float]],
) -> list[tuple[int, int, float, int]]:
    """Agglomerative clustering with Ward's method.

    Returns a linkage list of (cluster_i, cluster_j, distance, new_size)
    with N-1 entries.
    """
    n = len(names)

    # Initialize: each point is its own cluster
    # Active clusters tracked by id; ids 0..n-1 are initial singletons
    sizes: dict[int, int] = {i: 1 for i in range(n)}
    active: set[int] = set(range(n))

    # Pairwise squared Euclidean distance matrix (stored as dict for easy update)
    dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = _sq_euclidean(vectors[i], vectors[j])

    def _get_dist(a: int, b: int) -> float:
        if a > b:
            a, b = b, a
        return dist[(a, b)]

    linkage: list[tuple[int, int, float, int]] = []
    next_id = n

    for _ in range(n - 1):
        # Find minimum distance pair among active clusters
        best_dist = float("inf")
        best_i = -1
        best_j = -1
        active_list = sorted(active)
        for idx_a in range(len(active_list)):
            for idx_b in range(idx_a + 1, len(active_list)):
                ci, cj = active_list[idx_a], active_list[idx_b]
                d = _get_dist(ci, cj)
                if d < best_dist:
                    best_dist = d
                    best_i = ci
                    best_j = cj

        # Record merge
        new_size = sizes[best_i] + sizes[best_j]
        linkage.append((best_i, best_j, best_dist, new_size))

        # Update distances using Lance-Williams formula for Ward's method
        new_id = next_id
        next_id += 1
        n_i = sizes[best_i]
        n_j = sizes[best_j]
        d_ij = best_dist

        for k in active:
            if k in (best_i, best_j):
                continue
            n_k = sizes[k]
            d_ik = _get_dist(best_i, k)
            d_jk = _get_dist(best_j, k)
            # Ward's Lance-Williams:
            # d(new, k) = ((n_k + n_i)*d(i,k) + (n_k + n_j)*d(j,k) - n_k*d(i,j)) / (n_k + n_i + n_j)
            d_new = ((n_k + n_i) * d_ik + (n_k + n_j) * d_jk - n_k * d_ij) / (n_k + n_i + n_j)
            key = (min(new_id, k), max(new_id, k))
            dist[key] = d_new

        # Remove merged clusters, add new one
        active.discard(best_i)
        active.discard(best_j)
        active.add(new_id)
        sizes[new_id] = new_size

    return linkage


def _cut_dendrogram(
    linkage: list[tuple[int, int, float, int]],
    n: int,
    k: int,
) -> list[int]:
    """Cut a dendrogram to produce k clusters.

    Returns a list of cluster labels (length n), one per original point.
    """
    # Take the first (n - k) merges. The remaining k groups are the clusters.
    # Build a union-find to track which original points end up together.
    parent: dict[int, int] = {i: i for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int, new_id: int) -> None:
        parent[new_id] = new_id
        ra, rb = find(a), find(b)
        parent[ra] = new_id
        parent[rb] = new_id

    # Apply the first (n - k) merges
    n_merges = n - k
    for merge_idx in range(n_merges):
        ci, cj, _d, _s = linkage[merge_idx]
        new_id = n + merge_idx
        union(ci, cj, new_id)

    # Assign cluster labels
    root_to_label: dict[int, int] = {}
    labels: list[int] = []
    next_label = 0
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels.append(root_to_label[root])

    return labels


def silhouette_score(
    labels: list[int],
    distance_matrix: list[list[float]],
) -> float:
    """Compute mean silhouette coefficient.

    Args:
        labels: Cluster assignment for each point. Length N.
        distance_matrix: N x N symmetric distance matrix.

    Returns:
        Mean silhouette score in [-1, 1]. Higher is better.
        Returns 0.0 if all points are in one cluster.
    """
    n = len(labels)
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return 0.0

    # Group points by cluster
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)

    silhouettes: list[float] = []
    for i in range(n):
        ci = labels[i]
        members_ci = clusters[ci]

        # a(i): mean distance to other members of same cluster
        if len(members_ci) <= 1:
            a_i = 0.0
        else:
            a_i = sum(distance_matrix[i][j] for j in members_ci if j != i) / (len(members_ci) - 1)

        # b(i): min mean distance to any other cluster
        b_i = float("inf")
        for lab, members in clusters.items():
            if lab == ci:
                continue
            mean_dist = sum(distance_matrix[i][j] for j in members) / len(members)
            if mean_dist < b_i:
                b_i = mean_dist

        denom = max(a_i, b_i)
        if denom == 0:
            s_i = 0.0
        else:
            s_i = (b_i - a_i) / denom
        silhouettes.append(s_i)

    return sum(silhouettes) / len(silhouettes)


def cluster_personas(
    persona_responses: dict[str, list[str]],
    *,
    categories: list[str] | None = None,
    min_k: int = 2,
    max_k: int = 5,
) -> ClusterResult:
    """Cluster personas by response patterns using agglomerative clustering.

    Uses Ward's method with one-hot encoded response vectors. Optimal k
    selected by maximum mean silhouette score.

    Args:
        persona_responses: persona_name -> list of responses (one per question).
        categories: Explicit category list. If None, inferred from data.
        min_k: Minimum number of clusters to test. Default 2.
        max_k: Maximum number of clusters to test. Default 5.

    Returns:
        ClusterResult with optimal clustering, silhouette score, assignments.

    Raises:
        ValueError: If fewer than 2*min_k personas, or inconsistent response lengths.
    """
    names = sorted(persona_responses.keys())
    n = len(names)

    if n < 2 * min_k:
        raise ValueError(f"Need at least {2 * min_k} personas (2*min_k), got {n}")

    # Validate consistent response lengths
    q = len(persona_responses[names[0]])
    for name in names:
        if len(persona_responses[name]) != q:
            raise ValueError(f"Persona {name!r} has {len(persona_responses[name])} responses, expected {q}")

    # Cap max_k at N-1
    max_k = min(max_k, n - 1)
    if max_k < min_k:
        max_k = min_k

    # 1. Encode
    if categories is None:
        cats: set[str] = set()
        for resps in persona_responses.values():
            cats.update(resps)
        categories = sorted(cats)
    encoded = _encode_responses(persona_responses, categories)
    vectors = [encoded[name] for name in names]

    # 2. Full pairwise distance matrix (squared Euclidean)
    full_dist: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _sq_euclidean(vectors[i], vectors[j])
            full_dist[i][j] = d
            full_dist[j][i] = d

    # 3. Agglomerative clustering
    linkage = _agglomerative_ward(names, vectors)

    # 4-6. Try each k, compute silhouette, pick best
    best_k = min_k
    best_score = -2.0
    best_labels: list[int] = []

    for k in range(min_k, max_k + 1):
        labels = _cut_dendrogram(linkage, n, k)
        score = silhouette_score(labels, full_dist)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # 7. Build result
    assignments = {names[i]: best_labels[i] for i in range(n)}

    # Group personas by cluster
    cluster_groups: dict[int, list[str]] = {}
    for i, name in enumerate(names):
        cluster_groups.setdefault(best_labels[i], []).append(name)

    clusters: list[Cluster] = []
    for cid in sorted(cluster_groups.keys()):
        members = cluster_groups[cid]
        # Dominant response per question (mode)
        dominant_responses: dict[int, str] = {}
        dominant_proportions: dict[int, float] = {}
        for qi in range(q):
            counts: dict[str, int] = {}
            for name in members:
                resp = persona_responses[name][qi]
                counts[resp] = counts.get(resp, 0) + 1
            mode = max(counts, key=lambda x: counts[x])
            dominant_responses[qi] = mode
            dominant_proportions[qi] = counts[mode] / len(members)

        clusters.append(
            Cluster(
                cluster_id=cid,
                persona_names=members,
                size=len(members),
                dominant_responses=dominant_responses,
                dominant_proportions=dominant_proportions,
            )
        )

    return ClusterResult(
        clusters=clusters,
        n_clusters=best_k,
        silhouette_score=best_score,
        persona_assignments=assignments,
        k_range_tested=(min_k, max_k),
    )


# ---------------------------------------------------------------------------
# robustness_score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobustnessResult:
    """Robustness score for a finding across persona variants."""

    finding_value: str
    overall_robustness: float
    per_persona: dict[str, float]
    n_personas: int
    k_variants_per_persona: int
    interpretation: str


def _interpret_robustness(r: float) -> str:
    if r >= 0.8:
        return "robust"
    elif r >= 0.6:
        return "moderately robust"
    elif r >= 0.4:
        return "sensitive"
    else:
        return "fragile"


def robustness_score(
    variant_responses: dict[str, list[str]],
    finding_value: str,
) -> RobustnessResult:
    """Compute robustness score for a finding across persona variants.

    Args:
        variant_responses: Mapping from source persona name to a list of
            responses from that persona's variants.
        finding_value: The value to test robustness for.

    Returns:
        RobustnessResult with per-persona and overall robustness scores.

    Raises:
        ValueError: If variant_responses is empty.
    """
    if not variant_responses:
        raise ValueError("variant_responses must not be empty")

    per_persona: dict[str, float] = {}
    k_values: list[int] = []

    for name, responses in variant_responses.items():
        k_i = len(responses)
        k_values.append(k_i)
        if k_i == 0:
            per_persona[name] = 0.0
        else:
            per_persona[name] = sum(1 for r in responses if r == finding_value) / k_i

    n = len(per_persona)
    overall = sum(per_persona.values()) / n

    # Median K across personas
    sorted_k = sorted(k_values)
    mid = len(sorted_k) // 2
    if len(sorted_k) % 2 == 0:
        median_k = (sorted_k[mid - 1] + sorted_k[mid]) // 2
    else:
        median_k = sorted_k[mid]

    return RobustnessResult(
        finding_value=finding_value,
        overall_robustness=overall,
        per_persona=per_persona,
        n_personas=n,
        k_variants_per_persona=median_k,
        interpretation=_interpret_robustness(overall),
    )
