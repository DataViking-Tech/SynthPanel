"""Tests for synth_panel.stats — sp-5on.7 + sp-5on.8."""

from __future__ import annotations

import pytest

from synth_panel.stats import (
    bootstrap_ci,
    borda_count,
    chi_squared_test,
    frequency_table,
    kendall_w,
    krippendorff_alpha,
    proportion_stat,
)

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_known_mean(self):
        """Mean of [0]*50 + [1]*50 should have CI containing 0.5."""
        data = [0.0] * 50 + [1.0] * 50
        result = bootstrap_ci(data, lambda x: sum(x) / len(x), seed=42)
        assert result.method == "BCa"
        assert result.ci_lower <= 0.5 <= result.ci_upper
        assert abs(result.estimate - 0.5) < 0.01

    def test_proportion_stat(self):
        """proportion_stat helper returns correct proportion."""
        data = ["A", "B", "B", "C", "B"]
        fn = proportion_stat("B")
        assert fn(data) == pytest.approx(0.6)

    def test_proportion_ci_contains_true_value(self):
        """CI for proportion should contain the observed proportion."""
        data = ["B"] * 40 + ["A"] * 10 + ["C"] * 10 + ["D"] * 10
        result = bootstrap_ci(data, proportion_stat("B"), seed=42)
        assert result.ci_lower < result.estimate < result.ci_upper
        assert result.ci_lower > 0
        assert result.ci_upper < 1

    def test_narrow_ci_for_large_n(self):
        """CI width should shrink with larger N."""
        small = [0.0] * 10 + [1.0] * 10
        large = [0.0] * 200 + [1.0] * 200
        r_small = bootstrap_ci(small, lambda x: sum(x) / len(x), seed=42)
        r_large = bootstrap_ci(large, lambda x: sum(x) / len(x), seed=42)
        width_small = r_small.ci_upper - r_small.ci_lower
        width_large = r_large.ci_upper - r_large.ci_lower
        assert width_large < width_small

    def test_min_data_size(self):
        """Should reject data with fewer than 5 observations."""
        with pytest.raises(ValueError, match="at least 5"):
            bootstrap_ci([1.0, 2.0, 3.0], lambda x: sum(x) / len(x))

    def test_confidence_bounds(self):
        """Confidence must be in (0, 1)."""
        data = [1.0] * 10
        with pytest.raises(ValueError, match="confidence"):
            bootstrap_ci(data, lambda x: sum(x) / len(x), confidence=1.5)

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        data = [float(i) for i in range(50)]
        r1 = bootstrap_ci(data, lambda x: sum(x) / len(x), seed=123)
        r2 = bootstrap_ci(data, lambda x: sum(x) / len(x), seed=123)
        assert r1 == r2


# ---------------------------------------------------------------------------
# chi_squared_test
# ---------------------------------------------------------------------------


class TestChiSquaredTest:
    def test_uniform_data_high_p(self):
        """Perfectly uniform data should have high p-value."""
        observed = {"A": 25, "B": 25, "C": 25, "D": 25}
        result = chi_squared_test(observed)
        assert result.statistic == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0, abs=0.01)
        assert result.df == 3

    def test_skewed_data_low_p(self):
        """Highly skewed data should have low p-value."""
        observed = {"A": 5, "B": 80, "C": 10, "D": 5}
        result = chi_squared_test(observed)
        assert result.statistic > 50
        assert result.p_value < 0.001

    def test_known_statistic(self):
        """Hand-computed chi-squared: O={10,20,30}, E=uniform=20 each.
        chi2 = (10-20)^2/20 + (20-20)^2/20 + (30-20)^2/20 = 5+0+5 = 10."""
        observed = {"A": 10, "B": 20, "C": 30}
        result = chi_squared_test(observed)
        assert result.statistic == pytest.approx(10.0)
        assert result.df == 2
        # p-value for chi2=10, df=2 is ~0.0067
        assert result.p_value == pytest.approx(0.0067, abs=0.001)

    def test_custom_expected(self):
        """Custom expected distribution."""
        observed = {"A": 30, "B": 70}
        expected = {"A": 40.0, "B": 60.0}
        result = chi_squared_test(observed, expected)
        # chi2 = (30-40)^2/40 + (70-60)^2/60 = 2.5 + 1.667 = 4.167
        assert result.statistic == pytest.approx(4.167, abs=0.01)

    def test_cramers_v_perfect_skew(self):
        """Cramer's V should be > 0 for non-uniform data."""
        observed = {"A": 5, "B": 80, "C": 10, "D": 5}
        result = chi_squared_test(observed)
        assert result.cramers_v > 0.3  # large effect

    def test_cramers_v_uniform(self):
        """Cramer's V should be 0 for perfectly uniform data."""
        observed = {"A": 25, "B": 25, "C": 25, "D": 25}
        result = chi_squared_test(observed)
        assert result.cramers_v == pytest.approx(0.0, abs=0.001)

    def test_warning_small_expected(self):
        """Warn when any expected count < 5."""
        observed = {"A": 2, "B": 3, "C": 1, "D": 2}
        result = chi_squared_test(observed)
        assert result.warning is not None
        assert "expected count" in result.warning.lower()

    def test_empty_rejects(self):
        with pytest.raises(ValueError):
            chi_squared_test({})

    def test_mismatched_keys_rejects(self):
        with pytest.raises(ValueError):
            chi_squared_test({"A": 10, "B": 20}, {"A": 15.0, "C": 15.0})


# ---------------------------------------------------------------------------
# kendall_w
# ---------------------------------------------------------------------------


class TestKendallW:
    def test_perfect_agreement(self):
        """All raters rank identically -> W = 1.0."""
        rankings = [[1, 2, 3, 4]] * 10
        result = kendall_w(rankings)
        assert result.w == pytest.approx(1.0)
        assert result.n_raters == 10
        assert result.n_items == 4

    def test_no_agreement(self):
        """Two raters with opposite rankings -> W = 0."""
        rankings = [[1, 2, 3, 4], [4, 3, 2, 1]]
        result = kendall_w(rankings)
        assert result.w == pytest.approx(0.0, abs=0.01)

    def test_known_w(self):
        """Hand-computed example.
        3 raters, 4 items:
        Rater 1: [1, 2, 3, 4]
        Rater 2: [1, 2, 4, 3]
        Rater 3: [1, 3, 2, 4]
        R_j = [3, 7, 9, 11], R_bar = 3*5/2 = 7.5
        S = (3-7.5)^2 + (7-7.5)^2 + (9-7.5)^2 + (11-7.5)^2
          = 20.25 + 0.25 + 2.25 + 12.25 = 35.0
        W = 12*35 / (9 * (64-4)) = 420 / 540 = 0.7778
        """
        rankings = [[1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4]]
        result = kendall_w(rankings)
        assert result.w == pytest.approx(0.7778, abs=0.001)
        assert result.df == 3
        assert result.chi_squared == pytest.approx(3 * 3 * 0.7778, abs=0.01)

    def test_p_value_significant(self):
        """High W with many raters should yield small p-value."""
        rankings = [[1, 2, 3, 4]] * 20  # perfect agreement, 20 raters
        result = kendall_w(rankings)
        assert result.p_value < 0.001

    def test_empty_rejects(self):
        with pytest.raises(ValueError):
            kendall_w([])

    def test_inconsistent_lengths_rejects(self):
        with pytest.raises(ValueError):
            kendall_w([[1, 2, 3], [1, 2]])


# ---------------------------------------------------------------------------
# frequency_table
# ---------------------------------------------------------------------------


class TestFrequencyTable:
    def test_basic_counts(self):
        responses = ["A", "B", "B", "C", "B", "A"]
        ft = frequency_table(responses, bootstrap_ci_conf=None)
        row_map = {r.category: r for r in ft.rows}
        assert row_map["B"].count == 3
        assert row_map["B"].proportion == pytest.approx(0.5)
        assert row_map["A"].count == 2
        assert row_map["C"].count == 1
        assert ft.total == 6

    def test_explicit_categories_includes_zeros(self):
        responses = ["A", "A", "A"]
        ft = frequency_table(responses, categories=["A", "B", "C"], bootstrap_ci_conf=None)
        row_map = {r.category: r for r in ft.rows}
        assert row_map["B"].count == 0
        assert row_map["C"].count == 0

    def test_sorted_descending(self):
        responses = ["C"] * 5 + ["A"] * 10 + ["B"] * 3
        ft = frequency_table(responses, bootstrap_ci_conf=None)
        counts = [r.count for r in ft.rows]
        assert counts == sorted(counts, reverse=True)

    def test_bootstrap_cis_present(self):
        responses = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
        ft = frequency_table(responses, bootstrap_ci_conf=0.95, seed=42)
        for row in ft.rows:
            assert row.ci_lower <= row.proportion <= row.ci_upper

    def test_chi_squared_included(self):
        responses = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
        ft = frequency_table(responses, bootstrap_ci_conf=None)
        assert ft.chi_squared is not None
        assert ft.chi_squared.df == 2


# ---------------------------------------------------------------------------
# borda_count
# ---------------------------------------------------------------------------


class TestBordaCount:
    def test_unanimous_ranking(self):
        """All voters rank the same -> Borda matches."""
        rankings = [{"A": 1, "B": 2, "C": 3}] * 5
        result = borda_count(rankings)
        assert result.ranking == ["A", "B", "C"]
        # A gets 2 pts each = 10 total / 5 voters = 2.0 mean
        assert result.scores["A"] == pytest.approx(2.0)
        assert result.scores["B"] == pytest.approx(1.0)
        assert result.scores["C"] == pytest.approx(0.0)

    def test_tie(self):
        """Equal votes produce tied scores."""
        rankings = [
            {"A": 1, "B": 2},
            {"A": 2, "B": 1},
        ]
        result = borda_count(rankings)
        assert result.scores["A"] == result.scores["B"]

    def test_known_result(self):
        """3 voters, 4 items. A is consistently top-2."""
        rankings = [
            {"A": 1, "B": 2, "C": 3, "D": 4},
            {"A": 2, "B": 1, "C": 3, "D": 4},
            {"A": 1, "B": 3, "C": 2, "D": 4},
        ]
        result = borda_count(rankings)
        # A: (3+2+3)/3 = 2.67, B: (2+3+1)/3 = 2.0, C: (1+1+2)/3 = 1.33, D: 0
        assert result.ranking[0] == "A"
        assert result.scores["A"] == pytest.approx(2.667, abs=0.01)

    def test_empty_rejects(self):
        with pytest.raises(ValueError):
            borda_count([])

    def test_inconsistent_items_rejects(self):
        with pytest.raises(ValueError):
            borda_count([{"A": 1, "B": 2}, {"A": 1, "C": 2}])


# ---------------------------------------------------------------------------
# krippendorff_alpha
# ---------------------------------------------------------------------------


class TestKrippendorffAlpha:
    def test_perfect_agreement_nominal(self):
        """All raters agree on every item -> alpha = 1.0."""
        data = [
            ["A", "B", "C", "A", "B"],
            ["A", "B", "C", "A", "B"],
            ["A", "B", "C", "A", "B"],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert result.alpha == pytest.approx(1.0)
        assert result.level == "nominal"

    def test_no_agreement_nominal(self):
        """Systematic disagreement -> alpha near 0 or negative."""
        data = [
            ["A", "B", "C"],
            ["B", "C", "A"],
            ["C", "A", "B"],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert result.alpha < 0.1

    def test_known_alpha_nominal(self):
        """4 raters, 12 items, some missing data.
        Verified against krippendorff 0.8.1 reference package: alpha = 0.871."""
        data = [
            [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3],
            [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None],
            [None, None, 2, 1, 3, 3, 4, 2, 2, 1, 1, 3],
            [1, None, 2, 1, 3, 3, 4, 2, 2, 1, 1, 3],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert result.alpha == pytest.approx(0.871, abs=0.01)
        assert result.n_raters == 4
        assert result.n_items == 12

    def test_ordinal_level(self):
        """Ordinal alpha should differ from nominal for ordered data."""
        data = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 5, 4],  # swapped last two
        ]
        result_nom = krippendorff_alpha(data, "nominal")
        result_ord = krippendorff_alpha(data, "ordinal")
        # Ordinal should be higher: swapping adjacent ranks is a smaller
        # disagreement in ordinal than in nominal
        assert result_ord.alpha > result_nom.alpha

    def test_interval_level(self):
        """Interval alpha for numeric data."""
        data = [
            [1.0, 2.0, 3.0, 4.0],
            [1.1, 2.1, 2.9, 4.1],
            [0.9, 1.9, 3.1, 3.9],
        ]
        result = krippendorff_alpha(data, "interval")
        assert result.alpha > 0.9  # Very close agreement
        assert result.level == "interval"

    def test_handles_missing_data(self):
        """None values should be skipped, not crash."""
        data = [
            ["A", None, "C"],
            [None, "B", "C"],
            ["A", "B", None],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert isinstance(result.alpha, float)
        assert result.n_raters == 3
        assert result.n_items == 3

    def test_all_same_value(self):
        """All values identical -> alpha = 1.0 (D_e = 0 special case)."""
        data = [
            ["X", "X", "X"],
            ["X", "X", "X"],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert result.alpha == pytest.approx(1.0)

    def test_invalid_level_rejects(self):
        with pytest.raises(ValueError, match="level"):
            krippendorff_alpha([["A"]], "ratio")

    def test_empty_rejects(self):
        with pytest.raises(ValueError):
            krippendorff_alpha([])

    def test_mismatched_lengths_rejects(self):
        with pytest.raises(ValueError):
            krippendorff_alpha([["A", "B"], ["A"]])

    def test_interpretation_strong(self):
        data = [["A", "B", "C"]] * 5
        result = krippendorff_alpha(data, "nominal")
        assert "Strong" in result.interpretation or "reliable" in result.interpretation

    def test_interpretation_weak(self):
        """Low agreement should flag as weak/unreliable."""
        data = [
            ["A", "B", "C", "A"],
            ["B", "C", "A", "B"],
            ["C", "A", "B", "C"],
        ]
        result = krippendorff_alpha(data, "nominal")
        assert "caution" in result.interpretation.lower() or "No meaningful" in result.interpretation
