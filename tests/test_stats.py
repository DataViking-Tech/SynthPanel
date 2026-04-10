"""Tests for synth_panel.stats — sp-5on.7 + sp-5on.8 + sp-5on.9 + sp-5on.12 + sp-5on.14."""

from __future__ import annotations

import pytest

from synth_panel.stats import (
    ConvergenceLevel,
    bootstrap_ci,
    borda_count,
    chi_squared_test,
    cluster_personas,
    convergence_report,
    frequency_table,
    kendall_w,
    krippendorff_alpha,
    proportion_stat,
    robustness_score,
    silhouette_score,
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


# ---------------------------------------------------------------------------
# convergence_report (sp-5on.12)
# ---------------------------------------------------------------------------


class TestConvergenceReport:
    def test_perfect_agreement(self):
        """All models give identical responses -> strong convergence."""
        responses = {
            "model_a": [["B", "A"], ["B", "A"], ["B", "A"]],
            "model_b": [["B", "A"], ["B", "A"], ["B", "A"]],
            "model_c": [["B", "A"], ["B", "A"], ["B", "A"]],
        }
        result = convergence_report(
            responses,
            ["Q1", "Q2"],
        )
        assert result.overall_level == ConvergenceLevel.STRONG
        assert result.overall_alpha == pytest.approx(1.0)
        assert result.n_convergent == 2
        assert result.n_divergent == 0
        for f in result.findings:
            assert f.top_choice_agreement is True
            assert f.divergent_models == []

    def test_total_disagreement(self):
        """Models systematically disagree -> no convergence."""
        # 3 models, 6 personas, 1 question.
        # Model A always says X, Model B always says Y, Model C always says Z.
        responses = {
            "model_a": [["X"]] * 6,
            "model_b": [["Y"]] * 6,
            "model_c": [["Z"]] * 6,
        }
        result = convergence_report(responses, ["Q1"])
        assert result.overall_level == ConvergenceLevel.NONE
        assert result.overall_alpha < 0.1
        assert result.n_divergent == 1

    def test_partial_convergence(self):
        """Two models agree, one diverges on some questions."""
        # Q1: all agree on B. Q2: model_c diverges.
        responses = {
            "model_a": [["B", "X"], ["B", "X"], ["B", "X"], ["B", "X"]],
            "model_b": [["B", "X"], ["B", "X"], ["B", "X"], ["B", "X"]],
            "model_c": [["B", "Y"], ["B", "Y"], ["B", "Y"], ["B", "Y"]],
        }
        result = convergence_report(responses, ["Q1", "Q2"])
        # Q1: all agree, alpha = 1.0
        assert result.findings[0].level == ConvergenceLevel.STRONG
        assert result.findings[0].top_choice_agreement is True
        # Q2: model_c diverges completely
        assert result.findings[1].alpha < 0.5
        assert "model_c" in result.findings[1].divergent_models

    def test_divergent_model_identified(self):
        """The divergent model should be named."""
        responses = {
            "gemini": [["A"]] * 5,
            "haiku": [["A"]] * 5,
            "gpt4o": [["B"]] * 5,
        }
        result = convergence_report(responses, ["Q1"])
        assert "gpt4o" in result.findings[0].divergent_models
        assert "gemini" not in result.findings[0].divergent_models

    def test_per_model_distributions(self):
        """Per-model distribution should reflect actual response proportions."""
        responses = {
            "model_a": [["X"], ["X"], ["Y"], ["X"]],
            "model_b": [["X"], ["Y"], ["Y"], ["Y"]],
        }
        result = convergence_report(responses, ["Q1"])
        for md in result.findings[0].per_model:
            if md.model == "model_a":
                assert md.distribution["X"] == pytest.approx(0.75)
                assert md.top_choice == "X"
            if md.model == "model_b":
                assert md.distribution["Y"] == pytest.approx(0.75)
                assert md.top_choice == "Y"

    def test_ordinal_level(self):
        """Should pass ordinal level_of_measurement through to alpha."""
        responses = {
            "m1": [["1", "2"], ["1", "2"], ["1", "2"]],
            "m2": [["1", "2"], ["1", "2"], ["2", "1"]],
        }
        result = convergence_report(
            responses,
            ["Q1", "Q2"],
            level_of_measurement="ordinal",
        )
        assert isinstance(result.overall_alpha, float)

    def test_fewer_than_2_models_rejects(self):
        with pytest.raises(ValueError, match="model"):
            convergence_report({"only_one": [["A"]]}, ["Q1"])

    def test_inconsistent_n_rejects(self):
        """Models must have the same number of personas."""
        with pytest.raises(ValueError):
            convergence_report(
                {"m1": [["A"], ["A"]], "m2": [["A"]]},
                ["Q1"],
            )

    def test_inconsistent_q_rejects(self):
        """Models must have the same number of questions per persona."""
        with pytest.raises(ValueError):
            convergence_report(
                {"m1": [["A", "B"]], "m2": [["A"]]},
                ["Q1", "Q2"],
            )

    def test_question_texts_length_mismatch_rejects(self):
        with pytest.raises(ValueError):
            convergence_report(
                {"m1": [["A"]], "m2": [["A"]]},
                ["Q1", "Q2"],  # 2 texts but only 1 question in data
            )

    def test_report_metadata(self):
        responses = {
            "gemini": [["A"], ["B"]],
            "haiku": [["A"], ["B"]],
        }
        result = convergence_report(responses, ["Q1"])
        assert result.n_models == 2
        assert set(result.model_names) == {"gemini", "haiku"}


# ---------------------------------------------------------------------------
# cluster_personas (sp-5on.9)
# ---------------------------------------------------------------------------


class TestClusterPersonas:
    def test_two_obvious_clusters(self):
        """Two groups with completely different response patterns."""
        responses = {
            "A1": ["X", "X", "X"],
            "A2": ["X", "X", "X"],
            "A3": ["X", "X", "X"],
            "A4": ["X", "X", "X"],
            "B1": ["Y", "Y", "Y"],
            "B2": ["Y", "Y", "Y"],
            "B3": ["Y", "Y", "Y"],
            "B4": ["Y", "Y", "Y"],
        }
        result = cluster_personas(responses, min_k=2, max_k=3)
        assert result.n_clusters == 2
        assert result.silhouette_score > 0.5
        # A-group and B-group should be in different clusters
        assert result.persona_assignments["A1"] == result.persona_assignments["A2"]
        assert result.persona_assignments["B1"] == result.persona_assignments["B2"]
        assert result.persona_assignments["A1"] != result.persona_assignments["B1"]

    def test_uniform_responses_low_silhouette(self):
        """All personas respond identically -> no cluster structure."""
        responses = {f"P{i}": ["A", "B", "C"] for i in range(10)}
        result = cluster_personas(responses, min_k=2, max_k=3)
        assert result.silhouette_score <= 0.0

    def test_dominant_responses_reported(self):
        """Cluster dominant response should reflect the majority."""
        responses = {
            "A1": ["X", "Y"],
            "A2": ["X", "Y"],
            "A3": ["X", "Z"],
            "B1": ["Y", "X"],
            "B2": ["Y", "X"],
            "B3": ["Y", "X"],
        }
        result = cluster_personas(responses, min_k=2, max_k=2)
        for cluster in result.clusters:
            if "A1" in cluster.persona_names:
                assert cluster.dominant_responses[0] == "X"
            if "B1" in cluster.persona_names:
                assert cluster.dominant_responses[0] == "Y"

    def test_max_k_capped_at_n_minus_1(self):
        """max_k silently capped if it exceeds N-1."""
        responses = {f"P{i}": ["A"] for i in range(4)}
        result = cluster_personas(responses, min_k=2, max_k=10)
        assert result.k_range_tested[1] <= 3  # N-1 = 3

    def test_too_few_personas_rejects(self):
        """Need at least 2*min_k personas."""
        responses = {"P1": ["A"], "P2": ["B"]}
        with pytest.raises(ValueError):
            cluster_personas(responses, min_k=2)

    def test_inconsistent_response_lengths_rejects(self):
        responses = {"P1": ["A", "B"], "P2": ["A"]}
        with pytest.raises(ValueError):
            cluster_personas(responses)

    def test_persona_assignments_complete(self):
        """Every persona should appear in assignments."""
        responses = {f"P{i}": ["A" if i < 5 else "B"] for i in range(10)}
        result = cluster_personas(responses)
        assert set(result.persona_assignments.keys()) == set(responses.keys())


# ---------------------------------------------------------------------------
# silhouette_score (sp-5on.9)
# ---------------------------------------------------------------------------


class TestSilhouetteScore:
    def test_perfect_clusters(self):
        """Two well-separated clusters -> high silhouette."""
        labels = [0, 0, 1, 1]
        dist = [
            [0, 1, 200, 201],
            [1, 0, 201, 202],
            [200, 201, 0, 1],
            [201, 202, 1, 0],
        ]
        score = silhouette_score(labels, dist)
        assert score > 0.9

    def test_single_cluster_returns_zero(self):
        labels = [0, 0, 0]
        dist = [[0, 1, 2], [1, 0, 1], [2, 1, 0]]
        assert silhouette_score(labels, dist) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# robustness_score (sp-5on.14)
# ---------------------------------------------------------------------------


class TestRobustnessScore:
    def test_perfect_robustness(self):
        """All variants agree -> R = 1.0."""
        responses = {
            "Alice": ["B", "B", "B", "B", "B"],
            "Bob": ["B", "B", "B", "B", "B"],
        }
        result = robustness_score(responses, "B")
        assert result.overall_robustness == pytest.approx(1.0)
        assert result.interpretation == "robust"

    def test_zero_robustness(self):
        """No variants agree -> R = 0.0."""
        responses = {
            "Alice": ["A", "C", "D", "A", "C"],
            "Bob": ["C", "A", "D", "A", "C"],
        }
        result = robustness_score(responses, "B")
        assert result.overall_robustness == pytest.approx(0.0)
        assert result.interpretation == "fragile"

    def test_partial_robustness(self):
        """4/5 agree per persona -> R = 0.8."""
        responses = {
            "Alice": ["B", "B", "B", "B", "A"],
            "Bob": ["B", "B", "B", "B", "C"],
        }
        result = robustness_score(responses, "B")
        assert result.overall_robustness == pytest.approx(0.8)
        assert result.interpretation == "robust"

    def test_mixed_persona_robustness(self):
        """Different robustness per persona."""
        responses = {
            "Alice": ["B", "B", "B", "B", "B"],  # 5/5 = 1.0
            "Bob": ["B", "A", "A", "A", "A"],  # 1/5 = 0.2
        }
        result = robustness_score(responses, "B")
        assert result.overall_robustness == pytest.approx(0.6)
        assert result.per_persona["Alice"] == pytest.approx(1.0)
        assert result.per_persona["Bob"] == pytest.approx(0.2)
        assert result.interpretation == "moderately robust"

    def test_per_persona_dict(self):
        responses = {"X": ["B", "B", "A"]}
        result = robustness_score(responses, "B")
        assert "X" in result.per_persona
        assert result.per_persona["X"] == pytest.approx(2 / 3, abs=0.01)

    def test_empty_rejects(self):
        with pytest.raises(ValueError):
            robustness_score({}, "B")
