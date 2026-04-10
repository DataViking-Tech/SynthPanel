"""Tests for synth_panel.aggregation -- sp-5on.16."""

from __future__ import annotations

import pytest

from synth_panel.aggregation import (
    VariantGroup,
    aggregate_variants,
    robustness_report,
)
from synth_panel.cost import ZERO_USAGE
from synth_panel.orchestrator import PanelistResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pr(name: str, responses: list[str]) -> PanelistResult:
    """Build a PanelistResult with simple text responses."""
    return PanelistResult(
        persona_name=name,
        responses=[{"question": f"Q{i}", "response": r} for i, r in enumerate(responses)],
        usage=ZERO_USAGE,
    )


def _persona(name: str, variant_of: str | None = None) -> dict:
    """Build a minimal persona dict."""
    p: dict = {"name": name, "age": 30}
    if variant_of is not None:
        p["_variant_of"] = variant_of
    return p


# ---------------------------------------------------------------------------
# aggregate_variants
# ---------------------------------------------------------------------------


class TestAggregateVariants:
    def test_groups_by_variant_of(self):
        """Variants grouped under their source persona."""
        personas = [
            _persona("Alice"),
            _persona("Alice (v0)", variant_of="Alice"),
            _persona("Alice (v1)", variant_of="Alice"),
            _persona("Bob"),
            _persona("Bob (v0)", variant_of="Bob"),
        ]
        results = [
            _pr("Alice", ["A"]),
            _pr("Alice (v0)", ["A"]),
            _pr("Alice (v1)", ["B"]),
            _pr("Bob", ["C"]),
            _pr("Bob (v0)", ["C"]),
        ]

        groups = aggregate_variants(results, personas)

        assert len(groups) == 2
        alice_group = next(g for g in groups if g.source_name == "Alice")
        bob_group = next(g for g in groups if g.source_name == "Bob")

        assert alice_group.original is not None
        assert alice_group.original.persona_name == "Alice"
        assert alice_group.k == 2
        assert bob_group.k == 1

    def test_sorted_by_source_name(self):
        """Groups are in alphabetical order by source name."""
        personas = [
            _persona("Zara"),
            _persona("Zara (v0)", variant_of="Zara"),
            _persona("Alice"),
            _persona("Alice (v0)", variant_of="Alice"),
        ]
        results = [
            _pr("Zara", ["X"]),
            _pr("Zara (v0)", ["X"]),
            _pr("Alice", ["Y"]),
            _pr("Alice (v0)", ["Y"]),
        ]

        groups = aggregate_variants(results, personas)
        assert [g.source_name for g in groups] == ["Alice", "Zara"]

    def test_excludes_personas_without_variants(self):
        """Originals without any variants are not included."""
        personas = [
            _persona("Alice"),
            _persona("Bob"),
            _persona("Bob (v0)", variant_of="Bob"),
        ]
        results = [
            _pr("Alice", ["A"]),
            _pr("Bob", ["B"]),
            _pr("Bob (v0)", ["B"]),
        ]

        groups = aggregate_variants(results, personas)
        assert len(groups) == 1
        assert groups[0].source_name == "Bob"

    def test_variant_without_original_result(self):
        """Variants whose original was not run still form a group."""
        personas = [
            _persona("Alice (v0)", variant_of="Alice"),
            _persona("Alice (v1)", variant_of="Alice"),
        ]
        results = [
            _pr("Alice (v0)", ["A"]),
            _pr("Alice (v1)", ["B"]),
        ]

        groups = aggregate_variants(results, personas)
        assert len(groups) == 1
        assert groups[0].source_name == "Alice"
        assert groups[0].original is None
        assert groups[0].k == 2

    def test_empty_results(self):
        """Empty inputs produce no groups."""
        assert aggregate_variants([], []) == []


# ---------------------------------------------------------------------------
# robustness_report
# ---------------------------------------------------------------------------


class TestRobustnessReport:
    def test_perfect_robustness(self):
        """All variants agree with originals -> R = 1.0, robust."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A", "B"]),
                variants=[_pr("v0", ["A", "B"]), _pr("v1", ["A", "B"])],
            ),
            VariantGroup(
                source_name="Bob",
                original=_pr("Bob", ["C", "D"]),
                variants=[_pr("v0", ["C", "D"]), _pr("v1", ["C", "D"])],
            ),
        ]

        report = robustness_report(groups)

        assert report.aggregate_robustness == pytest.approx(1.0)
        assert report.n_personas == 2
        assert len(report.findings) == 2
        assert all(f.classification == "robust" for f in report.findings)
        assert report.fragile_findings == []
        assert report.per_persona_robustness["Alice"] == pytest.approx(1.0)
        assert report.per_persona_robustness["Bob"] == pytest.approx(1.0)

    def test_zero_robustness(self):
        """No variants agree -> R = 0.0, fragile."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A"]),
                variants=[_pr("v0", ["X"]), _pr("v1", ["Y"])],
            ),
            VariantGroup(
                source_name="Bob",
                original=_pr("Bob", ["B"]),
                variants=[_pr("v0", ["X"]), _pr("v1", ["Y"])],
            ),
        ]

        report = robustness_report(groups)

        assert report.aggregate_robustness == pytest.approx(0.0)
        assert all(f.classification == "fragile" for f in report.findings)
        assert len(report.fragile_findings) == 1

    def test_known_answer_mixed(self):
        """Known-answer: Alice 2/3, Bob 1/3 -> R = 0.5 (sensitive)."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A"]),
                variants=[_pr("v0", ["A"]), _pr("v1", ["A"]), _pr("v2", ["B"])],
            ),
            VariantGroup(
                source_name="Bob",
                original=_pr("Bob", ["A"]),
                variants=[_pr("v0", ["A"]), _pr("v1", ["B"]), _pr("v2", ["B"])],
            ),
        ]

        report = robustness_report(groups)

        # Alice: 2/3, Bob: 1/3, mean = (2/3 + 1/3) / 2 = 0.5
        assert report.findings[0].per_persona["Alice"] == pytest.approx(2 / 3, abs=0.001)
        assert report.findings[0].per_persona["Bob"] == pytest.approx(1 / 3, abs=0.001)
        assert report.findings[0].score == pytest.approx(0.5)
        assert report.findings[0].classification == "sensitive"
        assert len(report.fragile_findings) == 1

    def test_multiple_questions(self):
        """Robustness computed independently per question."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A", "X"]),
                variants=[
                    _pr("v0", ["A", "Y"]),  # Q0 agree, Q1 disagree
                    _pr("v1", ["A", "X"]),  # Q0 agree, Q1 agree
                ],
            ),
        ]

        report = robustness_report(groups)

        assert len(report.findings) == 2
        # Q0: 2/2 = 1.0
        assert report.findings[0].score == pytest.approx(1.0)
        assert report.findings[0].classification == "robust"
        # Q1: 1/2 = 0.5
        assert report.findings[1].score == pytest.approx(0.5)
        assert report.findings[1].classification == "sensitive"
        # Aggregate: (1.0 + 0.5) / 2 = 0.75
        assert report.aggregate_robustness == pytest.approx(0.75)

    def test_per_persona_robustness_is_mean_across_questions(self):
        """Per-persona robustness averages across all questions."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A", "B", "C"]),
                variants=[
                    _pr("v0", ["A", "X", "C"]),  # agree, disagree, agree
                ],
            ),
        ]

        report = robustness_report(groups)

        # Alice: Q0=1.0, Q1=0.0, Q2=1.0 -> mean = 2/3
        assert report.per_persona_robustness["Alice"] == pytest.approx(2 / 3, abs=0.001)

    def test_classification_thresholds(self):
        """Verify all four classification tiers."""

        # Build groups where each question produces a specific score
        def _group_with_score(n_agree: int, k: int) -> list[VariantGroup]:
            """One persona, one question, n_agree of k variants agree."""
            variants = []
            for i in range(k):
                val = "A" if i < n_agree else "X"
                variants.append(_pr(f"v{i}", [val]))
            return [
                VariantGroup(
                    source_name="P",
                    original=_pr("P", ["A"]),
                    variants=variants,
                )
            ]

        # R = 1.0 -> robust
        r = robustness_report(_group_with_score(10, 10))
        assert r.findings[0].classification == "robust"

        # R = 0.8 -> robust (boundary)
        r = robustness_report(_group_with_score(8, 10))
        assert r.findings[0].classification == "robust"

        # R = 0.7 -> moderate
        r = robustness_report(_group_with_score(7, 10))
        assert r.findings[0].classification == "moderate"

        # R = 0.6 -> moderate (boundary)
        r = robustness_report(_group_with_score(6, 10))
        assert r.findings[0].classification == "moderate"

        # R = 0.5 -> sensitive
        r = robustness_report(_group_with_score(5, 10))
        assert r.findings[0].classification == "sensitive"

        # R = 0.4 -> sensitive (boundary)
        r = robustness_report(_group_with_score(4, 10))
        assert r.findings[0].classification == "sensitive"

        # R = 0.3 -> fragile
        r = robustness_report(_group_with_score(3, 10))
        assert r.findings[0].classification == "fragile"

        # R = 0.0 -> fragile
        r = robustness_report(_group_with_score(0, 10))
        assert r.findings[0].classification == "fragile"

    def test_fragile_threshold_at_0_6(self):
        """fragile_findings includes scores < 0.6 (sensitive + fragile)."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["A", "B", "C"]),
                variants=[
                    # Q0: agree (1.0), Q1: disagree (0.0), Q2: agree (1.0)
                    _pr("v0", ["A", "X", "C"]),
                ],
            ),
        ]

        report = robustness_report(groups)

        assert len(report.fragile_findings) == 1
        assert report.fragile_findings[0].question_index == 1

    def test_custom_response_extractor(self):
        """Custom extractor used for comparison."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=PanelistResult(
                    persona_name="Alice",
                    responses=[{"question": "Q0", "response": {"theme": "price"}}],
                    usage=ZERO_USAGE,
                ),
                variants=[
                    PanelistResult(
                        persona_name="v0",
                        responses=[{"question": "Q0", "response": {"theme": "price"}}],
                        usage=ZERO_USAGE,
                    ),
                    PanelistResult(
                        persona_name="v1",
                        responses=[{"question": "Q0", "response": {"theme": "quality"}}],
                        usage=ZERO_USAGE,
                    ),
                ],
            ),
        ]

        def extract_theme(resp: dict) -> str:
            r = resp.get("response", {})
            return r.get("theme", "") if isinstance(r, dict) else str(r)

        report = robustness_report(groups, response_extractor=extract_theme)

        assert report.findings[0].per_persona["Alice"] == pytest.approx(0.5)

    def test_empty_groups_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            robustness_report([])

    def test_no_scorable_groups_raises(self):
        """Groups without originals cannot be scored."""
        groups = [
            VariantGroup(
                source_name="Alice",
                original=None,
                variants=[_pr("v0", ["A"])],
            ),
        ]
        with pytest.raises(ValueError, match="no groups with both"):
            robustness_report(groups)

    def test_two_level_nested_formula(self):
        """Verify the exact two-level formula from the spec.

        R(F) = (1/N) * sum_i[(1/K_i) * sum_j 1{variant_ij agrees with F}]

        Setup: 3 personas, varying K and agreement:
          Alice: K=4, 3 agree -> 3/4 = 0.75
          Bob:   K=2, 1 agrees -> 1/2 = 0.50
          Carol: K=3, 3 agree -> 3/3 = 1.00
        R = (0.75 + 0.50 + 1.00) / 3 = 0.75
        """
        groups = [
            VariantGroup(
                source_name="Alice",
                original=_pr("Alice", ["yes"]),
                variants=[_pr(f"a{i}", ["yes"]) for i in range(3)] + [_pr("a3", ["no"])],
            ),
            VariantGroup(
                source_name="Bob",
                original=_pr("Bob", ["yes"]),
                variants=[_pr("b0", ["yes"]), _pr("b1", ["no"])],
            ),
            VariantGroup(
                source_name="Carol",
                original=_pr("Carol", ["yes"]),
                variants=[_pr(f"c{i}", ["yes"]) for i in range(3)],
            ),
        ]

        report = robustness_report(groups)

        assert report.findings[0].per_persona["Alice"] == pytest.approx(0.75)
        assert report.findings[0].per_persona["Bob"] == pytest.approx(0.50)
        assert report.findings[0].per_persona["Carol"] == pytest.approx(1.0)
        assert report.findings[0].score == pytest.approx(0.75)
        assert report.n_personas == 3


# ---------------------------------------------------------------------------
# Integration: aggregate_variants -> robustness_report
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self):
        """aggregate_variants -> robustness_report end-to-end."""
        personas = [
            _persona("Alice"),
            _persona("Alice (v0)", variant_of="Alice"),
            _persona("Alice (v1)", variant_of="Alice"),
            _persona("Bob"),
            _persona("Bob (v0)", variant_of="Bob"),
            _persona("Bob (v1)", variant_of="Bob"),
        ]
        results = [
            _pr("Alice", ["yes", "10"]),
            _pr("Alice (v0)", ["yes", "10"]),
            _pr("Alice (v1)", ["yes", "8"]),
            _pr("Bob", ["no", "5"]),
            _pr("Bob (v0)", ["no", "5"]),
            _pr("Bob (v1)", ["yes", "5"]),
        ]

        groups = aggregate_variants(results, personas)
        report = robustness_report(groups)

        assert report.n_personas == 2
        assert len(report.findings) == 2

        # Q0: Alice 2/2=1.0, Bob 1/2=0.5 -> mean 0.75
        assert report.findings[0].score == pytest.approx(0.75)
        # Q1: Alice 1/2=0.5, Bob 2/2=1.0 -> mean 0.75
        assert report.findings[1].score == pytest.approx(0.75)
        # Aggregate: (0.75 + 0.75) / 2 = 0.75
        assert report.aggregate_robustness == pytest.approx(0.75)
