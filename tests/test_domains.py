"""Tests for domain prompt templates and persona diversity validation."""

from __future__ import annotations

import pytest

from synth_panel.domains import (
    DomainNotFoundError,
    get_domain_template,
    list_domain_templates,
    validate_persona_diversity,
)

# ---------------------------------------------------------------------------
# Domain template registry
# ---------------------------------------------------------------------------


class TestGetDomainTemplate:
    def test_known_domain(self):
        t = get_domain_template("career-tech")
        assert t["name"] == "Career Tech Workers"
        assert "template" in t
        assert len(t["template"]) > 0

    def test_all_domains_loadable(self):
        for entry in list_domain_templates():
            t = get_domain_template(entry["name"])
            assert "template" in t
            assert "description" in t

    def test_unknown_raises(self):
        with pytest.raises(DomainNotFoundError, match="nonexistent"):
            get_domain_template("nonexistent")

    def test_error_lists_known(self):
        with pytest.raises(DomainNotFoundError, match="career-tech"):
            get_domain_template("nope")


class TestListDomainTemplates:
    def test_returns_at_least_three(self):
        templates = list_domain_templates()
        assert len(templates) >= 3

    def test_entries_have_required_keys(self):
        for entry in list_domain_templates():
            assert "name" in entry
            assert "description" in entry

    def test_sorted_by_name(self):
        templates = list_domain_templates()
        names = [t["name"] for t in templates]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Persona diversity validation
# ---------------------------------------------------------------------------


def _persona(
    name: str,
    age: int = 30,
    traits: list[str] | None = None,
    background: str = "Unique background",
) -> dict:
    return {
        "name": name,
        "age": age,
        "personality_traits": traits or ["trait-a"],
        "background": background,
    }


class TestValidatePersonaDiversity:
    def test_diverse_panel_no_warnings(self):
        personas = [
            _persona("A", age=23, traits=["curious", "eager"], background="Bootcamp grad"),
            _persona("B", age=35, traits=["analytical", "skeptical"], background="Senior engineer"),
            _persona("C", age=50, traits=["strategic", "patient"], background="VP of engineering"),
        ]
        warnings = validate_persona_diversity(personas)
        assert warnings == []

    def test_single_persona_no_warnings(self):
        warnings = validate_persona_diversity([_persona("Solo")])
        assert warnings == []

    def test_empty_panel_no_warnings(self):
        warnings = validate_persona_diversity([])
        assert warnings == []


class TestAgeDiversity:
    def test_same_age_bracket_warns(self):
        personas = [
            _persona("A", age=26, background="bg1"),
            _persona("B", age=28, background="bg2"),
            _persona("C", age=30, background="bg3"),
        ]
        warnings = validate_persona_diversity(personas)
        assert any("age diversity" in w.lower() for w in warnings)

    def test_spread_ages_ok(self):
        personas = [
            _persona("A", age=22, background="bg1"),
            _persona("B", age=40, background="bg2"),
        ]
        warnings = validate_persona_diversity(personas)
        assert not any("age diversity" in w.lower() for w in warnings)

    def test_no_ages_warns_for_larger_panels(self):
        personas = [
            {"name": "A", "background": "bg1"},
            {"name": "B", "background": "bg2"},
            {"name": "C", "background": "bg3"},
        ]
        warnings = validate_persona_diversity(personas)
        assert any("no age data" in w.lower() for w in warnings)

    def test_no_ages_skipped_for_two_personas(self):
        personas = [
            {"name": "A", "background": "bg1"},
            {"name": "B", "background": "bg2"},
        ]
        warnings = validate_persona_diversity(personas)
        assert not any("no age data" in w.lower() for w in warnings)


class TestTraitDiversity:
    def test_identical_traits_warns(self):
        personas = [
            _persona("A", traits=["analytical", "detail-oriented"], background="bg1"),
            _persona("B", traits=["analytical", "detail-oriented"], background="bg2"),
            _persona("C", traits=["analytical", "detail-oriented"], background="bg3"),
        ]
        warnings = validate_persona_diversity(personas)
        assert any("trait overlap" in w.lower() for w in warnings)

    def test_distinct_traits_ok(self):
        personas = [
            _persona("A", traits=["creative", "bold"], background="bg1"),
            _persona("B", traits=["analytical", "cautious"], background="bg2"),
            _persona("C", traits=["empathetic", "patient"], background="bg3"),
        ]
        warnings = validate_persona_diversity(personas)
        assert not any("trait overlap" in w.lower() for w in warnings)

    def test_trait_string_normalized(self):
        """CSV-style trait strings are parsed and compared."""
        personas = [
            _persona("A", background="bg1"),
            _persona("B", background="bg2"),
        ]
        personas[0]["personality_traits"] = "creative, bold"
        personas[1]["personality_traits"] = "creative, bold"
        warnings = validate_persona_diversity(personas)
        assert any("trait overlap" in w.lower() for w in warnings)


class TestBackgroundVariety:
    def test_identical_backgrounds_warns(self):
        personas = [
            _persona("A", age=25, background="Software engineer at a startup"),
            _persona("B", age=40, background="Software engineer at a startup"),
        ]
        warnings = validate_persona_diversity(personas)
        assert any("background variety" in w.lower() for w in warnings)

    def test_distinct_backgrounds_ok(self):
        personas = [
            _persona("A", age=25, background="Junior designer at agency"),
            _persona("B", age=40, background="VP of sales at enterprise company"),
        ]
        warnings = validate_persona_diversity(personas)
        assert not any("background variety" in w.lower() for w in warnings)


class TestCustomThresholds:
    def test_strict_age_brackets(self):
        personas = [
            _persona("A", age=25, background="bg1"),
            _persona("B", age=33, background="bg2"),
        ]
        # Default min_age_brackets=2 should pass since 25-34 and 25-34... wait
        # 25 is in "25-34", 33 is in "25-34" — same bracket
        warnings = validate_persona_diversity(personas, min_age_brackets=2)
        assert any("age diversity" in w.lower() for w in warnings)

    def test_relaxed_trait_overlap(self):
        """With max_trait_overlap=1.0, even identical traits don't warn."""
        personas = [
            _persona("A", traits=["same"], background="bg1"),
            _persona("B", traits=["same"], background="bg2"),
        ]
        warnings = validate_persona_diversity(personas, max_trait_overlap=1.0)
        assert not any("trait overlap" in w.lower() for w in warnings)


class TestDeliberatelyHomogeneous:
    """Integration: a panel that fails all three diversity checks."""

    def test_all_same(self):
        personas = [
            {
                "name": f"Clone-{i}",
                "age": 30,
                "personality_traits": ["analytical", "detail-oriented"],
                "background": "Senior software engineer at a tech startup",
            }
            for i in range(5)
        ]
        warnings = validate_persona_diversity(personas)
        assert len(warnings) >= 2  # age + background at minimum
        categories = " ".join(warnings).lower()
        assert "age diversity" in categories
        assert "background variety" in categories


# ---------------------------------------------------------------------------
# New domain templates (sp-ils5)
# ---------------------------------------------------------------------------


def _assert_domain_shape(key: str) -> None:
    t = get_domain_template(key)
    assert isinstance(t["name"], str) and t["name"]
    assert isinstance(t["description"], str) and t["description"]
    assert isinstance(t["template"], str) and len(t["template"]) > 50


class TestNewDomainTemplates:
    def test_healthcare_providers_shape(self):
        _assert_domain_shape("healthcare-providers")

    def test_education_k12_shape(self):
        _assert_domain_shape("education-K12")

    def test_smb_owners_shape(self):
        _assert_domain_shape("smb-owners")

    def test_enterprise_buyers_shape(self):
        _assert_domain_shape("enterprise-buyers")

    def test_creators_shape(self):
        _assert_domain_shape("creators")

    def test_graduate_students_shape(self):
        _assert_domain_shape("graduate-students")

    def test_all_new_domains_in_list(self):
        keys = {t["name"] for t in list_domain_templates()}
        for expected in ("healthcare-providers", "education-K12", "smb-owners", "enterprise-buyers", "creators", "graduate-students"):
            assert expected in keys, f"domain '{expected}' missing from list"
