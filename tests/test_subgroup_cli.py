"""Tests for the ``analyze subgroup`` CLI surface (sp-9293sj / GH #341).

Covers four layers:

* Statistical helpers — :func:`_f_sf` matches reference values and
  :func:`one_way_anova` recovers known η²/F.
* :func:`auto_bin_value` decade-binning behaviour.
* :func:`analyze_subgroup` — end-to-end report construction over a
  fixture run with a known group difference.
* The ``analyze subgroup`` CLI — text + JSON output, ``--personas``
  fallback, legacy-form backward compatibility.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from synth_panel.analysis.subgroup import (
    UnknownPersonaFieldError,
    analyze_subgroup,
    auto_bin_value,
    format_subgroup_text,
)
from synth_panel.main import _rewrite_legacy_analyze, main
from synth_panel.stats import _f_sf, one_way_anova


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


class TestFSurvivalFunction:
    @pytest.mark.parametrize(
        "f, df1, df2, expected",
        [
            # Reference values from scipy.stats.f.sf(F, df1, df2). Tolerance
            # is 1e-4 which is plenty given the continued-fraction
            # convergence we use.
            (3.0, 5, 10, 0.06556),
            (4.8, 2, 9, 0.03813),
            (1.0, 1, 1, 0.5),
            (10.0, 3, 12, 0.00135),
        ],
    )
    def test_f_sf_matches_reference(self, f, df1, df2, expected):
        assert math.isclose(_f_sf(f, df1, df2), expected, abs_tol=1e-4)

    def test_f_sf_at_zero(self):
        assert _f_sf(0.0, 5, 5) == 1.0

    def test_f_sf_negative_treated_as_zero(self):
        assert _f_sf(-1.0, 5, 5) == 1.0


class TestOneWayANOVA:
    def test_equal_means_have_zero_eta_squared(self):
        groups = {"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5], "c": [1, 2, 3, 4, 5]}
        result = one_way_anova(groups)
        assert result.eta_squared == 0.0
        assert result.f_statistic == 0.0
        assert result.p_value == 1.0

    def test_known_difference_recovers_expected_stats(self):
        # Hand-computed: SS_between = 54, SS_within = 6, η² = 54/60 = 0.9
        # F = (54/2) / (6/6) = 27. scipy.stats.f.sf(27, 2, 6) ≈ 0.001.
        groups = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
        result = one_way_anova(groups)
        assert math.isclose(result.eta_squared, 0.9, abs_tol=1e-6)
        assert math.isclose(result.f_statistic, 27.0, abs_tol=1e-6)
        assert math.isclose(result.p_value, 0.001029, abs_tol=1e-4)

    def test_single_group_reports_insufficient(self):
        result = one_way_anova({"only": [1, 2, 3]})
        assert result.insufficient_data is True
        assert result.warning is not None

    def test_singleton_groups_warn(self):
        # Each group has n=1 → df_within=0 → F undefined.
        result = one_way_anova({"a": [1], "b": [2], "c": [3]})
        assert result.insufficient_data is True
        assert result.warning and "n=1" in result.warning

    def test_zero_within_variance_yields_inf_F(self):
        result = one_way_anova({"a": [1, 1, 1], "b": [2, 2, 2]})
        assert math.isinf(result.f_statistic)
        assert result.p_value == 0.0
        assert result.warning and "within-group variance" in result.warning


# ---------------------------------------------------------------------------
# auto_bin_value
# ---------------------------------------------------------------------------


class TestAutoBinValue:
    def test_age_decade_basic(self):
        assert auto_bin_value("age_decade", {"age": 25}) == ("age", "20s")
        assert auto_bin_value("age_decade", {"age": 30}) == ("age", "30s")
        assert auto_bin_value("age_decade", {"age": 49}) == ("age", "40s")

    def test_age_decade_5y(self):
        assert auto_bin_value("age_decade_5y", {"age": 22}) == ("age", "20-24")
        assert auto_bin_value("age_decade_5y", {"age": 25}) == ("age", "25-29")
        assert auto_bin_value("age_decade_5y", {"age": 39}) == ("age", "35-39")

    def test_age_missing_returns_none_label(self):
        assert auto_bin_value("age_decade", {}) == ("age", None)
        assert auto_bin_value("age_decade", {"age": None}) == ("age", None)

    def test_age_non_numeric_rejected(self):
        # Bools and strings shouldn't accidentally bin.
        assert auto_bin_value("age_decade", {"age": True}) == ("age", None)
        assert auto_bin_value("age_decade", {"age": "old"}) == ("age", None)

    def test_direct_field_passthrough(self):
        assert auto_bin_value("role", {"role": "pm"}) == ("role", "pm")
        assert auto_bin_value("role", {}) == ("role", None)
        assert auto_bin_value("active", {"active": True}) == ("active", "true")


# ---------------------------------------------------------------------------
# analyze_subgroup — high-level report
# ---------------------------------------------------------------------------


def _scale_panel_with_group_difference() -> dict:
    """6 personas across 3 age decades, scale-typed responses with a
    monotone trend (20s high → 40s low) so the test can assert that
    age_decade explains a large share of variance."""
    return {
        "id": "result-test-subgroup-scale",
        "model": "haiku",
        "persona_count": 6,
        "question_count": 1,
        "personas": [
            {"name": "A", "age": 25, "role": "pm"},
            {"name": "B", "age": 27, "role": "pm"},
            {"name": "C", "age": 32, "role": "eng"},
            {"name": "D", "age": 35, "role": "eng"},
            {"name": "E", "age": 47, "role": "design"},
            {"name": "F", "age": 49, "role": "design"},
        ],
        "questions": [
            {
                "text": "Recommend score?",
                "response_schema": {"type": "scale", "min": 1, "max": 5},
            }
        ],
        "results": [
            {"persona": "A", "responses": [{"question": "Recommend score?", "response": 5}]},
            {"persona": "B", "responses": [{"question": "Recommend score?", "response": 4}]},
            {"persona": "C", "responses": [{"question": "Recommend score?", "response": 4}]},
            {"persona": "D", "responses": [{"question": "Recommend score?", "response": 3}]},
            {"persona": "E", "responses": [{"question": "Recommend score?", "response": 2}]},
            {"persona": "F", "responses": [{"question": "Recommend score?", "response": 2}]},
        ],
        "total_usage": {"input_tokens": 0, "output_tokens": 0},
        "total_cost": "$0.00",
    }


def _enum_panel_with_group_difference() -> dict:
    """Categorical responses where role explains a strong skew —
    PMs say 'yes', engineers say 'no'."""
    return {
        "id": "result-test-subgroup-enum",
        "model": "haiku",
        "persona_count": 6,
        "question_count": 1,
        "personas": [
            {"name": "A", "role": "pm"},
            {"name": "B", "role": "pm"},
            {"name": "C", "role": "pm"},
            {"name": "D", "role": "eng"},
            {"name": "E", "role": "eng"},
            {"name": "F", "role": "eng"},
        ],
        "questions": [
            {
                "text": "Use it?",
                "response_schema": {"type": "enum", "options": ["yes", "no"]},
            }
        ],
        "results": [
            {"persona": "A", "responses": [{"question": "Use it?", "response": "yes"}]},
            {"persona": "B", "responses": [{"question": "Use it?", "response": "yes"}]},
            {"persona": "C", "responses": [{"question": "Use it?", "response": "yes"}]},
            {"persona": "D", "responses": [{"question": "Use it?", "response": "no"}]},
            {"persona": "E", "responses": [{"question": "Use it?", "response": "no"}]},
            {"persona": "F", "responses": [{"question": "Use it?", "response": "no"}]},
        ],
        "total_usage": {"input_tokens": 0, "output_tokens": 0},
        "total_cost": "$0.00",
    }


class TestAnalyzeSubgroup:
    def test_scale_response_with_age_decade(self):
        report = analyze_subgroup(_scale_panel_with_group_difference(), by="age_decade")
        assert report["field"] == "age_decade"
        assert report["source_field"] == "age"
        assert report["n_panelists"] == 6
        assert report["subgroups"] == {"20s": 2, "30s": 2, "40s": 2}

        per_q = report["per_question"]
        assert len(per_q) == 1
        q = per_q[0]
        assert q["response_type"] == "scale"
        es = q["effect_size"]
        # η² should be very large given the monotone trend in the fixture.
        assert es["metric"] == "eta_squared"
        assert es["value"] > 0.7
        assert es["label"] == "large"
        assert q["test"]["name"] == "one_way_anova"
        # Bucket means line up with the fixture: 20s=4.5, 30s=3.5, 40s=2.0.
        means = {b["label"]: b["mean"] for b in q["buckets"]}
        assert math.isclose(means["20s"], 4.5)
        assert math.isclose(means["30s"], 3.5)
        assert math.isclose(means["40s"], 2.0)

    def test_categorical_response_with_role(self):
        report = analyze_subgroup(_enum_panel_with_group_difference(), by="role")
        q = report["per_question"][0]
        assert q["response_type"] == "enum"
        es = q["effect_size"]
        # PMs all yes, engineers all no → maximal Cramer's V.
        assert es["metric"] == "cramers_v"
        assert es["value"] >= 0.99
        assert es["label"] == "large"
        assert q["test"]["name"] == "chi_squared"

    def test_sparse_subgroup_warns_but_still_reports(self):
        # 2 buckets with n=2 each → still computes η², but warns.
        panel = _scale_panel_with_group_difference()
        # Drop the 30s and 40s personas so we only have the 20s and a
        # singleton 50s case.
        panel["personas"] = panel["personas"][:2] + [{"name": "Z", "age": 50}]
        panel["results"] = panel["results"][:2] + [
            {"persona": "Z", "responses": [{"question": "Recommend score?", "response": 3}]}
        ]
        panel["persona_count"] = 3
        report = analyze_subgroup(panel, by="age_decade", min_subgroup_n=3)
        q = report["per_question"][0]
        assert q["effect_size"] is not None
        assert any("n<3" in w for w in q["warnings"])

    def test_missing_persona_attrs_raises(self):
        panel = _scale_panel_with_group_difference()
        # Strip the personas block AND the inline persona dicts so
        # nothing carries the 'age' field.
        panel["personas"] = [{"name": p["name"]} for p in panel["personas"]]
        with pytest.raises(UnknownPersonaFieldError):
            analyze_subgroup(panel, by="age_decade")

    def test_missing_field_for_some_personas_buckets_into_missing(self):
        panel = _scale_panel_with_group_difference()
        # Remove age from two personas → they go in the 'missing' bucket.
        panel["personas"][4].pop("age")
        panel["personas"][5].pop("age")
        report = analyze_subgroup(panel, by="age_decade")
        assert "missing" in report["subgroups"]
        # Missing bucket should be last in the per-question buckets too.
        labels = [b["label"] for b in report["per_question"][0]["buckets"]]
        assert labels[-1] == "missing"

    def test_format_text_renders_effect_size_and_p_value(self):
        report = analyze_subgroup(_scale_panel_with_group_difference(), by="age_decade")
        text = format_subgroup_text(report)
        assert "Effect size: η²" in text
        assert "Statistical reliability: F=" in text
        assert "20s" in text and "30s" in text and "40s" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestLegacyArgvRewrite:
    def test_legacy_form_gains_summary(self):
        rewritten = _rewrite_legacy_analyze(["analyze", "result-id"])
        assert rewritten == ["analyze", "summary", "result-id"]

    def test_legacy_form_with_options(self):
        rewritten = _rewrite_legacy_analyze(
            ["analyze", "result-id", "--output", "json"]
        )
        assert rewritten == ["analyze", "summary", "result-id", "--output", "json"]

    def test_modern_subgroup_form_left_alone(self):
        argv = ["analyze", "subgroup", "result-id", "--by", "role"]
        assert _rewrite_legacy_analyze(argv) == argv

    def test_explicit_summary_form_left_alone(self):
        argv = ["analyze", "summary", "result-id"]
        assert _rewrite_legacy_analyze(argv) == argv

    def test_help_flag_left_alone(self):
        argv = ["analyze", "--help"]
        assert _rewrite_legacy_analyze(argv) == argv

    def test_bare_analyze_left_alone(self):
        assert _rewrite_legacy_analyze(["analyze"]) == ["analyze"]

    def test_other_top_level_command_left_alone(self):
        argv = ["panel", "run", "--personas", "x.yaml"]
        assert _rewrite_legacy_analyze(argv) == argv


class TestAnalyzeSubgroupCLI:
    def _write_panel(self, panel: dict) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False)
        json.dump(panel, f)
        f.flush()
        f.close()
        return f.name

    def test_text_output_smoke(self, capsys):
        path = self._write_panel(_scale_panel_with_group_difference())
        exit_code = main(["analyze", "subgroup", path, "--by", "age_decade"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Subgroup analysis (by age_decade" in captured.out
        assert "η²" in captured.out
        assert "20s" in captured.out

    def test_json_output_is_parseable(self, capsys):
        path = self._write_panel(_scale_panel_with_group_difference())
        exit_code = main(["analyze", "subgroup", path, "--by", "age_decade", "--format", "json"])
        captured = capsys.readouterr()
        assert exit_code == 0
        parsed = json.loads(captured.out)
        assert parsed["field"] == "age_decade"
        assert parsed["per_question"][0]["effect_size"]["metric"] == "eta_squared"

    def test_legacy_flat_form_still_works(self, capsys):
        # ``analyze RESULT_ID`` (no subcommand) must still route to the
        # original analyze handler thanks to the argv rewrite.
        from tests.test_analyze import _make_panel_result

        path = self._write_panel(_make_panel_result())
        exit_code = main(["analyze", path])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "DESCRIPTIVE STATISTICS" in captured.out

    def test_missing_personas_block_yields_helpful_error(self, capsys):
        panel = _scale_panel_with_group_difference()
        # Strip the personas block and any inline attrs.
        panel["personas"] = [{"name": p["name"]} for p in panel["personas"]]
        path = self._write_panel(panel)
        exit_code = main(["analyze", "subgroup", path, "--by", "age_decade"])
        captured = capsys.readouterr()
        assert exit_code == 1
        # The hint specifically points at --personas because the
        # personas block is absent — not just incomplete.
        assert "missing" in captured.err.lower()

    def test_personas_yaml_fallback(self, capsys, tmp_path: Path):
        # Old-style result with no personas block; a --personas YAML
        # rescues it.
        panel = _scale_panel_with_group_difference()
        panel.pop("personas")
        result_path = self._write_panel(panel)

        personas_yaml = tmp_path / "personas.yaml"
        personas_yaml.write_text(
            "personas:\n"
            "  - name: A\n    age: 25\n"
            "  - name: B\n    age: 27\n"
            "  - name: C\n    age: 32\n"
            "  - name: D\n    age: 35\n"
            "  - name: E\n    age: 47\n"
            "  - name: F\n    age: 49\n",
            encoding="utf-8",
        )
        exit_code = main(
            [
                "analyze",
                "subgroup",
                result_path,
                "--by",
                "age_decade",
                "--personas",
                str(personas_yaml),
                "--format",
                "json",
            ]
        )
        captured = capsys.readouterr()
        assert exit_code == 0
        parsed = json.loads(captured.out)
        assert set(parsed["subgroups"].keys()) == {"20s", "30s", "40s"}
