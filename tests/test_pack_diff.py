"""Tests for `synthpanel pack diff` (GH-308)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from synth_panel.main import main
from synth_panel.pack_diff import (
    _normalize_traits,
    _role_bucket,
    compute_pack_diff,
    load_pack,
    trait_delta,
)

# ---------------------------------------------------------------------------
# Synthetic v1 / v2 fixture pair
# ---------------------------------------------------------------------------

# Five overlap personas. v2 changes Alice's age + traits, removes Bob entirely,
# leaves Carol unchanged, and adds two new personas (Dawn, Erin).

_PACK_V1: dict = {
    "name": "Sample Pack",
    "version": "1",
    "description": "v1 baseline",
    "personas": [
        {
            "name": "Alice",
            "age": 30,
            "occupation": "Senior Backend Engineer",
            "background": "Builds APIs.",
            "personality_traits": ["analytical", "pragmatic"],
        },
        {
            "name": "Bob",
            "age": 41,
            "occupation": "Engineering Manager",
            "background": "Manages a team.",
            "personality_traits": ["organized", "calm"],
        },
        {
            "name": "Carol",
            "age": 25,
            "occupation": "Junior Frontend Developer",
            "background": "Bootcamp grad.",
            "personality_traits": ["eager", "curious"],
        },
    ],
}

_PACK_V2: dict = {
    "name": "Sample Pack",
    "version": "2",
    "description": "v2 with one age tweak, Bob removed, two new personas",
    "personas": [
        {
            "name": "Alice",
            "age": 31,  # +1 from v1
            "occupation": "Senior Backend Engineer",
            "background": "Builds APIs.",
            "personality_traits": ["analytical", "pragmatic", "skeptical"],
        },
        {
            "name": "Carol",
            "age": 25,
            "occupation": "Junior Frontend Developer",
            "background": "Bootcamp grad.",
            "personality_traits": ["eager", "curious"],
        },
        {
            "name": "Dawn",
            "age": 38,
            "occupation": "Staff DevOps Engineer",
            "background": "Maintains the platform.",
            "personality_traits": ["meticulous"],
        },
        {
            "name": "Erin",
            "age": 28,
            "occupation": "ML Engineer",
            "background": "Trains models.",
            "personality_traits": ["experimental"],
        },
    ],
}


# ---------------------------------------------------------------------------
# compute_pack_diff
# ---------------------------------------------------------------------------


class TestComputePackDiff:
    def test_added_removed_unchanged_changed_buckets(self) -> None:
        diff = compute_pack_diff(_PACK_V1, _PACK_V2, pack_a_id="v1", pack_b_id="v2")
        assert diff.added == ["Dawn", "Erin"]
        assert diff.removed == ["Bob"]
        assert diff.unchanged == ["Carol"]
        assert [c.name for c in diff.changed] == ["Alice"]

    def test_alice_field_diff(self) -> None:
        diff = compute_pack_diff(_PACK_V1, _PACK_V2)
        alice = next(c for c in diff.changed if c.name == "Alice")
        assert "age" in alice.changed
        assert alice.changed["age"] == {"a": 30, "b": 31}
        assert "personality_traits" in alice.changed
        added, removed = trait_delta(alice)
        assert added == ["skeptical"]
        assert removed == []
        # Background and occupation are unchanged → not in diff
        assert "background" not in alice.changed
        assert "occupation" not in alice.changed

    def test_composition_age_stats(self) -> None:
        diff = compute_pack_diff(_PACK_V1, _PACK_V2)
        a, b = diff.composition_a, diff.composition_b
        assert a.persona_count == 3
        assert b.persona_count == 4
        assert a.age_min == 25 and a.age_max == 41
        assert b.age_min == 25 and b.age_max == 38
        # 3 personas, mean=(30+41+25)/3=32.0
        assert a.age_mean == 32.0
        # 4 personas, mean=(31+25+38+28)/4=30.5
        assert b.age_mean == 30.5

    def test_role_distribution_drops_seniority(self) -> None:
        diff = compute_pack_diff(_PACK_V1, _PACK_V2)
        # "Senior Backend Engineer" -> "backend"
        assert diff.composition_a.role_distribution.get("backend") == 1
        # "Junior Frontend Developer" -> "frontend"
        assert diff.composition_a.role_distribution.get("frontend") == 1
        # v2 has "Staff DevOps Engineer" (seniority dropped) -> "devops"
        assert diff.composition_b.role_distribution.get("devops") == 1
        # "ML Engineer" has no seniority prefix -> "ml"
        assert diff.composition_b.role_distribution.get("ml") == 1

    def test_gender_split_empty_without_field(self) -> None:
        # Neither fixture has a gender field; gender_split must be empty.
        diff = compute_pack_diff(_PACK_V1, _PACK_V2)
        assert diff.composition_a.gender_split == {}
        assert diff.composition_b.gender_split == {}

    def test_gender_split_populated_when_present(self) -> None:
        a = {"personas": [{"name": "X", "gender": "f"}, {"name": "Y", "gender": "m"}]}
        b = {"personas": [{"name": "Z", "gender": "f"}]}
        diff = compute_pack_diff(a, b)
        assert diff.composition_a.gender_split == {"f": 1, "m": 1}
        assert diff.composition_b.gender_split == {"f": 1}

    def test_self_diff_yields_only_unchanged(self) -> None:
        diff = compute_pack_diff(_PACK_V1, _PACK_V1)
        assert diff.added == []
        assert diff.removed == []
        assert diff.changed == []
        assert sorted(diff.unchanged) == ["Alice", "Bob", "Carol"]

    def test_empty_packs(self) -> None:
        diff = compute_pack_diff({"personas": []}, {"personas": []})
        assert diff.added == diff.removed == diff.unchanged == diff.changed == []
        assert diff.composition_a.age_min is None
        assert diff.composition_b.age_mean is None

    def test_traits_normalized_so_casing_does_not_flag_change(self) -> None:
        a = {"personas": [{"name": "X", "personality_traits": ["Curious", " Eager"]}]}
        b = {"personas": [{"name": "X", "personality_traits": ["curious", "eager"]}]}
        diff = compute_pack_diff(a, b)
        assert diff.changed == []
        assert diff.unchanged == ["X"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_normalize_traits_handles_string_list_and_none(self) -> None:
        assert _normalize_traits(None) == []
        assert _normalize_traits(["B", "a "]) == ["a", "b"]
        assert _normalize_traits("x, Y , z") == ["x", "y", "z"]
        assert _normalize_traits(123) == []

    def test_role_bucket(self) -> None:
        assert _role_bucket("Senior Backend Engineer") == "backend"
        assert _role_bucket("Junior Frontend Developer") == "frontend"
        assert _role_bucket("Engineering Manager") == "engineering"
        assert _role_bucket("DevOps / Platform Engineer") == "devops"
        # All-seniority input falls back to first token
        assert _role_bucket("Senior") == "senior"


# ---------------------------------------------------------------------------
# load_pack
# ---------------------------------------------------------------------------


class TestLoadPack:
    def test_load_from_file_path(self, tmp_path: Path) -> None:
        path = tmp_path / "mypack.yaml"
        path.write_text(yaml.safe_dump(_PACK_V1), encoding="utf-8")
        data, label = load_pack(str(path))
        assert label == "mypack"
        assert data["personas"][0]["name"] == "Alice"

    def test_load_from_builtin_pack_id(self) -> None:
        data, label = load_pack("developer")
        assert label == "developer"
        # bundled developer pack ships with at least 5 personas
        assert isinstance(data["personas"], list)
        assert len(data["personas"]) >= 5

    def test_missing_id_raises_filenotfound(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pack("definitely-not-a-real-pack-xyz")

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_pack(str(path))


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestPackDiffCLI:
    def _write_v1_v2(self, tmp_path: Path) -> tuple[Path, Path]:
        a = tmp_path / "pack_v1.yaml"
        b = tmp_path / "pack_v2.yaml"
        a.write_text(yaml.safe_dump(_PACK_V1), encoding="utf-8")
        b.write_text(yaml.safe_dump(_PACK_V2), encoding="utf-8")
        return a, b

    def test_text_mode_smoke(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        a, b = self._write_v1_v2(tmp_path)
        rc = main(["pack", "diff", str(a), str(b)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Pack A" in out and "Pack B" in out
        assert "Added in B:    2" in out
        assert "Removed in B:  1" in out
        assert "Unchanged:     1" in out
        assert "Changed:       1" in out
        assert "Dawn" in out and "Erin" in out
        assert "Bob" in out  # listed in removed
        # Trait delta surfaces "skeptical"
        assert "skeptical" in out

    def test_format_json_emits_machine_payload(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        a, b = self._write_v1_v2(tmp_path)
        rc = main(["pack", "diff", str(a), str(b), "--format", "json"])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["added"] == ["Dawn", "Erin"]
        assert payload["removed"] == ["Bob"]
        assert payload["unchanged"] == ["Carol"]
        assert len(payload["changed"]) == 1
        alice = payload["changed"][0]
        assert alice["name"] == "Alice"
        assert alice["fields"]["age"] == {"a": 30, "b": 31}
        traits = alice["fields"]["personality_traits"]
        assert traits["added"] == ["skeptical"]
        assert traits["removed"] == []
        # Composition stats included
        assert payload["pack_a"]["composition"]["persona_count"] == 3
        assert payload["pack_b"]["composition"]["persona_count"] == 4

    def test_global_output_format_json_works(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        # The global --output-format json should also produce structured data
        # (via emit() / NDJSON-style payload merging).
        a, b = self._write_v1_v2(tmp_path)
        rc = main(["--output-format", "json", "pack", "diff", str(a), str(b)])
        assert rc == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["message"] == "pack_diff"
        assert payload["added"] == ["Dawn", "Erin"]

    def test_accepts_builtin_name_for_one_side(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        a, _ = self._write_v1_v2(tmp_path)
        # Mix file path + bundled name (against a real bundled pack).
        rc = main(["pack", "diff", str(a), "developer"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "developer" in out
        # Disjoint persona sets → all from a are removed, all from developer are added
        assert "Removed in B:" in out
        assert "Added in B:" in out

    def test_missing_pack_returns_exit_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        a, _ = self._write_v1_v2(tmp_path)
        rc = main(["pack", "diff", str(a), "definitely-not-a-real-pack"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Persona pack not found" in err

    def test_invalid_yaml_returns_exit_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(": :\n- nope\n", encoding="utf-8")
        a, _ = self._write_v1_v2(tmp_path)
        rc = main(["pack", "diff", str(a), str(bad)])
        assert rc == 1
