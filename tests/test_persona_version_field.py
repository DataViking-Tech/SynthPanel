"""Tests for the optional pack-level ``version`` field on persona packs
and the opt-in shadow warning in ``list_persona_packs`` (sp-lk3w).
"""

from __future__ import annotations

import warnings

import pytest
import yaml


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    """Point SYNTH_PANEL_DATA_DIR at a temp directory for every test."""
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))


# Import after env is set so ``_data_dir()`` picks up the temp path.
from synth_panel.mcp.data import (
    PackValidationError,
    get_persona_pack,
    list_persona_packs,
    save_persona_pack,
    validate_pack_manifest,
)


class TestValidatePackManifest:
    def test_missing_version_defaults_to_one(self):
        result = validate_pack_manifest({"name": "P"})
        assert result["version"] == "1"

    def test_explicit_version_passes_through(self):
        result = validate_pack_manifest({"name": "P", "version": "2"})
        assert result["version"] == "2"

    def test_non_string_version_rejected_int(self):
        with pytest.raises(PackValidationError, match="version must be a string"):
            validate_pack_manifest({"version": 2})

    def test_non_string_version_rejected_float(self):
        with pytest.raises(PackValidationError, match="version must be a string"):
            validate_pack_manifest({"version": 1.0})

    def test_non_string_version_rejected_list(self):
        with pytest.raises(PackValidationError, match="version must be a string"):
            validate_pack_manifest({"version": ["1"]})

    def test_non_dict_rejected(self):
        with pytest.raises(PackValidationError, match="must be a mapping"):
            validate_pack_manifest("not a dict")  # type: ignore[arg-type]

    def test_no_mutation_of_input(self):
        original = {"name": "P"}
        validate_pack_manifest(original)
        assert "version" not in original


class TestVersionRoundTrip:
    def test_save_with_version_get_returns_it(self):
        save_persona_pack("With Version", [{"name": "Alice"}], pack_id="v2-pack", version="2")
        pack = get_persona_pack("v2-pack")
        assert pack["version"] == "2"
        assert pack["name"] == "With Version"
        assert pack["id"] == "v2-pack"

    def test_save_returns_version_in_meta(self):
        meta = save_persona_pack("X", [{"name": "A"}], pack_id="meta-v", version="3")
        assert meta["version"] == "3"

    def test_save_without_version_get_defaults_to_one(self):
        save_persona_pack("No Version", [{"name": "Alice"}], pack_id="no-v")
        pack = get_persona_pack("no-v")
        assert pack["version"] == "1"

    def test_save_without_version_omits_field_in_yaml(self):
        """Backwards compat: omitting version does not write a version: key."""
        from synth_panel.mcp.data import _packs_dir

        save_persona_pack("Legacy", [{"name": "A"}], pack_id="legacy")
        raw = (_packs_dir() / "legacy.yaml").read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw)
        assert "version" not in parsed

    def test_save_with_non_string_version_rejected(self):
        with pytest.raises(PackValidationError, match="version must be a string"):
            save_persona_pack("Bad", [{"name": "A"}], pack_id="badv", version=2)  # type: ignore[arg-type]


class TestBundledPacksRegression:
    """All 9 bundled packs must load unchanged (with defaulted version="1")."""

    BUNDLED = (
        "developer",
        "enterprise-buyer",
        "general-consumer",
        "healthcare-patient",
        "startup-founder",
        "job-seekers",
        "recruiters-talent",
        "product-research",
        "ai-eval-buyers",
    )

    @pytest.mark.parametrize("pack_id", BUNDLED)
    def test_bundled_pack_loads_with_default_version(self, pack_id):
        pack = get_persona_pack(pack_id)
        assert pack["id"] == pack_id
        assert pack["version"] == "1"
        # Still has personas (existing behavior unchanged).
        assert isinstance(pack.get("personas"), list)
        assert pack["personas"]


class TestShadowWarning:
    def test_local_import_silent_when_shadowing(self):
        """Default call path does NOT warn when user shadows bundled."""
        save_persona_pack("My Devs", [{"name": "Dev"}], pack_id="developer")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list_persona_packs()
        assert not any("shadowed" in str(w.message) for w in caught)

    def test_registry_import_warns_on_shadow(self):
        """Opt-in flag fires a UserWarning per shadowed bundled pack."""
        save_persona_pack("My Devs", [{"name": "Dev"}], pack_id="developer")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list_persona_packs(warn_on_shadow=True)
        shadow_warnings = [w for w in caught if "shadowed" in str(w.message)]
        assert len(shadow_warnings) == 1
        assert shadow_warnings[0].category is UserWarning
        assert "developer" in str(shadow_warnings[0].message)

    def test_registry_import_silent_when_no_shadow(self):
        """Opt-in flag does not warn when no user pack shadows anything."""
        save_persona_pack("Custom", [{"name": "A"}], pack_id="my-own-pack")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            list_persona_packs(warn_on_shadow=True)
        assert not any("shadowed" in str(w.message) for w in caught)
