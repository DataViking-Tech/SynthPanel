"""Tests for the profile system (sp-prof)."""

from __future__ import annotations

import argparse
import textwrap

import pytest

from synth_panel.profiles import (
    Profile,
    apply_profile_to_args,
    list_available_profiles,
    load_profile_by_name,
    load_profile_from_path,
)

# --- Profile dataclass ---


class TestProfile:
    def test_to_dict_minimal(self):
        p = Profile(name="test")
        assert p.to_dict() == {"name": "test"}

    def test_to_dict_full(self):
        p = Profile(
            name="full",
            model="haiku",
            temperature=0.7,
            top_p=0.9,
            synthesis_model="sonnet",
            synthesis_temperature=0.3,
            prompt_template="tpl.j2",
            models="haiku:0.5,gemini:0.5",
        )
        d = p.to_dict()
        assert d["name"] == "full"
        assert d["model"] == "haiku"
        assert d["temperature"] == 0.7
        assert d["top_p"] == 0.9
        assert d["synthesis_model"] == "sonnet"
        assert d["synthesis_temperature"] == 0.3
        assert d["prompt_template"] == "tpl.j2"
        assert d["models"] == "haiku:0.5,gemini:0.5"

    def test_config_hash_deterministic(self):
        p = Profile(name="test", model="haiku", temperature=0.7)
        h1 = p.config_hash()
        h2 = p.config_hash()
        assert h1 == h2
        assert len(h1) == 16  # truncated sha256

    def test_config_hash_differs_for_different_profiles(self):
        p1 = Profile(name="a", model="haiku")
        p2 = Profile(name="b", model="sonnet")
        assert p1.config_hash() != p2.config_hash()


# --- Loading from path ---


class TestLoadFromPath:
    def test_load_valid_profile(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: my-profile
            model: haiku
            temperature: 0.7
            top_p: 0.9
            synthesis_model: sonnet
            synthesis_temperature: 0.3
        """)
        f = tmp_path / "profile.yaml"
        f.write_text(yaml_content)

        p = load_profile_from_path(str(f))
        assert p.name == "my-profile"
        assert p.model == "haiku"
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.synthesis_model == "sonnet"
        assert p.synthesis_temperature == 0.3
        assert p.source_path == str(f)

    def test_load_minimal_profile(self, tmp_path):
        f = tmp_path / "bare.yaml"
        f.write_text("name: bare\n")
        p = load_profile_from_path(str(f))
        assert p.name == "bare"
        assert p.model is None
        assert p.temperature is None

    def test_load_empty_file(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        p = load_profile_from_path(str(f))
        assert p.name == "empty"  # derives name from stem

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile_from_path("/nonexistent/path.yaml")

    def test_load_non_mapping_raises(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_profile_from_path(str(f))

    def test_null_values_parse_as_none(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            name: nulls
            model: null
            temperature: null
            top_p: null
        """)
        f = tmp_path / "nulls.yaml"
        f.write_text(yaml_content)
        p = load_profile_from_path(str(f))
        assert p.model is None
        assert p.temperature is None
        assert p.top_p is None


# --- Loading by name ---


class TestLoadByName:
    def test_bundled_default_exists(self):
        p = load_profile_by_name("default")
        assert p.name == "default"

    def test_bundled_consumer_exists(self):
        p = load_profile_by_name("consumer")
        assert p.name == "consumer"
        assert p.model == "haiku"
        assert p.temperature == 0.7

    def test_bundled_research_exists(self):
        p = load_profile_by_name("research")
        assert p.name == "research"
        assert p.temperature == 1.0
        assert p.models == "haiku:0.5,gemini:0.5"

    def test_nonexistent_profile_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile_by_name("nonexistent-profile-xyz")

    def test_strip_yaml_extension(self):
        p = load_profile_by_name("default.yaml")
        assert p.name == "default"

    def test_local_profiles_dir(self, tmp_path, monkeypatch):
        """Profiles in ./profiles/ are found."""
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        (profiles_dir / "local-test.yaml").write_text("name: local-test\nmodel: opus\n")
        monkeypatch.chdir(tmp_path)
        p = load_profile_by_name("local-test")
        assert p.name == "local-test"
        assert p.model == "opus"


# --- apply_profile_to_args ---


class TestApplyProfile:
    def _make_args(self, **kwargs):
        ns = argparse.Namespace()
        ns.model = None
        ns.models = None
        ns.temperature = None
        ns.top_p = None
        ns.synthesis_model = None
        ns.synthesis_temperature = None
        ns.prompt_template = None
        for k, v in kwargs.items():
            setattr(ns, k, v)
        return ns

    def test_applies_all_defaults(self):
        profile = Profile(
            name="test",
            model="haiku",
            temperature=0.7,
            synthesis_model="sonnet",
        )
        args = self._make_args()
        applied = apply_profile_to_args(profile, args)

        assert args.model == "haiku"
        assert args.temperature == 0.7
        assert args.synthesis_model == "sonnet"
        assert applied == {"model": "haiku", "temperature": 0.7, "synthesis_model": "sonnet"}

    def test_cli_flags_override_profile(self):
        profile = Profile(name="test", model="haiku", temperature=0.7)
        args = self._make_args(model="opus", temperature=1.5)
        applied = apply_profile_to_args(profile, args)

        assert args.model == "opus"  # CLI wins
        assert args.temperature == 1.5  # CLI wins
        assert applied == {}  # nothing was applied from profile

    def test_partial_override(self):
        profile = Profile(name="test", model="haiku", temperature=0.7, top_p=0.9)
        args = self._make_args(temperature=1.0)
        applied = apply_profile_to_args(profile, args)

        assert args.model == "haiku"  # from profile
        assert args.temperature == 1.0  # CLI override
        assert args.top_p == 0.9  # from profile
        assert "temperature" not in applied

    def test_models_spec_applied(self):
        profile = Profile(name="test", models="haiku:0.5,gemini:0.5")
        args = self._make_args()
        applied = apply_profile_to_args(profile, args)
        assert args.models == "haiku:0.5,gemini:0.5"
        assert applied["models"] == "haiku:0.5,gemini:0.5"

    def test_models_not_applied_when_model_set(self):
        profile = Profile(name="test", models="haiku:0.5,gemini:0.5")
        args = self._make_args(model="opus")
        applied = apply_profile_to_args(profile, args)
        assert args.models is None  # not applied
        assert "models" not in applied


# --- list_available_profiles ---


class TestListProfiles:
    def test_includes_bundled(self):
        profiles = list_available_profiles()
        names = [p["name"] for p in profiles]
        assert "default" in names
        assert "consumer" in names
        assert "research" in names

    def test_bundled_source(self):
        profiles = list_available_profiles()
        for p in profiles:
            if p["name"] == "default":
                assert p["source"] == "bundled"
                break


# --- CLI parser integration ---


class TestParserProfile:
    def test_profile_flag_parsed(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["--profile", "consumer", "prompt", "hello"])
        assert args.profile == "consumer"

    def test_profile_default_none(self):
        from synth_panel.cli.parser import build_parser

        parser = build_parser()
        args = parser.parse_args(["prompt", "hello"])
        assert args.profile is None
