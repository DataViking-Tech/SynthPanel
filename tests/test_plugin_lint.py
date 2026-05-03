"""Tests for `synthpanel plugin lint` (sy-0rr)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from synth_panel.cli.commands import handle_plugin_lint
from synth_panel.cli.output import OutputFormat
from synth_panel.cli.parser import build_parser
from synth_panel.plugins.lint import lint_plugin

FIXTURES = Path(__file__).parent / "fixtures" / "plugins"


# ---------------------------------------------------------------------------
# Core linter
# ---------------------------------------------------------------------------


class TestLintPlugin:
    def test_good_fixture_passes(self) -> None:
        report = lint_plugin(FIXTURES / "good")
        assert report.errors == []
        assert report.warnings == []
        assert report.passed()
        assert report.passed(strict=True)
        assert report.plugin_name == "good-plugin"

    def test_bad_manifest_missing_version(self) -> None:
        report = lint_plugin(FIXTURES / "bad-manifest")
        assert not report.passed()
        codes = [i.code for i in report.errors]
        assert "missing-field" in codes
        assert any("version" in i.message for i in report.errors)

    def test_bad_hooks_unknown_name_and_wrong_type(self) -> None:
        report = lint_plugin(FIXTURES / "bad-hooks")
        assert not report.passed()
        codes = [i.code for i in report.errors]
        assert "unknown-hook" in codes
        assert "field-type" in codes
        assert any("on_persona_load" in i.message for i in report.errors)

    def test_bad_entrypoint_non_mapping_manifest(self) -> None:
        report = lint_plugin(FIXTURES / "bad-entrypoint")
        assert not report.passed()
        assert any(i.code == "manifest-shape" for i in report.errors)

    def test_missing_path(self, tmp_path: Path) -> None:
        report = lint_plugin(tmp_path / "does-not-exist")
        assert not report.passed()
        assert any(i.code == "path-missing" for i in report.errors)

    def test_directory_without_manifest(self, tmp_path: Path) -> None:
        report = lint_plugin(tmp_path)
        assert not report.passed()
        assert any(i.code == "manifest-missing" for i in report.errors)

    def test_yaml_syntax_error(self, tmp_path: Path) -> None:
        (tmp_path / "plugin.yaml").write_text("not: [valid", encoding="utf-8")
        report = lint_plugin(tmp_path)
        assert not report.passed()
        assert any(i.code == "manifest-yaml" for i in report.errors)

    def test_accepts_manifest_file_path_directly(self) -> None:
        report = lint_plugin(FIXTURES / "good" / "plugin.yaml")
        assert report.passed()
        assert report.plugin_name == "good-plugin"

    def test_empty_hook_list_is_warning_not_error(self, tmp_path: Path) -> None:
        manifest = {
            "name": "warn-plugin",
            "version": "1.0.0",
            "hooks": {"pre_tool_use": []},
        }
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        report = lint_plugin(tmp_path)
        assert report.errors == []
        assert any(i.code == "empty-hook" for i in report.warnings)
        assert report.passed()
        assert not report.passed(strict=True)

    def test_default_enabled_must_be_bool(self, tmp_path: Path) -> None:
        manifest = {
            "name": "p",
            "version": "1.0.0",
            "default_enabled": "yes",
        }
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        report = lint_plugin(tmp_path)
        assert any("default_enabled" in i.message and i.severity == "error" for i in report.issues)

    def test_unknown_lifecycle_stage(self, tmp_path: Path) -> None:
        manifest = {
            "name": "p",
            "version": "1.0.0",
            "lifecycle": {"on_load": ["echo nope"]},
        }
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        report = lint_plugin(tmp_path)
        assert any(i.code == "unknown-lifecycle" for i in report.errors)

    def test_empty_command_string_is_error(self, tmp_path: Path) -> None:
        manifest = {
            "name": "p",
            "version": "1.0.0",
            "hooks": {"pre_tool_use": ["   "]},
        }
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")
        report = lint_plugin(tmp_path)
        assert any(i.code == "empty-command" for i in report.errors)

    def test_to_dict_shape(self) -> None:
        report = lint_plugin(FIXTURES / "bad-hooks")
        d = report.to_dict()
        assert d["plugin_path"]
        assert d["plugin_name"] == "bad-hooks-plugin"
        assert isinstance(d["errors"], list)
        assert isinstance(d["warnings"], list)
        assert all("code" in e and "message" in e for e in d["errors"])


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------


class TestHandlePluginLint:
    def _parse(self, *argv: str):
        return build_parser().parse_args(["plugin", "lint", *argv])

    def test_good_fixture_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse(str(FIXTURES / "good"))
        rc = handle_plugin_lint(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "ok" in out
        assert "lint OK" in out

    def test_bad_fixture_returns_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse(str(FIXTURES / "bad-manifest"))
        rc = handle_plugin_lint(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 1
        assert "fail" in out
        assert "lint failed" in out

    def test_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse(str(FIXTURES / "bad-hooks"))
        rc = handle_plugin_lint(args, OutputFormat.JSON)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert rc == 1
        assert payload["failed"] == 1
        assert payload["passed"] == 0
        assert payload["reports"][0]["plugin_name"] == "bad-hooks-plugin"

    def test_strict_promotes_warnings(self, tmp_path: Path) -> None:
        manifest = {
            "name": "p",
            "version": "1.0.0",
            "hooks": {"pre_tool_use": []},
        }
        (tmp_path / "plugin.yaml").write_text(yaml.dump(manifest), encoding="utf-8")

        relaxed = handle_plugin_lint(self._parse(str(tmp_path)), OutputFormat.JSON)
        assert relaxed == 0

        strict = handle_plugin_lint(self._parse(str(tmp_path), "--strict"), OutputFormat.JSON)
        assert strict == 1

    def test_no_args_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse()
        rc = handle_plugin_lint(args, OutputFormat.TEXT)
        err = capsys.readouterr().err
        assert rc == 1
        assert "plugin path" in err

    def test_path_and_all_conflict(self, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse(str(FIXTURES / "good"), "--all")
        rc = handle_plugin_lint(args, OutputFormat.TEXT)
        err = capsys.readouterr().err
        assert rc == 1
        assert "either" in err

    def test_all_with_empty_config_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        args = self._parse("--all", "--config-dir", str(tmp_path))
        rc = handle_plugin_lint(args, OutputFormat.TEXT)
        out = capsys.readouterr().out
        assert rc == 0
        assert "No plugins installed" in out

    def test_all_walks_installed_plugins(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plugins_root = tmp_path / "plugins"
        plugins_root.mkdir()

        good = plugins_root / "good"
        good.mkdir()
        (good / "plugin.yaml").write_text(yaml.dump({"name": "good", "version": "1.0.0"}), encoding="utf-8")

        bad = plugins_root / "bad"
        bad.mkdir()
        (bad / "plugin.yaml").write_text(yaml.dump({"name": "bad"}), encoding="utf-8")

        args = self._parse("--all", "--config-dir", str(tmp_path))
        rc = handle_plugin_lint(args, OutputFormat.JSON)
        out = capsys.readouterr().out
        payload = json.loads(out)

        assert rc == 1
        assert payload["passed"] == 1
        assert payload["failed"] == 1


# ---------------------------------------------------------------------------
# End-to-end CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_smoke_good_fixture() -> None:
    """Invoke the real CLI module and check the exit code is zero."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "synth_panel",
            "plugin",
            "lint",
            str(FIXTURES / "good"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_cli_smoke_bad_fixture_nonzero() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "synth_panel",
            "plugin",
            "lint",
            str(FIXTURES / "bad-hooks"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "fail" in proc.stdout
