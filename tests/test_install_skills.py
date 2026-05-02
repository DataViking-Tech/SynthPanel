"""Tests for `synthpanel install-skills` (sy-65k74)."""

from __future__ import annotations

import json

from synth_panel.cli.commands import _INSTALL_SKILLS_COMMANDS, _INSTALL_SKILLS_SKILLS
from synth_panel.cli.parser import build_parser
from synth_panel.main import main


class TestParser:
    def test_install_skills_registered(self):
        parser = build_parser()
        args = parser.parse_args(["install-skills"])
        assert args.command == "install-skills"
        assert args.target is None

    def test_install_skills_accepts_target(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(["install-skills", "--target", str(tmp_path)])
        assert args.target == str(tmp_path)


class TestInstallSkillsFiles:
    def test_package_data_commands_exist(self):
        from importlib.resources import files as _resource_files

        pkg = _resource_files("synth_panel.agent_assets")
        for name in _INSTALL_SKILLS_COMMANDS:
            src = pkg / "commands" / name
            assert src.read_bytes(), f"commands/{name} is empty or missing in package data"

    def test_package_data_skills_exist(self):
        from importlib.resources import files as _resource_files

        pkg = _resource_files("synth_panel.agent_assets")
        for skill in _INSTALL_SKILLS_SKILLS:
            src = pkg / "skills" / skill / "SKILL.md"
            assert src.read_bytes(), f"skills/{skill}/SKILL.md is empty or missing in package data"


class TestInstallSkillsCommand:
    def test_installs_all_files_to_target(self, tmp_path, capsys):
        rc = main(["install-skills", "--target", str(tmp_path)])
        assert rc == 0

        for name in _INSTALL_SKILLS_COMMANDS:
            assert (tmp_path / "commands" / name).is_file(), f"commands/{name} not installed"

        for skill in _INSTALL_SKILLS_SKILLS:
            assert (tmp_path / "skills" / skill / "SKILL.md").is_file(), f"skills/{skill}/SKILL.md not installed"

        out = capsys.readouterr().out
        expected = 1 + len(_INSTALL_SKILLS_SKILLS)
        assert f"Installed {expected} file(s)" in out

    def test_idempotent(self, tmp_path):
        main(["install-skills", "--target", str(tmp_path)])
        rc = main(["install-skills", "--target", str(tmp_path)])
        assert rc == 0

    def test_json_output(self, tmp_path, capsys):
        rc = main(["--output-format", "json", "install-skills", "--target", str(tmp_path)])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["target"] == str(tmp_path)
        assert isinstance(payload["installed"], list)
        assert len(payload["installed"]) == 1 + len(_INSTALL_SKILLS_SKILLS)

    def test_installed_files_have_content(self, tmp_path):
        main(["install-skills", "--target", str(tmp_path)])
        poll_md = (tmp_path / "commands" / "synthpanel-poll.md").read_text()
        assert "run_quick_poll" in poll_md
        for skill in _INSTALL_SKILLS_SKILLS:
            skill_md = (tmp_path / "skills" / skill / "SKILL.md").read_text()
            assert "---" in skill_md, f"skills/{skill}/SKILL.md missing frontmatter"

    def test_creates_parent_directories(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        rc = main(["install-skills", "--target", str(deep)])
        assert rc == 0
        assert (deep / "commands").is_dir()
        assert (deep / "skills").is_dir()
