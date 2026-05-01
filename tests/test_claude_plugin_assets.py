"""Guard tests for the shipped Claude Code plugin assets.

These tests exist because sp-tcf was closed with the claim that
``/synthpanel-poll`` had shipped, but the file was never actually
committed (sp-ftr). A filesystem existence check is cheap insurance
against that regression for every command the plugin advertises.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_MANIFEST = REPO_ROOT / ".claude-plugin" / "plugin.json"
COMMANDS_DIR = REPO_ROOT / "commands"
SKILLS_DIR = REPO_ROOT / "skills"
AGENT_SKILLS_DOC = REPO_ROOT / "docs" / "agent-skills.md"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _read_frontmatter(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    assert match, f"{path} is missing YAML frontmatter"
    return match.group(1)


def test_plugin_manifest_exists_and_is_valid_json() -> None:
    assert PLUGIN_MANIFEST.is_file(), f"missing {PLUGIN_MANIFEST}"
    data = json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    assert data.get("name") == "synthpanel"
    assert "mcp_servers" in data


def test_synthpanel_poll_command_ships() -> None:
    """The `/synthpanel-poll` slash command must be present on disk.

    Claude Code auto-discovers ``commands/*.md`` at the plugin root, so
    the existence of this file is what determines whether installed
    users get the advertised slash command.
    """
    path = COMMANDS_DIR / "synthpanel-poll.md"
    assert path.is_file(), (
        "commands/synthpanel-poll.md is missing — the README advertises "
        "/synthpanel-poll as a plugin command, so the file must ship."
    )

    frontmatter = _read_frontmatter(path)
    assert "description:" in frontmatter
    assert "allowed-tools:" in frontmatter
    assert "mcp__synth_panel__run_quick_poll" in frontmatter, (
        "synthpanel-poll must list the run_quick_poll MCP tool in "
        "allowed-tools — otherwise the slash command can't call it."
    )


@pytest.mark.parametrize(
    "skill_rel_path",
    [
        "skills/focus-group/SKILL.md",
        "skills/name-test/SKILL.md",
        "skills/concept-test/SKILL.md",
        "skills/survey-prescreen/SKILL.md",
        "skills/pricing-probe/SKILL.md",
    ],
)
def test_plugin_manifest_skills_all_exist(skill_rel_path: str) -> None:
    """Every skill path in plugin.json must resolve to a real file."""
    manifest = json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    assert skill_rel_path in manifest["skills"], f"plugin.json must list {skill_rel_path} in its skills array"
    assert (REPO_ROOT / skill_rel_path).is_file(), f"{skill_rel_path} is listed in plugin.json but does not exist"


def test_agent_skills_doc_exists() -> None:
    """The install-path doc must ship — pip-installed users have no
    other discoverable way to learn how to copy ``commands/`` and
    ``skills/`` into ``~/.claude/`` (sp-700cf4).
    """
    assert AGENT_SKILLS_DOC.is_file(), f"missing {AGENT_SKILLS_DOC}"


@pytest.mark.parametrize(
    "artifact",
    [
        "synthpanel-poll",
        "focus-group",
        "name-test",
        "concept-test",
        "survey-prescreen",
        "pricing-probe",
    ],
)
def test_agent_skills_doc_references_all_artifacts(artifact: str) -> None:
    """Every shipped command/skill must be named in the install doc.

    The acceptance criterion on sp-700cf4 is that the doc covers all
    six artifacts; if a future change adds a skill without updating
    the doc, this test catches it.
    """
    text = AGENT_SKILLS_DOC.read_text(encoding="utf-8")
    assert artifact in text, f"docs/agent-skills.md must mention '{artifact}' (the file or skill named on disk)"
