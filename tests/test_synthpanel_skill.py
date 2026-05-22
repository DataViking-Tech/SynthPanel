"""Guard the Hermes-compatible director skill at skills/synthpanel/SKILL.md.

The top-level ``synthpanel`` skill exists so Hermes Agent (and other
skills-compatible hosts) can route SynthPanel triggers without ad-hoc
repo spelunking. That value depends on specific frontmatter fields
being present and shaped correctly — Hermes hides skills with malformed
or missing required metadata. These tests pin that contract.

Spec references:
- https://agentskills.io/ (open skills format; name + description required)
- https://hermes-agent.nousresearch.com/docs/developer-guide/creating-skills
  (Hermes-specific ``metadata.hermes.*`` and ``required_environment_variables``)
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / "skills" / "synthpanel" / "SKILL.md"

# Canonical trigger phrases from GH#474 — the issue specified these as the
# proposed triggers Hermes should match against, so the skill body must
# mention them or routing fails.
TRIGGER_PHRASES = (
    "synthetic panel",
    "focus group",
    "positioning",
    "compare product names",
    "pricing probe",
    "poll 30 personas",
    "prioritization poll",
)

# Anti-overclaiming and structural-quality patterns the GH issue called out.
GUIDANCE_PATTERNS = (
    "MCP",
    "CLI",
    "dry-run",
    "structured",
    "result ID",
    "synthetic preflight",
    "sycophancy",
)


def _read_frontmatter() -> dict:
    text = SKILL_PATH.read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{SKILL_PATH} missing YAML frontmatter"
    return yaml.safe_load(match.group(1))


def test_skill_file_exists() -> None:
    assert SKILL_PATH.is_file(), f"missing {SKILL_PATH.relative_to(REPO_ROOT)}"


def test_required_open_format_fields() -> None:
    """Agent Skills Discovery requires name + description at minimum."""
    fm = _read_frontmatter()
    assert fm["name"] == "synthpanel"
    assert isinstance(fm["description"], str) and len(fm["description"]) >= 40, (
        "description must be substantive enough for retrieval"
    )


def test_hermes_metadata_block_present() -> None:
    fm = _read_frontmatter()
    hermes = fm.get("metadata", {}).get("hermes")
    assert isinstance(hermes, dict), "metadata.hermes block missing"
    tags = hermes.get("tags")
    assert isinstance(tags, list) and tags, "metadata.hermes.tags must be a non-empty list"
    related = hermes.get("related_skills")
    assert isinstance(related, list) and set(related) >= {
        "focus-group",
        "name-test",
        "concept-test",
        "survey-prescreen",
        "pricing-probe",
    }, "related_skills must enumerate the five workflow skills"
    config = hermes.get("config")
    assert isinstance(config, list) and config, "metadata.hermes.config must declare at least one setting"
    for entry in config:
        assert {"key", "description", "default", "prompt"} <= set(entry), (
            f"Hermes config entry missing required keys: {entry}"
        )


def test_versioning_and_license_declared() -> None:
    """Hermes surfaces version and license in its skill catalog UI."""
    fm = _read_frontmatter()
    assert re.match(r"^\d+\.\d+\.\d+$", fm["version"]), f"version must be semver, got {fm['version']!r}"
    assert fm["license"], "license must be set"
    assert fm["author"], "author must be set"


def test_required_environment_variables_declared() -> None:
    fm = _read_frontmatter()
    env_vars = {entry["name"]: entry for entry in fm.get("required_environment_variables", [])}
    assert "ANTHROPIC_API_KEY" in env_vars, "default Claude provider key must be advertised"
    for name, entry in env_vars.items():
        assert "prompt" in entry, f"{name} missing 'prompt' for Hermes provisioning"
        assert "help" in entry, f"{name} missing 'help' text"
        assert "required_for" in entry, f"{name} missing 'required_for' scope hint"


def test_allowed_tools_covers_mcp_surface() -> None:
    """Claude Code reads allowed-tools to gate MCP calls — keep parity with the MCP server."""
    fm = _read_frontmatter()
    allowed = set(fm.get("allowed-tools", []))
    expected_core = {
        "mcp__synth_panel__run_prompt",
        "mcp__synth_panel__run_quick_poll",
        "mcp__synth_panel__run_panel",
    }
    assert expected_core <= allowed, f"core MCP tools missing from allowed-tools: {expected_core - allowed}"


def test_body_mentions_all_canonical_triggers() -> None:
    body = SKILL_PATH.read_text(encoding="utf-8").lower()
    missing = [phrase for phrase in TRIGGER_PHRASES if phrase.lower() not in body]
    assert not missing, f"GH#474 triggers missing from skill body: {missing}"


def test_body_includes_anti_overclaiming_guidance() -> None:
    body = SKILL_PATH.read_text(encoding="utf-8")
    missing = [pat for pat in GUIDANCE_PATTERNS if pat.lower() not in body.lower()]
    assert not missing, f"required guidance patterns missing: {missing}"


def test_body_uses_hermes_conventional_sections() -> None:
    body = SKILL_PATH.read_text(encoding="utf-8")
    for header in ("## When to use", "## Procedure", "## Pitfalls", "## Verification"):
        assert header in body, f"Hermes-conventional section missing: {header}"


def test_plugin_manifest_registers_director_skill() -> None:
    """The Claude Code plugin manifest must list the new skill so /plugin install works."""
    import json

    manifest = json.loads((REPO_ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert "skills/synthpanel/SKILL.md" in manifest["skills"], (
        "skills/synthpanel/SKILL.md missing from .claude-plugin/plugin.json"
    )
