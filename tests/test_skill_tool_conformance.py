"""Conformance guard: skills + docs reference only real MCP tools, in sync.

This is the durable guard behind the gap-analysis MCP findings (#4/#5):
a SKILL.md whose ``allowed-tools`` frontmatter or body names a tool that
does not exist in the MCP server leaves the agent unable to auto-approve
any real Althing tool, and the three shipped copies of each SKILL.md
silently drift apart. Both are exactly the kind of defect a human review
misses and a machine catches every time.

It asserts three things:

(a) every ``mcp__althing__<name>`` token referenced in any skill
    SKILL.md (all three copies), doc, or command resolves to a tool
    actually registered in ``src/althing/mcp/server.py``;
(b) each skill's ``allowed-tools`` frontmatter lists only real tools;
(c) the three SKILL.md copies (``skills/``, the packaged
    ``src/althing/agent_assets/skills/`` installer source, and the
    published ``site/.well-known/agent-skills/`` mirror) are byte-identical
    per skill, and the ``index.json`` digests match the served bodies.

The real tool list is **parsed from server.py's ``@mcp.tool()``
registrations via regex** — never hardcoded — so adding or renaming a
tool automatically keeps the conformance allowlist current.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

SERVER_PY = REPO_ROOT / "src" / "althing" / "mcp" / "server.py"
PLUGIN_MANIFEST = REPO_ROOT / ".claude-plugin" / "plugin.json"

# The three shipped copies of every SKILL.md that must stay in lockstep.
SKILLS_DIR = REPO_ROOT / "skills"
AGENT_ASSETS_SKILLS_DIR = REPO_ROOT / "src" / "althing" / "agent_assets" / "skills"
SITE_SKILLS_DIR = REPO_ROOT / "site" / ".well-known" / "agent-skills"
SITE_INDEX = SITE_SKILLS_DIR / "index.json"

# Files scanned for tool-name references beyond the skill copies.
_EXTRA_REFERENCE_GLOBS = (
    (REPO_ROOT / "docs", "*.md"),
    (REPO_ROOT / "commands", "*.md"),
    (REPO_ROOT / "src" / "althing" / "agent_assets" / "commands", "*.md"),
)
_EXTRA_REFERENCE_FILES = (REPO_ROOT / "README.md",)

# ``@mcp.tool()`` immediately above an (async) def is FastMCP's registration
# shape; the function name is the tool name. Tolerant of decorator args and
# surrounding whitespace, but does not need to be — the file uses bare
# ``@mcp.tool()``.
_TOOL_DECL_RE = re.compile(
    r"@mcp\.tool\([^)]*\)\s*\n\s*(?:async\s+)?def\s+(\w+)\s*\(",
    re.MULTILINE,
)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_TOOL_TOKEN_RE = re.compile(r"mcp__[a-z0-9_]+__[a-zA-Z_]\w*")


def _server_config_name() -> str:
    """The MCP server key agents address tools through (``althing``).

    Parsed from the plugin manifest's ``mcp_servers`` map rather than
    hardcoded, so a server rename can't leave this guard checking a stale
    prefix.
    """
    manifest = json.loads(PLUGIN_MANIFEST.read_text(encoding="utf-8"))
    servers = manifest.get("mcp_servers") or {}
    assert len(servers) == 1, f"expected exactly one mcp_servers entry, got {sorted(servers)}"
    return next(iter(servers))


def _real_tool_names() -> set[str]:
    """Bare tool names registered via ``@mcp.tool()`` in server.py."""
    source = SERVER_PY.read_text(encoding="utf-8")
    names = set(_TOOL_DECL_RE.findall(source))
    # Sanity floor: a regex break that captured nothing would make every
    # other assertion fail with a confusing message. Fail here instead,
    # loudly, pointing at the parser.
    assert len(names) >= 10, (
        f"parsed only {len(names)} @mcp.tool() registrations from {SERVER_PY} "
        f"({sorted(names)}) — the _TOOL_DECL_RE regex is likely stale."
    )
    return names


SERVER_NAME = _server_config_name()
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
REAL_TOOLS = _real_tool_names()
REAL_FULL_TOOLS = {f"{TOOL_PREFIX}{name}" for name in REAL_TOOLS}

# Canonical skill set = the directories under skills/ that contain a SKILL.md.
SKILL_NAMES = sorted(p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md"))


def _all_skill_md_paths() -> list[Path]:
    paths: list[Path] = []
    for base in (SKILLS_DIR, AGENT_ASSETS_SKILLS_DIR, SITE_SKILLS_DIR):
        paths.extend(sorted(base.glob("*/SKILL.md")))
    return paths


def _reference_files() -> list[Path]:
    files = list(_all_skill_md_paths())
    for base, pattern in _EXTRA_REFERENCE_GLOBS:
        if base.is_dir():
            files.extend(sorted(base.glob(pattern)))
    files.extend(f for f in _EXTRA_REFERENCE_FILES if f.is_file())
    return files


def _parse_allowed_tools(skill_md: str) -> list[str]:
    match = _FRONTMATTER_RE.match(skill_md)
    assert match, "SKILL.md missing YAML frontmatter"
    fm = yaml.safe_load(match.group(1)) or {}
    return list(fm.get("allowed-tools") or [])


def test_regex_parses_the_expected_core_tools() -> None:
    """Self-check: the parsed tool set includes the documented core tools."""
    for core in ("run_prompt", "run_panel", "run_quick_poll"):
        assert core in REAL_TOOLS, f"{core} not parsed from server.py registrations"


@pytest.mark.parametrize("ref_file", _reference_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_referenced_tool_names_are_real(ref_file: Path) -> None:
    """(a) Every ``mcp__<server>__<tool>`` token names a registered tool."""
    text = ref_file.read_text(encoding="utf-8")
    referenced = {tok for tok in _TOOL_TOKEN_RE.findall(text) if tok.startswith(TOOL_PREFIX)}
    unknown = sorted(referenced - REAL_FULL_TOOLS)
    assert not unknown, (
        f"{ref_file.relative_to(REPO_ROOT)} references MCP tools that are not "
        f"registered in {SERVER_PY.relative_to(REPO_ROOT)}: {unknown}. "
        f"Real tools: {sorted(REAL_TOOLS)}"
    )


@pytest.mark.parametrize("skill_md", _all_skill_md_paths(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_allowed_tools_lists_only_real_tools(skill_md: Path) -> None:
    """(b) ``allowed-tools`` frontmatter is a permission allowlist — every
    entry must be a real tool, or the skill can't auto-approve any call."""
    allowed = _parse_allowed_tools(skill_md.read_text(encoding="utf-8"))
    assert allowed, f"{skill_md.relative_to(REPO_ROOT)} has no allowed-tools entries"
    bad = sorted(t for t in allowed if t not in REAL_FULL_TOOLS)
    assert not bad, (
        f"{skill_md.relative_to(REPO_ROOT)} allowed-tools lists non-existent tools: {bad}. "
        f"Real tools: {sorted(REAL_FULL_TOOLS)}"
    )


def test_three_skill_copies_are_in_sync() -> None:
    """(c) The three shipped copies of each SKILL.md are byte-identical.

    ``skills/`` is the canonical source; the packaged installer copy
    (``agent_assets``) and the published site mirror must match it exactly,
    or manual installers / discovery clients get a stale skill.
    """
    agent_names = sorted(p.parent.name for p in AGENT_ASSETS_SKILLS_DIR.glob("*/SKILL.md"))
    site_names = sorted(p.parent.name for p in SITE_SKILLS_DIR.glob("*/SKILL.md"))
    assert agent_names == SKILL_NAMES, (
        f"agent_assets skills {agent_names} != canonical skills {SKILL_NAMES}. "
        "Copy every skills/<name>/SKILL.md into src/althing/agent_assets/skills/<name>/."
    )
    assert site_names == SKILL_NAMES, (
        f"site mirror skills {site_names} != canonical skills {SKILL_NAMES}. Run: python scripts/render_agent_skills.py"
    )
    for name in SKILL_NAMES:
        canonical = (SKILLS_DIR / name / "SKILL.md").read_bytes()
        agent = (AGENT_ASSETS_SKILLS_DIR / name / "SKILL.md").read_bytes()
        site = (SITE_SKILLS_DIR / name / "SKILL.md").read_bytes()
        assert agent == canonical, (
            f"src/althing/agent_assets/skills/{name}/SKILL.md drifted from skills/{name}/SKILL.md — re-copy it."
        )
        assert site == canonical, (
            f"site/.well-known/agent-skills/{name}/SKILL.md drifted from "
            f"skills/{name}/SKILL.md — run: python scripts/render_agent_skills.py"
        )


def test_site_index_digests_match_served_bodies() -> None:
    """(c) Each index.json digest matches the sha256 of its served SKILL.md."""
    index = json.loads(SITE_INDEX.read_text(encoding="utf-8"))
    entries = {e["name"]: e for e in index.get("skills", [])}
    assert sorted(entries) == SKILL_NAMES, (
        f"index.json skills {sorted(entries)} != canonical skills {SKILL_NAMES}. "
        "Run: python scripts/render_agent_skills.py"
    )
    for name, entry in entries.items():
        served = (REPO_ROOT / "site" / entry["url"].lstrip("/")).read_bytes()
        actual = "sha256:" + hashlib.sha256(served).hexdigest()
        assert actual == entry["digest"], (
            f"index.json digest for {name} does not match the served body at "
            f"{entry['url']} — run: python scripts/render_agent_skills.py"
        )
