"""Verify the published Agent Skills Discovery index stays in sync with skills/.

The published artifacts at ``site/.well-known/agent-skills/`` are derived
from the canonical ``skills/`` directory by ``scripts/render_agent_skills.py``.
If a skill's SKILL.md changes without a re-render, the digest in
``index.json`` no longer matches the served body — agents that verify the
digest reject the skill. These tests catch that drift in CI.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_DIR = REPO_ROOT / "site" / ".well-known" / "agent-skills"
INDEX_PATH = PUBLISH_DIR / "index.json"


def _load_renderer():
    spec = importlib.util.spec_from_file_location(
        "render_agent_skills", REPO_ROOT / "scripts" / "render_agent_skills.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_agent_skills"] = module
    spec.loader.exec_module(module)
    return module


def test_committed_index_matches_renderer() -> None:
    rendered = _load_renderer().render()
    committed = INDEX_PATH.read_text()
    assert rendered == committed, (
        "site/.well-known/agent-skills/index.json is stale. Run: python scripts/render_agent_skills.py"
    )


def test_index_schema_shape() -> None:
    index = json.loads(INDEX_PATH.read_text())
    assert index["$schema"].startswith("https://schemas.agentskills.io/discovery/")
    assert isinstance(index["skills"], list) and index["skills"], "skills must be a non-empty list"
    digest_re = re.compile(r"^sha256:[0-9a-f]{64}$")
    for entry in index["skills"]:
        assert set(entry) >= {"name", "type", "description", "url", "digest"}
        assert entry["type"] == "skill-md"
        assert entry["url"].startswith("/.well-known/agent-skills/")
        assert entry["url"].endswith("/SKILL.md")
        assert digest_re.match(entry["digest"]), f"bad digest format: {entry['digest']}"


def test_each_digest_matches_served_body() -> None:
    index = json.loads(INDEX_PATH.read_text())
    for entry in index["skills"]:
        served = (REPO_ROOT / "site" / entry["url"].lstrip("/")).read_bytes()
        actual = "sha256:" + hashlib.sha256(served).hexdigest()
        assert actual == entry["digest"], f"digest for {entry['name']} does not match served body at {entry['url']}"


def test_mirrored_skill_md_matches_canonical_source() -> None:
    """The mirror under site/.well-known is byte-identical to skills/<name>/SKILL.md."""
    index = json.loads(INDEX_PATH.read_text())
    for entry in index["skills"]:
        canonical = (REPO_ROOT / "skills" / entry["name"] / "SKILL.md").read_bytes()
        mirrored = (REPO_ROOT / "site" / entry["url"].lstrip("/")).read_bytes()
        assert canonical == mirrored, (
            f"skills/{entry['name']}/SKILL.md drifted from its site/ mirror. Run: python scripts/render_agent_skills.py"
        )
