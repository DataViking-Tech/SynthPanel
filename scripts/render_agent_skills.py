"""Render the Agent Skills Discovery index for synthpanel.dev.

Publishes ``/.well-known/agent-skills/index.json`` per the Agent Skills
Discovery RFC v0.2.0 (https://agentskills.io/), plus a mirror of each
skill's ``SKILL.md`` under ``/.well-known/agent-skills/<name>/SKILL.md``
so the digests in the index match what the site actually serves.

Single source of truth is the top-level ``skills/`` directory. Run
``python scripts/render_agent_skills.py`` to regenerate the published
files; ``tests/test_agent_skills_index.py`` enforces no drift.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
PUBLISH_DIR = REPO_ROOT / "site" / ".well-known" / "agent-skills"
INDEX_PATH = PUBLISH_DIR / "index.json"

SCHEMA_URL = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"

# Stable publication order — independent of filesystem traversal so the
# committed index.json is deterministic across platforms.
SKILL_ORDER = (
    "focus-group",
    "name-test",
    "concept-test",
    "survey-prescreen",
    "pricing-probe",
)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def _parse_description(skill_md: str, *, source: Path) -> str:
    match = _FRONTMATTER_RE.match(skill_md)
    if not match:
        raise RuntimeError(f"{source}: missing YAML frontmatter")
    for line in match.group(1).splitlines():
        if line.startswith("description:"):
            return line[len("description:") :].strip()
    raise RuntimeError(f"{source}: frontmatter has no 'description' field")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def render(*, write: bool = False) -> str:
    """Build the index. When ``write`` is true, also materialise the
    mirrored SKILL.md files and the index.json under ``site/``."""
    skills = []
    for name in SKILL_ORDER:
        source = SKILLS_DIR / name / "SKILL.md"
        body = source.read_bytes()
        description = _parse_description(body.decode("utf-8"), source=source)
        url = f"/.well-known/agent-skills/{name}/SKILL.md"

        if write:
            mirror = PUBLISH_DIR / name / "SKILL.md"
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_bytes(body)

        skills.append(
            {
                "name": name,
                "type": "skill-md",
                "description": description,
                "url": url,
                "digest": _sha256(body),
            }
        )

    index = {"$schema": SCHEMA_URL, "skills": skills}
    rendered = json.dumps(index, indent=2, ensure_ascii=False) + "\n"

    if write:
        PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_PATH.write_text(rendered)

    return rendered


def main() -> int:
    rendered = render(write=True)
    rel = INDEX_PATH.relative_to(REPO_ROOT)
    print(f"Rendered {rel} ({len(rendered)} bytes, {len(SKILL_ORDER)} skills)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
