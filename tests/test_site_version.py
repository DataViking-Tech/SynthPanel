"""Guard against site/index.html version drift vs. pyproject.toml.

sp-lwy: the landing page hero badge, footer, and JSON-LD softwareVersion
have silently drifted behind pyproject.toml across three separate releases.
Fail CI whenever they disagree so the next bump can't ship stale.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    return data["project"]["version"]


def test_site_index_version_matches_pyproject() -> None:
    version = _pyproject_version()
    html = (REPO_ROOT / "site" / "index.html").read_text()

    hero = re.search(r"v(\d+\.\d+\.\d+)\s+—\s+public beta", html)
    assert hero, "hero version badge not found in site/index.html"
    assert hero.group(1) == version, f"hero badge shows v{hero.group(1)} but pyproject.toml is {version}"

    footer = re.search(r"MIT-licensed · v(\d+\.\d+\.\d+) ·", html)
    assert footer, "footer version not found in site/index.html"
    assert footer.group(1) == version, f"footer shows v{footer.group(1)} but pyproject.toml is {version}"

    jsonld = re.search(r'"softwareVersion":\s*"(\d+\.\d+\.\d+)"', html)
    assert jsonld, "JSON-LD softwareVersion not found in site/index.html"
    assert jsonld.group(1) == version, f"JSON-LD softwareVersion {jsonld.group(1)} but pyproject.toml is {version}"
