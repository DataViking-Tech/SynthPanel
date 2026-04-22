"""Render site/index.html.j2 -> site/index.html with the canonical version.

Single source of truth is ``src/synth_panel/__version__.py``. The template
uses ``{{ version }}`` and ``{{ release_date }}`` placeholders; release date
is parsed from CHANGELOG.md (line matching ``## [X.Y.Z] - YYYY-MM-DD``) and
falls back to an empty string when the version has no dated CHANGELOG entry
yet (e.g. unreleased development builds).

Invoke directly (``python scripts/render_site.py``) or import ``render()``
from tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "site" / "index.html.j2"
OUTPUT_PATH = REPO_ROOT / "site" / "index.html"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"


def _read_version() -> str:
    # Read from source without importing the package, so this script works
    # before an editable install and in clean build environments.
    src = (REPO_ROOT / "src" / "synth_panel" / "__version__.py").read_text()
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', src, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse __version__ from {REPO_ROOT}/src/synth_panel/__version__.py")
    return match.group(1)


def _read_release_date(version: str) -> str:
    if not CHANGELOG_PATH.exists():
        return ""
    for line in CHANGELOG_PATH.read_text().splitlines():
        match = re.match(rf"##\s*\[{re.escape(version)}\]\s*-\s*(\d{{4}}-\d{{2}}-\d{{2}})", line)
        if match:
            return match.group(1)
    return ""


def _substitute(template: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in context:
            raise KeyError(f"Template references undefined variable {{{{ {key} }}}}")
        return context[key]

    return re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", replace, template)


def render(*, write: bool = False) -> str:
    version = _read_version()
    release_date = _read_release_date(version)
    template = TEMPLATE_PATH.read_text()
    rendered = _substitute(template, {"version": version, "release_date": release_date})
    if write:
        OUTPUT_PATH.write_text(rendered)
    return rendered


def main() -> int:
    rendered = render(write=True)
    version = _read_version()
    print(f"Rendered {OUTPUT_PATH.relative_to(REPO_ROOT)} with version {version} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
