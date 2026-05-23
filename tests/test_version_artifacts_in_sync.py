"""sy-hs4: unified drift guard for every committed file that embeds the version.

The v1.5.0 release shipped via three follow-up PRs (#488 / #489 / #490)
because :file:`.github/workflows/auto-tag.yml`'s render step touched
``site/index.html`` but not ``site/index.md`` or
``site/.well-known/mcp/server-card.json``. Individual drift guards exist
for each artifact (``test_site_version.py``, ``test_site_markdown.py``,
``test_well_known_server_card.py``), but the *aggregate* contract was
implicit — adding a new artifact meant remembering to file a new test
*and* a new render-script call in auto-tag.

This module is the single explicit aggregate. It enumerates every
artifact the auto-tag workflow MUST keep in lockstep with
``src/synth_panel/__version__.py`` and asserts each one literally
matches. Adding a future artifact = adding one entry to ``_ARTIFACTS``
below and one matching render call in ``auto-tag.yml``.

The tests deliberately read the committed files rather than re-render —
the rendered output is exactly what ships to PyPI / Cloudflare Pages,
so any drift here is a real release defect, not a tooling quirk.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_PATH = _REPO_ROOT / "src" / "synth_panel" / "__version__.py"
_SITE_HTML = _REPO_ROOT / "site" / "index.html"
_SITE_MD = _REPO_ROOT / "site" / "index.md"
_SERVER_CARD = _REPO_ROOT / "site" / ".well-known" / "mcp" / "server-card.json"


def _canonical_version() -> str:
    """Single source of truth — parse __version__.py without importing.

    Mirrors :func:`scripts.render_site._read_version` so the guard
    works pre-install and pre-pythonpath in any CI shape.
    """
    src = _VERSION_PATH.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', src, re.MULTILINE)
    assert match, f"Could not parse __version__ from {_VERSION_PATH}"
    return match.group(1)


@pytest.fixture(scope="module")
def canonical_version() -> str:
    return _canonical_version()


def test_site_index_html_carries_version(canonical_version: str) -> None:
    """site/index.html: the landing page hero badge + JSON-LD softwareVersion."""
    html = _SITE_HTML.read_text(encoding="utf-8")

    # The hero shows ``v{version} — public beta``; the JSON-LD block
    # exposes the same value via ``"softwareVersion": "..."``. Both
    # render from the same template, but assert both anyway so a
    # mistakenly-edited template fragment also fails.
    assert f"v{canonical_version} — public beta" in html, (
        "site/index.html hero badge missing current __version__. Run `python scripts/render_site.py` to refresh."
    )
    jsonld = re.search(r'"softwareVersion":\s*"([^"]+)"', html)
    assert jsonld, "site/index.html JSON-LD missing softwareVersion"
    assert jsonld.group(1) == canonical_version, (
        f"JSON-LD softwareVersion {jsonld.group(1)!r} != __version__ {canonical_version!r}"
    )


def test_site_index_md_carries_version(canonical_version: str) -> None:
    """site/index.md: the markdown rendition served by the content-negotiation worker."""
    md = _SITE_MD.read_text(encoding="utf-8")
    assert f"v{canonical_version}" in md, (
        f"site/index.md missing v{canonical_version}. Run `python scripts/render_site_markdown.py` to refresh."
    )


def test_server_card_carries_version_in_all_three_slots(canonical_version: str) -> None:
    """server-card.json: top, serverInfo, packages[synthpanel].

    These are the three slots the MCP server-card schema exposes. The
    discovery doc gets fetched by every MCP-capable host; if any slot
    lags the package version, agents see a stale capability surface.
    """
    card = json.loads(_SERVER_CARD.read_text(encoding="utf-8"))
    assert card["version"] == canonical_version, (
        f"server-card.json top version {card['version']!r} != "
        f"__version__ {canonical_version!r}. "
        "Run `python scripts/render_server_card.py` to refresh."
    )
    assert card["serverInfo"]["version"] == canonical_version, (
        f"server-card.json serverInfo.version {card['serverInfo']['version']!r} != __version__ {canonical_version!r}"
    )
    for pkg in card.get("packages") or []:
        if pkg.get("identifier") == "synthpanel" and "version" in pkg:
            assert pkg["version"] == canonical_version, (
                f"server-card.json packages[synthpanel].version {pkg['version']!r} != __version__ {canonical_version!r}"
            )


def test_auto_tag_workflow_calls_every_renderer() -> None:
    """auto-tag.yml must invoke every renderer this module guards.

    Reading the workflow YAML as text (rather than parsing it) is
    intentional — we only care that the *string* ``python scripts/X.py``
    appears, not where in the job graph it sits. The render step is
    the contract; later steps can be reorganised freely as long as the
    renderers stay called.

    sy-hs4 origin: the v1.5.0 release proved that "we added a new
    render script" without "we added the call to auto-tag" is a real
    failure mode. This test makes the omission impossible to merge.
    """
    workflow = (_REPO_ROOT / ".github" / "workflows" / "auto-tag.yml").read_text(encoding="utf-8")
    required_renderers = (
        "scripts/render_site.py",
        "scripts/render_site_markdown.py",
        "scripts/render_server_card.py",
    )
    missing = [r for r in required_renderers if r not in workflow]
    assert not missing, (
        "auto-tag.yml is missing renderer invocations for: "
        f"{missing}. Add `python {missing[0]}` to the "
        "'Bump __version__.py and re-render every version artifact' step."
    )


def test_render_server_card_is_idempotent() -> None:
    """A no-op render must produce a no-op diff.

    Pulls in :mod:`scripts.render_server_card` and re-renders. The
    output bytes must equal the committed file's bytes — if they
    don't, either the script's serialization is drifting (indent /
    trailing newline) or someone edited the card by hand without
    re-running the script.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_server_card",
        _REPO_ROOT / "scripts" / "render_server_card.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rendered, _ = mod.render(write=False)
    on_disk = _SERVER_CARD.read_text(encoding="utf-8")
    assert rendered == on_disk, (
        "scripts/render_server_card.py produces output that differs "
        "from the committed file. Either run the script to refresh, "
        "or fix the script's serialization (indent, trailing newline) "
        "so the no-op render matches what's committed."
    )
