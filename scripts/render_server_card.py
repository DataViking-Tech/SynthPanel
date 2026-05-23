"""Sync site/.well-known/mcp/server-card.json to the canonical __version__ (sy-hs4).

Single-source-of-truth model identical to :mod:`scripts.render_site`:
read ``src/synth_panel/__version__.py``, then rewrite the three
version fields the MCP server-card schema exposes:

1. Top-level ``version``
2. ``serverInfo.version`` (runtime metadata the host server prints)
3. Every ``packages[].version`` where ``identifier == "synthpanel"``

We do *targeted line-level substitution* rather than ``json.load`` +
``json.dump`` because the committed file uses a compact style
(inline single-line objects for ``capabilities``, ``runtimeArguments``,
etc.) that Python's ``json.dumps(indent=2)`` would explode across
multiple lines. A reformatting render would produce a constant
diff every release. Targeted regex substitution keeps the rendered
output byte-identical to the committed file when the version is
already current.

One regex covers all three slots — the indentation differs but the
shape ``  "version": "X.Y.Z"`` is uniform across top-level,
``serverInfo.version``, and ``packages[i].version``. A
``_EXPECTED_VERSION_LINE_COUNT`` constant guards against silent
regressions: if the schema gains a fourth version slot (or loses one)
the renderer refuses to write a half-synced card rather than ship
inconsistent metadata.

The auto-tag workflow calls this between the ``__version__.py`` bump
and ``git push``. v1.5.0 required three follow-up PRs (#488/#489/#490)
because the original auto-tag step only re-rendered ``site/index.html``;
this script + the markdown re-render closes the gap.

Run directly (``python scripts/render_server_card.py``) or import
``render()`` from tests.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = REPO_ROOT / "site" / ".well-known" / "mcp" / "server-card.json"
VERSION_PATH = REPO_ROOT / "src" / "synth_panel" / "__version__.py"


def _read_version() -> str:
    """Parse ``__version__`` without importing the package."""
    src = VERSION_PATH.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', src, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not parse __version__ from {VERSION_PATH}")
    return match.group(1)


# Single substitution pattern: any line where the only key is
# ``"version"`` and the value is a string. Anchoring to ``^\s+`` plus
# the literal key avoids over-matching things like
# ``"$schema": "...server-card-version-1.json"`` (where the URL might
# contain the word "version" as part of a path component — the regex
# requires ``"version"`` to be the JSON KEY, not a value).
#
# Why a single pattern: the three slots — top-level (2-space indent),
# serverInfo.version (4-space indent), packages[i].version (6-space
# indent) — all share the shape ``  ... "version": "X.Y.Z"`` and a
# single multi-line match handles them uniformly. Adding indent-specific
# patterns proved fragile when the v1.5.0 server-card landed an extra
# nesting level; this pattern survives the next schema bump too.
_VERSION_LINE = re.compile(
    r'^(?P<lead>\s+"version":\s*")[^"]+(?P<trail>",?)$',
    re.MULTILINE,
)

# Expected number of version lines in a well-formed card. v1.5.0 has
# exactly three: top, serverInfo, packages[0]. Asserting this guards
# against (a) silent regressions where a slot moves and we stop
# matching, (b) silent additions where someone adds a new ``version``
# field that we should be syncing too.
_EXPECTED_VERSION_LINE_COUNT = 3


def render(*, write: bool = False) -> tuple[str, str]:
    """Return ``(rendered_text, version)``; write to disk when requested.

    Returns the rendered text so the idempotency test can compare
    against the on-disk file without re-reading.
    """
    version = _read_version()
    text = CARD_PATH.read_text(encoding="utf-8")

    new_text, n = _VERSION_LINE.subn(
        rf"\g<lead>{version}\g<trail>",
        text,
    )
    if n != _EXPECTED_VERSION_LINE_COUNT:
        # Mismatch means the schema drifted (someone added or moved a
        # version slot). Refuse to write a half-synced card — the next
        # release would land with mismatched fields, exactly the v1.5.0
        # failure mode this whole script exists to prevent.
        raise RuntimeError(
            f"render_server_card: matched {n} version line(s) in {CARD_PATH}, "
            f"expected {_EXPECTED_VERSION_LINE_COUNT} "
            "(top-level, serverInfo.version, packages[0].version). "
            "If the schema legitimately changed, update "
            "_EXPECTED_VERSION_LINE_COUNT in scripts/render_server_card.py."
        )

    if write:
        CARD_PATH.write_text(new_text, encoding="utf-8")
    return new_text, version


def main() -> int:
    rendered, version = render(write=True)
    print(f"Rendered {CARD_PATH.relative_to(REPO_ROOT)} with version {version} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
