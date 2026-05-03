"""Verify site/_headers ships RFC 8288 Link headers for agent discovery.

The committed `_headers` file is consumed by Cloudflare Pages and applied to
the live response on `https://synthpanel.dev/`. This test guards against
silent removal of the `Link` headers (sy-7r1, AR-1) so agents can keep
discovering the api-catalog and service-doc endpoints declared by the SKILL
spec at <https://isitagentready.com/.well-known/agent-skills/link-headers/SKILL.md>.

This test only validates the file's contents — actual header propagation is
verified manually via `curl -sI https://synthpanel.dev/` after deploy.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADERS_PATH = REPO_ROOT / "site" / "_headers"


def _parse_headers_file(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a Cloudflare Pages / Netlify _headers file into {path: [(name, value), ...]}.

    Comments (lines starting with '#') and blank lines separate blocks but are
    otherwise ignored. Path lines start at column 0; header lines are indented.
    """
    blocks: dict[str, list[tuple[str, str]]] = {}
    current_path: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not raw.startswith((" ", "\t")):
            current_path = line.strip()
            blocks.setdefault(current_path, [])
            continue
        if current_path is None:
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        name, _, value = stripped.partition(":")
        blocks[current_path].append((name.strip(), value.strip()))
    return blocks


def test_homepage_has_link_headers() -> None:
    blocks = _parse_headers_file(HEADERS_PATH.read_text())
    assert "/" in blocks, "_headers is missing a homepage (`/`) block for Link headers"

    link_values = [v for n, v in blocks["/"] if n.lower() == "link"]
    assert link_values, "homepage block has no Link headers"

    rels: dict[str, str] = {}
    for value in link_values:
        match = re.match(r"<([^>]+)>\s*;\s*rel=\"([^\"]+)\"", value)
        assert match, f"Link header is not RFC 8288 compliant: {value!r}"
        rels[match.group(2)] = match.group(1)

    assert rels.get("api-catalog") == "/.well-known/api-catalog", (
        "api-catalog Link header missing or pointing at the wrong target — "
        "must be /.well-known/api-catalog per RFC 9727"
    )
    assert "service-doc" in rels, "service-doc Link header missing"
    assert rels["service-doc"].startswith("/"), (
        f"service-doc target should be a site-relative path, got {rels['service-doc']!r}"
    )


def test_security_headers_still_apply_globally() -> None:
    """Adding the homepage Link block must not displace the global /* security headers."""
    blocks = _parse_headers_file(HEADERS_PATH.read_text())
    assert "/*" in blocks, "_headers lost its global /* block"
    names = {n.lower() for n, _ in blocks["/*"]}
    for required in (
        "x-frame-options",
        "x-content-type-options",
        "content-security-policy",
        "strict-transport-security",
    ):
        assert required in names, f"global /* block lost {required!r}"
