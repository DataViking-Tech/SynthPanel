"""Verify the /.well-known/mcp/server-card.json discovery doc.

Implements AR-7 (sy-02p) — the MCP Server Card per SEP-2127.
The card is served as a static asset by Cloudflare Pages from
``site/.well-known/mcp/server-card.json``. These tests guard:

  * file exists and is valid JSON
  * required fields are present (serverInfo, capabilities, transport)
  * version everywhere matches ``synth_panel.__version__`` so a release
    bump cannot leave the card pointing at a stale version
  * the Cloudflare ``site/_headers`` block grants the cross-origin
    access agents need to fetch the card from a different host

The skill spec (https://isitagentready.com/.well-known/agent-skills/mcp-server-card/SKILL.md)
validates discovery via POST /api/scan; these checks mirror what that
scanner needs to see.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD_PATH = REPO_ROOT / "site" / ".well-known" / "mcp" / "server-card.json"
HEADERS_PATH = REPO_ROOT / "site" / "_headers"


def _load_card() -> dict:
    assert CARD_PATH.is_file(), f"missing {CARD_PATH.relative_to(REPO_ROOT)}"
    return json.loads(CARD_PATH.read_text(encoding="utf-8"))


def test_card_is_valid_json_with_schema_pin() -> None:
    card = _load_card()
    schema = card.get("$schema", "")
    assert schema.startswith("https://static.modelcontextprotocol.io/schemas/"), (
        f"$schema must pin a dated MCP server schema, got: {schema!r}"
    )
    assert schema.endswith("/server.schema.json"), (
        "$schema must reference server.schema.json (server-card is a strict subset)"
    )


def test_card_has_required_top_level_fields() -> None:
    card = _load_card()
    # SEP-2127: name + version are mandatory; title/description/serverInfo
    # are required by the SKILL contract for AR-7.
    for field in ("name", "title", "description", "version", "serverInfo", "capabilities"):
        assert field in card, f"server card missing required field: {field}"


def test_server_info_shape() -> None:
    card = _load_card()
    info = card["serverInfo"]
    assert isinstance(info, dict), "serverInfo must be an object"
    assert isinstance(info.get("name"), str) and info["name"], "serverInfo.name required"
    assert isinstance(info.get("version"), str) and info["version"], "serverInfo.version required"


def test_capabilities_declare_mcp_surface() -> None:
    card = _load_card()
    caps = card["capabilities"]
    # synthpanel's MCP server exposes tools, resources, and prompts —
    # advertise all three so discovery scanners know the surface.
    for cap in ("tools", "resources", "prompts"):
        assert cap in caps, f"capabilities.{cap} not declared"
        assert isinstance(caps[cap], dict), f"capabilities.{cap} must be an object"


def test_transport_endpoint_documented() -> None:
    """SEP-2127 requires the card to describe how to reach the server.

    synthpanel ships a stdio MCP server, so the transport is documented
    via ``packages[].transport`` (mirroring server.json) rather than via
    ``remotes`` (HTTP/SSE). Either path satisfies the contract — what
    must NOT happen is publishing a card with no transport at all.
    """
    card = _load_card()
    has_remotes = bool(card.get("remotes"))
    packages = card.get("packages") or []
    has_pkg_transport = any(
        isinstance(p, dict) and isinstance(p.get("transport"), dict) and p["transport"].get("type") for p in packages
    )
    assert has_remotes or has_pkg_transport, (
        "server card must document a transport endpoint via remotes[] or packages[].transport"
    )


def test_versions_match_package_version() -> None:
    """Drift guard: bumping ``__version__`` without updating the card is a bug.

    Mirrors the contract enforced by ``test_site_version.py`` for the
    landing page. Update both atomically at release time (or wire into
    a render script later).
    """
    from synth_panel import __version__

    card = _load_card()
    assert card["version"] == __version__, (
        f"server-card.json version {card['version']!r} != package __version__ {__version__!r}"
    )
    assert card["serverInfo"]["version"] == __version__, (
        f"serverInfo.version {card['serverInfo']['version']!r} != package __version__ {__version__!r}"
    )
    for pkg in card.get("packages", []):
        if pkg.get("identifier") == "synthpanel" and "version" in pkg:
            assert pkg["version"] == __version__, (
                f"packages[].version {pkg['version']!r} != package __version__ {__version__!r}"
            )


def test_headers_serve_card_with_json_and_cors() -> None:
    """The card is useless if browsers/agents can't fetch it cross-origin.

    Cloudflare Pages picks up ``site/_headers`` automatically; this guards
    against a future edit accidentally dropping the JSON content-type or
    the wildcard CORS that public discovery scanners depend on.
    """
    text = HEADERS_PATH.read_text(encoding="utf-8")
    assert "/.well-known/mcp/server-card.json" in text, (
        "site/_headers must declare a header block for /.well-known/mcp/server-card.json"
    )
    # Locate the block and assert its directives.
    lines = text.splitlines()
    try:
        idx = lines.index("/.well-known/mcp/server-card.json")
    except ValueError as exc:
        raise AssertionError("header block path not found at start of a line") from exc
    block: list[str] = []
    for line in lines[idx + 1 :]:
        if line and not line.startswith((" ", "\t")):
            break
        block.append(line.strip())
    block_text = "\n".join(block).lower()
    assert "content-type: application/json" in block_text, "card must be served as application/json"
    assert "access-control-allow-origin: *" in block_text, (
        "card must allow cross-origin GET so public agent scanners can fetch it"
    )
