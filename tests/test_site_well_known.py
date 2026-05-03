"""Verify the site ships RFC 9728 OAuth Protected Resource Metadata.

Guards the static `/.well-known/oauth-protected-resource` document and the
matching `_headers` Content-Type rule that Cloudflare Pages uses to serve it
as `application/json` (sy-4nf, AR-6).

SKILL spec:
  https://isitagentready.com/.well-known/agent-skills/oauth-protected-resource/SKILL.md
RFC 9728:
  https://www.rfc-editor.org/rfc/rfc9728

This test only validates the committed files. Live HTTP behaviour is verified
manually via `curl -sI https://synthpanel.dev/.well-known/oauth-protected-resource`
after deploy.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = REPO_ROOT / "site" / ".well-known" / "oauth-protected-resource"
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


def test_metadata_file_exists_and_is_valid_json() -> None:
    assert METADATA_PATH.is_file(), f"missing {METADATA_PATH.relative_to(REPO_ROOT)} — RFC 9728 metadata document"
    payload = json.loads(METADATA_PATH.read_text())
    assert isinstance(payload, dict), "metadata document root must be a JSON object"


def test_metadata_has_required_fields() -> None:
    """Per the agent-ready SKILL spec, `resource` and `authorization_servers` are required."""
    payload = json.loads(METADATA_PATH.read_text())

    resource = payload.get("resource")
    assert isinstance(resource, str) and resource, "`resource` must be a non-empty string"
    assert resource.startswith("https://"), f"`resource` must be an https URL per RFC 9728 §2, got {resource!r}"
    assert resource == "https://synthpanel.dev", f"`resource` must identify the canonical site, got {resource!r}"

    authorization_servers = payload.get("authorization_servers")
    assert isinstance(authorization_servers, list), "`authorization_servers` must be an array"
    for entry in authorization_servers:
        assert isinstance(entry, str) and entry.startswith("https://"), (
            f"each authorization_servers entry must be an https URL, got {entry!r}"
        )


def test_metadata_optional_fields_are_well_typed_when_present() -> None:
    payload = json.loads(METADATA_PATH.read_text())
    if "scopes_supported" in payload:
        scopes = payload["scopes_supported"]
        assert isinstance(scopes, list), "`scopes_supported` must be an array when present"
        for scope in scopes:
            assert isinstance(scope, str), f"each scope must be a string, got {scope!r}"


def test_headers_serves_metadata_as_json() -> None:
    """Cloudflare Pages must serve the well-known file with Content-Type: application/json.

    The file has no extension, so without an explicit _headers rule the default
    Content-Type would be application/octet-stream and clients (including the
    isitagentready scanner) would reject the response. The global /* block sends
    `X-Content-Type-Options: nosniff`, so the Content-Type override is mandatory.
    """
    blocks = _parse_headers_file(HEADERS_PATH.read_text())
    path = "/.well-known/oauth-protected-resource"
    assert path in blocks, f"_headers is missing the `{path}` block (sy-4nf, AR-6)"

    content_types = [v for n, v in blocks[path] if n.lower() == "content-type"]
    assert content_types == ["application/json"], (
        f"`{path}` must declare Content-Type: application/json, got {content_types!r}"
    )


def test_global_security_headers_still_apply() -> None:
    """Adding the well-known block must not displace the global /* security headers."""
    blocks = _parse_headers_file(HEADERS_PATH.read_text())
    assert "/*" in blocks, "_headers lost its global /* block"
    names = {n.lower() for n, _ in blocks["/*"]}
    for required in (
        "x-frame-options",
        "x-content-type-options",
        "content-security-policy",
        "strict-transport-security",
    ):
        assert required in names, f"global /* block lost required `{required}` header"
