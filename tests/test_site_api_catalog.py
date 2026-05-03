"""Verify site/.well-known/api-catalog conforms to RFC 9727 + the agent-skills SKILL.

The committed file is served verbatim by Cloudflare Pages at
<https://synthpanel.dev/.well-known/api-catalog>. The agent-skills validator
(POST https://isitagentready.com/api/scan) requires:
  - JSON parses,
  - top-level `linkset` is a non-empty array,
  - each entry has an `anchor` URL,
  - each entry's `links` array includes both `service-desc` and `service-doc`,
  - the response is served as `application/linkset+json`.

This test only validates the file's contents and the matching `_headers`
override. End-to-end media-type propagation must be smoke-checked manually
against the deployed site after merge:

    curl -sI https://synthpanel.dev/.well-known/api-catalog | grep -i content-type
    # -> content-type: application/linkset+json
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "site" / ".well-known" / "api-catalog"
HEADERS_PATH = REPO_ROOT / "site" / "_headers"


def _parse_headers_file(text: str) -> dict[str, list[tuple[str, str]]]:
    """Parse a Cloudflare Pages / Netlify _headers file into {path: [(name, value), ...]}.

    Path lines start at column 0; header lines are indented; '#' comments and
    blank lines are ignored.
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


def _load_catalog() -> dict:
    raw = CATALOG_PATH.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"api-catalog is not valid JSON ({exc.msg} at line {exc.lineno})") from exc


def test_api_catalog_file_exists_at_well_known_path() -> None:
    assert CATALOG_PATH.is_file(), (
        f"missing api-catalog at {CATALOG_PATH.relative_to(REPO_ROOT)} — "
        "Cloudflare Pages will 404 on /.well-known/api-catalog"
    )


def test_api_catalog_has_linkset_array() -> None:
    doc = _load_catalog()
    assert isinstance(doc, dict), "api-catalog root must be a JSON object"
    linkset = doc.get("linkset")
    assert isinstance(linkset, list) and linkset, "api-catalog must have a non-empty `linkset` array per RFC 9727"


def test_each_linkset_entry_has_required_relations() -> None:
    """Every entry must declare an anchor + at least service-desc and service-doc."""
    doc = _load_catalog()
    for idx, entry in enumerate(doc["linkset"]):
        anchor = entry.get("anchor")
        assert isinstance(anchor, str) and anchor, f"linkset[{idx}] missing string `anchor`"
        parsed = urlparse(anchor)
        assert parsed.scheme in {"http", "https"}, (
            f"linkset[{idx}] anchor must be an absolute http(s) URL, got {anchor!r}"
        )

        links = entry.get("links")
        assert isinstance(links, list) and links, f"linkset[{idx}] must contain a non-empty `links` array"

        rels: set[str] = set()
        for link_idx, link in enumerate(links):
            assert isinstance(link, dict), f"linkset[{idx}].links[{link_idx}] must be an object"
            rel = link.get("rel")
            href = link.get("href")
            assert isinstance(rel, str) and rel, f"linkset[{idx}].links[{link_idx}] missing `rel`"
            assert isinstance(href, str) and href, f"linkset[{idx}].links[{link_idx}] ({rel!r}) missing `href`"
            href_parsed = urlparse(href)
            assert href_parsed.scheme in {"http", "https"}, (
                f"linkset[{idx}].links[{link_idx}] ({rel!r}) href must be absolute, got {href!r}"
            )
            rels.add(rel)

        for required in ("service-desc", "service-doc"):
            assert required in rels, (
                f"linkset[{idx}] missing required `{required}` link relation (see agent-skills SKILL: api-catalog)"
            )


def test_headers_file_serves_catalog_as_linkset_json() -> None:
    """_headers must override the default Content-Type for the extension-less file."""
    blocks = _parse_headers_file(HEADERS_PATH.read_text())
    catalog_block = blocks.get("/.well-known/api-catalog")
    assert catalog_block is not None, (
        "_headers is missing a `/.well-known/api-catalog` block — Cloudflare will fall back to application/octet-stream"
    )
    content_types = [v for n, v in catalog_block if n.lower() == "content-type"]
    assert content_types == ["application/linkset+json"], (
        f"/.well-known/api-catalog must declare Content-Type: application/linkset+json, got {content_types!r}"
    )


def test_security_headers_still_apply_globally() -> None:
    """Adding the api-catalog block must not displace the global /* security headers."""
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
