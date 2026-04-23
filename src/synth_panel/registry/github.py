"""GitHub URL resolver for remote pack sources.

Accepted source forms:

- ``gh:user/repo`` — defaults to ``ref=main``, ``path=synthpanel-pack.yaml``
- ``gh:user/repo@ref``
- ``gh:user/repo:path``
- ``gh:user/repo@ref:path``
- ``https://raw.githubusercontent.com/...`` — passthrough
- ``https://github.com/user/repo/blob/ref/path`` — rewritten to raw

When the default path is used (no ``:path`` given) and the default file
is not present (HTTP 404), the resolver falls back to listing the repo
root via the GitHub Contents API and selects the single ``*.yaml`` or
``*.yml`` at the root. Zero or multiple candidates raise ``ValueError``.

Per S-gate OQ1, the default publishable path is ``synthpanel-pack.yaml``
— namespaced to avoid collisions with unrelated ``pack.yaml`` files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_API_BASE = "https://api.github.com"
DEFAULT_REF = "main"
DEFAULT_PATH = "synthpanel-pack.yaml"

_HTTP_TIMEOUT = 10.0


@dataclass(frozen=True)
class GitHubSource:
    """Parsed components of a ``gh:user/repo[@ref][:path]`` source spec.

    ``path_is_explicit`` indicates whether the user supplied ``:path``
    explicitly; the resolver only attempts the single-yaml fallback when
    this is ``False``.
    """

    user: str
    repo: str
    ref: str
    path: str
    path_is_explicit: bool


def parse_gh_source(source: str) -> GitHubSource:
    """Parse a ``gh:`` source spec into a :class:`GitHubSource`.

    Raises ``ValueError`` with a precise message for malformed input
    (missing ``user/repo``, empty segments, etc.).
    """
    if not isinstance(source, str):
        raise ValueError(f"gh source must be a string, got {type(source).__name__}")
    if not source.startswith("gh:"):
        raise ValueError(f"not a gh: source: {source!r}")

    body = source[len("gh:") :]
    if not body:
        raise ValueError("empty gh: source (expected gh:user/repo[@ref][:path])")

    at_idx = body.find("@")
    colon_idx = body.find(":")

    if at_idx >= 0 and (colon_idx < 0 or at_idx < colon_idx):
        user_repo = body[:at_idx]
        rest = body[at_idx + 1 :]
        rest_colon = rest.find(":")
        if rest_colon >= 0:
            ref = rest[:rest_colon]
            path: str | None = rest[rest_colon + 1 :]
        else:
            ref = rest
            path = None
    else:
        if colon_idx >= 0:
            user_repo = body[:colon_idx]
            path = body[colon_idx + 1 :]
        else:
            user_repo = body
            path = None
        ref = DEFAULT_REF

    parts = user_repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"invalid user/repo in gh: source {source!r} "
            "(expected gh:user/repo[@ref][:path])"
        )
    user, repo = parts

    if not ref:
        raise ValueError(f"empty ref in gh: source {source!r}")
    if path is not None and not path:
        raise ValueError(f"empty path in gh: source {source!r}")

    path_is_explicit = path is not None
    return GitHubSource(
        user=user,
        repo=repo,
        ref=ref,
        path=path if path_is_explicit else DEFAULT_PATH,
        path_is_explicit=path_is_explicit,
    )


def resolve_source(
    source: str,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Resolve any supported source spec to a raw-content URL.

    If ``source`` is a ``gh:`` URI with no explicit ``:path`` and the
    default path 404s, the repo root is inspected via the GitHub
    Contents API to locate a single ``*.yaml``/``*.yml`` file.

    ``client`` is an optional pre-configured :class:`httpx.Client`
    (used by tests with ``MockTransport``); when omitted, a fresh
    client with a 10-second timeout is created for the call.
    """
    if source.startswith("gh:"):
        parsed = parse_gh_source(source)
        return _resolve_github(parsed, client=client)
    if source.startswith("https://raw.githubusercontent.com/"):
        return source
    if source.startswith("https://github.com/"):
        return _rewrite_blob_url(source)
    raise ValueError(
        f"unsupported pack source {source!r} "
        "(expected gh:user/repo..., https://github.com/... or https://raw.githubusercontent.com/...)"
    )


def _resolve_github(
    parsed: GitHubSource,
    *,
    client: httpx.Client | None,
) -> str:
    default_url = _raw_url(parsed.user, parsed.repo, parsed.ref, parsed.path)
    if parsed.path_is_explicit:
        return default_url

    owns_client = client is None
    client = client or httpx.Client(timeout=_HTTP_TIMEOUT, follow_redirects=True)
    try:
        resp = client.head(default_url)
        if resp.status_code == 200:
            return default_url
        if resp.status_code == 404:
            return _fallback_root_yaml(parsed, client)
        raise ValueError(
            f"unexpected HTTP {resp.status_code} probing {default_url} "
            "(expected 200 or 404)"
        )
    finally:
        if owns_client:
            client.close()


def _fallback_root_yaml(parsed: GitHubSource, client: httpx.Client) -> str:
    api_url = (
        f"{GITHUB_API_BASE}/repos/{parsed.user}/{parsed.repo}/contents/"
        f"?ref={parsed.ref}"
    )
    resp = client.get(api_url)
    if resp.status_code == 404:
        raise ValueError(
            f"repo or ref not found: {parsed.user}/{parsed.repo}@{parsed.ref}"
        )
    if resp.status_code != 200:
        raise ValueError(
            f"cannot list repo contents (HTTP {resp.status_code}): {api_url}"
        )

    try:
        items = resp.json()
    except ValueError as exc:
        raise ValueError(f"malformed JSON from {api_url}: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError(
            f"expected a list of repo contents from {api_url}, got {type(items).__name__}"
        )

    yamls = [
        item["name"]
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "file"
        and isinstance(item.get("name"), str)
        and (item["name"].endswith(".yaml") or item["name"].endswith(".yml"))
    ]
    if not yamls:
        raise ValueError(
            f"no *.yaml at root of {parsed.user}/{parsed.repo}@{parsed.ref}; "
            f"expected {DEFAULT_PATH} or a single root-level yaml"
        )
    if len(yamls) > 1:
        raise ValueError(
            f"multiple yaml files at root of {parsed.user}/{parsed.repo}@{parsed.ref}: "
            f"{sorted(yamls)}; specify one explicitly with gh:{parsed.user}/{parsed.repo}"
            f"{'@' + parsed.ref if parsed.ref != DEFAULT_REF else ''}:<path>"
        )
    return _raw_url(parsed.user, parsed.repo, parsed.ref, yamls[0])


_BLOB_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/blob/(.+)$")


def _rewrite_blob_url(url: str) -> str:
    match = _BLOB_RE.match(url)
    if not match:
        raise ValueError(
            f"unsupported github.com URL {url!r} "
            "(expected https://github.com/user/repo/blob/ref/path)"
        )
    user, repo, tail = match.group(1), match.group(2), match.group(3)
    ref, _, path = tail.partition("/")
    if not ref or not path:
        raise ValueError(
            f"malformed github.com/blob/ URL {url!r} "
            "(expected .../blob/<ref>/<path>)"
        )
    return _raw_url(user, repo, ref, path)


def _raw_url(user: str, repo: str, ref: str, path: str) -> str:
    return f"{RAW_BASE}/{user}/{repo}/{ref}/{path}"
