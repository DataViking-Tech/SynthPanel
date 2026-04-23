"""Tests for the GitHub URL resolver (``synth_panel.registry.github``)."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from synth_panel.registry.github import (
    DEFAULT_PATH,
    DEFAULT_REF,
    GitHubSource,
    parse_gh_source,
    resolve_source,
)

# ---------- parse_gh_source ----------


def test_parse_gh_default_ref_and_path() -> None:
    parsed = parse_gh_source("gh:owner/repo")
    assert parsed == GitHubSource(
        user="owner",
        repo="repo",
        ref=DEFAULT_REF,
        path=DEFAULT_PATH,
        path_is_explicit=False,
    )


def test_parse_gh_with_ref() -> None:
    parsed = parse_gh_source("gh:owner/repo@v1.2")
    assert parsed.ref == "v1.2"
    assert parsed.path == DEFAULT_PATH
    assert parsed.path_is_explicit is False


def test_parse_gh_with_explicit_path() -> None:
    parsed = parse_gh_source("gh:owner/repo:packs/custom.yaml")
    assert parsed.ref == DEFAULT_REF
    assert parsed.path == "packs/custom.yaml"
    assert parsed.path_is_explicit is True


def test_parse_gh_with_ref_and_path() -> None:
    parsed = parse_gh_source("gh:owner/repo@abc123:nested/dir/pack.yaml")
    assert parsed.ref == "abc123"
    assert parsed.path == "nested/dir/pack.yaml"
    assert parsed.path_is_explicit is True


def test_parse_gh_repo_name_with_dots() -> None:
    parsed = parse_gh_source("gh:my-org/repo.name")
    assert parsed.user == "my-org"
    assert parsed.repo == "repo.name"


def test_parse_gh_path_with_at_sign_after_colon() -> None:
    # When ':' precedes '@', '@' must be treated as part of the path,
    # not a ref delimiter.
    parsed = parse_gh_source("gh:u/r:weird@name.yaml")
    assert parsed.ref == DEFAULT_REF
    assert parsed.path == "weird@name.yaml"


@pytest.mark.parametrize(
    "source",
    [
        "gh:",
        "gh:user",
        "gh:/repo",
        "gh:user/",
        "gh:@ref",
        "gh:user/repo@",
        "gh:user/repo:",
        "gh:user/repo@ref:",
        "gh:user/repo/extra",
    ],
)
def test_parse_gh_malformed_raises(source: str) -> None:
    with pytest.raises(ValueError):
        parse_gh_source(source)


def test_parse_gh_non_gh_source_raises() -> None:
    with pytest.raises(ValueError):
        parse_gh_source("https://example.com/pack.yaml")


# ---------- resolve_source (gh: variants, explicit paths) ----------


def _explode(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected network call: {request.method} {request.url}")


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolve_explicit_path_never_calls_network() -> None:
    with _client(_explode) as client:
        url = resolve_source("gh:owner/repo@v1:pkgs/my.yaml", client=client)
    assert url == "https://raw.githubusercontent.com/owner/repo/v1/pkgs/my.yaml"


def test_resolve_default_path_hit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        assert request.url.host == "raw.githubusercontent.com"
        assert request.url.path == f"/owner/repo/{DEFAULT_REF}/{DEFAULT_PATH}"
        return httpx.Response(200)

    with _client(handler) as client:
        url = resolve_source("gh:owner/repo", client=client)
    assert url == f"https://raw.githubusercontent.com/owner/repo/{DEFAULT_REF}/{DEFAULT_PATH}"


def test_resolve_default_path_404_fallback_single_yaml() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            assert request.method == "HEAD"
            return httpx.Response(404)
        assert request.url.host == "api.github.com"
        assert request.url.path == "/repos/owner/repo/contents/"
        assert request.url.params.get("ref") == DEFAULT_REF
        return httpx.Response(
            200,
            json=[
                {"name": "README.md", "type": "file"},
                {"name": "docs", "type": "dir"},
                {"name": "only-pack.yaml", "type": "file"},
            ],
        )

    with _client(handler) as client:
        url = resolve_source("gh:owner/repo", client=client)
    assert url == "https://raw.githubusercontent.com/owner/repo/main/only-pack.yaml"


def test_resolve_default_path_404_fallback_single_yml_extension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json=[{"name": "spec.yml", "type": "file"}],
        )

    with _client(handler) as client:
        url = resolve_source("gh:owner/repo", client=client)
    assert url.endswith("/spec.yml")


def test_resolve_default_path_404_fallback_zero_yamls_rejects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json=[
                {"name": "README.md", "type": "file"},
                {"name": "LICENSE", "type": "file"},
            ],
        )

    with _client(handler) as client, pytest.raises(ValueError, match="no .*yaml"):
        resolve_source("gh:owner/repo", client=client)


def test_resolve_default_path_404_fallback_multiple_yamls_rejects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json=[
                {"name": "one.yaml", "type": "file"},
                {"name": "two.yaml", "type": "file"},
            ],
        )

    with _client(handler) as client, pytest.raises(ValueError, match="multiple yaml"):
        resolve_source("gh:owner/repo", client=client)


def test_resolve_default_path_404_api_404_reports_missing_repo() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(404)
        return httpx.Response(404, json={"message": "Not Found"})

    with _client(handler) as client, pytest.raises(ValueError, match="repo or ref not found"):
        resolve_source("gh:owner/repo", client=client)


def test_resolve_default_path_unexpected_status_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with _client(handler) as client, pytest.raises(ValueError, match="HTTP 500"):
        resolve_source("gh:owner/repo", client=client)


def test_resolve_default_path_fallback_ignores_directories() -> None:
    # A directory whose name ends in .yaml should not be treated as a file.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            return httpx.Response(404)
        return httpx.Response(
            200,
            json=[
                {"name": "fake.yaml", "type": "dir"},
                {"name": "real-pack.yaml", "type": "file"},
            ],
        )

    with _client(handler) as client:
        url = resolve_source("gh:owner/repo", client=client)
    assert url.endswith("/real-pack.yaml")


def test_resolve_fallback_honors_ref() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "raw.githubusercontent.com":
            assert request.url.path == "/owner/repo/v2.0/synthpanel-pack.yaml"
            return httpx.Response(404)
        assert request.url.params.get("ref") == "v2.0"
        return httpx.Response(200, json=[{"name": "alt.yaml", "type": "file"}])

    with _client(handler) as client:
        url = resolve_source("gh:owner/repo@v2.0", client=client)
    assert url == "https://raw.githubusercontent.com/owner/repo/v2.0/alt.yaml"


# ---------- resolve_source (https passthrough / blob rewrite) ----------


def test_resolve_raw_url_passthrough_no_network() -> None:
    url = "https://raw.githubusercontent.com/u/r/main/pack.yaml"
    with _client(_explode) as client:
        assert resolve_source(url, client=client) == url


def test_resolve_blob_url_rewritten_no_network() -> None:
    with _client(_explode) as client:
        url = resolve_source(
            "https://github.com/owner/repo/blob/v1.0/packs/pack.yaml",
            client=client,
        )
    assert url == "https://raw.githubusercontent.com/owner/repo/v1.0/packs/pack.yaml"


def test_resolve_blob_url_with_nested_path() -> None:
    url = resolve_source(
        "https://github.com/owner/repo/blob/main/a/b/c/pack.yaml",
    )
    assert url == "https://raw.githubusercontent.com/owner/repo/main/a/b/c/pack.yaml"


def test_resolve_github_non_blob_url_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported github.com URL"):
        resolve_source("https://github.com/owner/repo/tree/main")


def test_resolve_github_blob_missing_path_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_source("https://github.com/owner/repo/blob/main")


def test_resolve_unsupported_scheme_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported pack source"):
        resolve_source("ftp://example.com/pack.yaml")
    with pytest.raises(ValueError, match="unsupported pack source"):
        resolve_source("/local/path.yaml")
