"""Live-registry acceptance smoke test (sp-m1mz).

Hits the real production registry URL — no mocks, no fixtures — and
verifies both the shape of the live JSON and that :func:`resolve_pack`
can hydrate the seeded ``icp-traitprint-cloud`` entry end to end.

Skipped by default (``-m 'not acceptance'`` in ``pyproject.toml``).
Run explicitly with::

    pytest tests/acceptance/test_registry_live.py -m acceptance

The test uses a throwaway cache dir via ``SYNTH_PANEL_DATA_DIR`` so it
never pollutes the user's real ``~/.synthpanel`` cache, and forces a
fresh network fetch so the assertions always reflect live state.
"""

from __future__ import annotations

import pytest

from synth_panel.registry import (
    DATA_DIR_ENV,
    DEFAULT_REGISTRY_URL,
    RegistryEntry,
    RegistryFetchError,
    fetch_registry_http,
    resolve_pack,
)

pytestmark = pytest.mark.acceptance


LIVE_URL = DEFAULT_REGISTRY_URL
SEED_PACK_ID = "icp-traitprint-cloud"

# Fields required by RegistryEntry.from_dict() + the registry envelope.
# These are the contract the live default.json must honor; if any is
# missing, the assertion message names it explicitly so a broken seed
# is debuggable from the CI log alone.
REQUIRED_ENVELOPE_FIELDS = ("schema_version", "packs")
REQUIRED_PACK_FIELDS = ("id", "repo", "path")


def _fetch_live() -> dict:
    """Fetch the live registry with explicit error messages for flake debugging.

    ``fetch_registry_http`` wraps every transport/HTTP/JSON failure mode in
    :class:`RegistryFetchError` whose string form already names the URL and
    either the HTTP status code or the underlying network error — so the
    raised exception is the debuggable artifact we want in CI logs.
    """
    try:
        result = fetch_registry_http(LIVE_URL)
    except RegistryFetchError as exc:
        pytest.fail(f"live registry fetch failed: {exc}")
    assert result.data is not None, (
        f"live registry GET {LIVE_URL} returned no body (not_modified={result.not_modified})"
    )
    return result.data


def test_live_registry_envelope_is_schema_valid() -> None:
    """The live default.json matches the documented envelope schema."""
    body = _fetch_live()

    assert isinstance(body, dict), f"expected JSON object at {LIVE_URL}, got {type(body).__name__}"
    for field in REQUIRED_ENVELOPE_FIELDS:
        assert field in body, (
            f"live registry at {LIVE_URL} is missing required envelope field {field!r}; "
            f"present keys: {sorted(body.keys())}"
        )

    assert isinstance(body["schema_version"], int), (
        f"schema_version must be int, got {type(body['schema_version']).__name__}"
    )
    assert body["schema_version"] == 1, (
        f"unexpected schema_version {body['schema_version']!r} at {LIVE_URL} (this test targets schema v1)"
    )

    packs = body["packs"]
    assert isinstance(packs, list), f"'packs' must be a list, got {type(packs).__name__}"
    assert packs, f"live registry at {LIVE_URL} has an empty 'packs' list"

    for idx, entry in enumerate(packs):
        assert isinstance(entry, dict), f"packs[{idx}] must be an object, got {type(entry).__name__}"
        for field in REQUIRED_PACK_FIELDS:
            assert field in entry, (
                f"packs[{idx}] (id={entry.get('id', '<unknown>')!r}) is missing "
                f"required field {field!r}; present keys: {sorted(entry.keys())}"
            )


def test_resolve_pack_hydrates_seed_entry(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``resolve_pack('icp-traitprint-cloud')`` returns a hydrated RegistryEntry.

    Uses a throwaway ``SYNTH_PANEL_DATA_DIR`` + ``refresh=True`` so the
    live registry is re-fetched for this assertion regardless of any
    pre-existing cache state on the runner.
    """
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("SYNTHPANEL_REGISTRY_OFFLINE", raising=False)
    monkeypatch.delenv("SYNTHPANEL_REGISTRY_URL", raising=False)

    entry = resolve_pack(SEED_PACK_ID, refresh=True)

    assert entry is not None, (
        f"resolve_pack({SEED_PACK_ID!r}) returned None against live registry "
        f"{LIVE_URL}; seed entry #7 ('seed icp-traitprint-cloud') may have been "
        f"reverted or the registry JSON may be broken"
    )
    assert isinstance(entry, RegistryEntry), f"expected RegistryEntry, got {type(entry).__name__}"
    assert entry.id == SEED_PACK_ID, f"entry.id mismatch: {entry.id!r} != {SEED_PACK_ID!r}"
    assert entry.repo, f"entry.repo is empty for {SEED_PACK_ID!r}"
    assert entry.path, f"entry.path is empty for {SEED_PACK_ID!r}"
    # ref has a schema-level default of 'main' so it must never be empty
    # after hydration — catches regressions where from_dict silently
    # drops the field.
    assert entry.ref, f"entry.ref is empty for {SEED_PACK_ID!r}"
