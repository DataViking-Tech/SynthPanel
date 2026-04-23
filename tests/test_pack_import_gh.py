"""Tests for the remote (``gh:``/``https://…``) branch of ``pack import``.

Covers the registry consultation → collision → save flow introduced
for sp-w9a5:

- ``gh:`` happy path with cached registry containing the source
  (silent success, one confirmation line, no warning block).
- ``--unverified`` required path when the source isn't in the registry;
  failure message guides users to re-run with the flag.
- ``--unverified`` warning block (URL, sha256 checksum, imported id).
- ``--unverified`` flag-unnecessary notice when the source IS already
  in the registry.
- Collision: bundled-id refused with ``--id`` hint.
- Collision: user-saved id refused without ``--force``; accepted with it.
- 404 handling with the private-repo / GITHUB_TOKEN hint.
- Local-path ``pack import`` regression (unchanged behavior).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from synth_panel.main import main
from synth_panel.registry import cache as registry_cache

SAMPLE_PACK_YAML = """\
name: Example Pack
personas:
  - name: Alice
    age: 34
    occupation: Engineer
  - name: Bob
    age: 41
    occupation: PM
"""

REGISTRY_WITH_ENTRY = {
    "schema_version": 1,
    "generated_at": "2026-04-22T00:00:00Z",
    "packs": [
        {
            "id": "icp-demo",
            "kind": "persona",
            "name": "Demo Pack",
            "description": "Example registered pack.",
            "repo": "example/demo",
            "path": "synthpanel-pack.yaml",
            "ref": "main",
            "author": {"github": "example"},
            "added_at": "2026-04-22",
            "calibration": None,
        }
    ],
}

EMPTY_REGISTRY = {
    "schema_version": 1,
    "generated_at": "2026-04-22T00:00:00Z",
    "packs": [],
}


def _install_httpx_mock(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    """Replace ``httpx.Client`` with a MockTransport-backed stand-in.

    Every ``httpx.Client(...)`` call in the target code path routes to
    *handler* instead of the real network.
    """
    transport = httpx.MockTransport(handler)
    real_init = httpx.Client.__init__

    def _patched_init(self: httpx.Client, *args: object, **kwargs: object) -> None:
        kwargs["transport"] = transport
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", _patched_init)


def _seed_registry_cache(tmp_path, registry: dict) -> None:
    """Pre-populate the on-disk registry cache with *registry*.

    Written to ``$SYNTH_PANEL_DATA_DIR/registry-cache.json`` so
    ``fetch_registry()`` returns it without any network call.
    """
    from datetime import datetime, timezone

    cache_dir = tmp_path
    cache_file = cache_dir / registry_cache.CACHE_FILENAME
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_url": "https://example/registry.json",
        "etag": None,
        "registry": registry,
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH_PANEL_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# gh: happy paths
# ---------------------------------------------------------------------------


class TestGhHappyPath:
    def test_registered_source_succeeds_silently(self, tmp_data_dir, monkeypatch, capsys):
        """Registered ``gh:`` import: exit 0, one confirmation line, no warnings."""
        _seed_registry_cache(tmp_data_dir, REGISTRY_WITH_ENTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("synthpanel-pack.yaml"):
                return httpx.Response(200, text=SAMPLE_PACK_YAML)
            return httpx.Response(404)

        _install_httpx_mock(monkeypatch, handler)

        code = main(["pack", "import", "gh:example/demo"])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "Imported pack" in captured.out
        assert "not in the synthpanel registry" not in captured.err
        assert "not in the synthpanel registry" not in captured.out

    def test_unregistered_requires_unverified_flag(self, tmp_data_dir, monkeypatch, capsys):
        """Unregistered ``gh:`` source without ``--unverified`` fails with guidance."""
        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(["pack", "import", "gh:stranger/pack"])
        err = capsys.readouterr().err
        assert code == 1
        assert "not in the synthpanel registry" in err
        assert "--unverified" in err

    def test_unverified_prints_warning_block(self, tmp_data_dir, monkeypatch, capsys):
        """``--unverified`` on unregistered source: warning block with URL + sha256 + id."""
        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(["pack", "import", "gh:stranger/pack", "--unverified"])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "not in the synthpanel registry" in captured.err
        assert "raw.githubusercontent.com/stranger/pack/main/synthpanel-pack.yaml" in captured.err
        assert "sha256:" in captured.err
        assert "Imported as: pack" in captured.err  # default id from repo slug
        assert "Imported pack" in captured.out

    def test_unverified_on_registered_source_warns_flag_unnecessary(self, tmp_data_dir, monkeypatch, capsys):
        """``--unverified`` on a registered source: note that the flag is unnecessary."""
        _seed_registry_cache(tmp_data_dir, REGISTRY_WITH_ENTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(["pack", "import", "gh:example/demo", "--unverified"])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "flag unnecessary" in captured.err
        # The section-4 warning block must NOT fire for a registered source.
        assert "Checksum:" not in captured.err
        assert "Imported pack" in captured.out


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------


class TestCollisions:
    def test_bundled_id_collision_refuses_with_id_hint(self, tmp_data_dir, monkeypatch, capsys):
        """Remote import with bundled id refuses and hints at ``--id``."""
        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(
            [
                "pack",
                "import",
                "gh:stranger/pack",
                "--unverified",
                "--id",
                "developer",  # bundled pack id
            ]
        )
        err = capsys.readouterr().err
        assert code == 1
        assert "bundled pack" in err
        assert "--id" in err

    def test_saved_id_collision_requires_force(self, tmp_data_dir, monkeypatch, capsys):
        """Remote import colliding with a saved pack id refuses without ``--force``."""
        from synth_panel.mcp.data import save_persona_pack

        save_persona_pack("Existing", [{"name": "Pre-existing"}], pack_id="mine")

        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(
            [
                "pack",
                "import",
                "gh:stranger/pack",
                "--unverified",
                "--id",
                "mine",
            ]
        )
        err = capsys.readouterr().err
        assert code == 1
        assert "already exists" in err
        assert "--force" in err

    def test_saved_id_collision_accepts_force(self, tmp_data_dir, monkeypatch, capsys):
        """``--force`` overwrites an existing user-saved pack."""
        from synth_panel.mcp.data import get_persona_pack, save_persona_pack

        save_persona_pack("Existing", [{"name": "Pre-existing"}], pack_id="mine")

        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=SAMPLE_PACK_YAML)

        _install_httpx_mock(monkeypatch, handler)

        code = main(
            [
                "pack",
                "import",
                "gh:stranger/pack",
                "--unverified",
                "--id",
                "mine",
                "--force",
            ]
        )
        captured = capsys.readouterr()
        assert code == 0, captured.err
        # Overwritten: new content should reflect the remote personas.
        pack = get_persona_pack("mine")
        names = {p.get("name") for p in pack.get("personas", [])}
        assert names == {"Alice", "Bob"}


# ---------------------------------------------------------------------------
# fetch failures
# ---------------------------------------------------------------------------


class TestFetchFailures:
    def test_404_hints_at_private_repo(self, tmp_data_dir, monkeypatch, capsys):
        """Explicit-path gh: 404 mentions private-repo + GITHUB_TOKEN."""
        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404)

        _install_httpx_mock(monkeypatch, handler)

        # Explicit path bypasses the root-yaml fallback probe in the resolver.
        code = main(["pack", "import", "gh:ghost/vanished:synthpanel-pack.yaml", "--unverified"])
        err = capsys.readouterr().err
        assert code == 1
        assert "private" in err.lower()
        assert "GITHUB_TOKEN" in err

    def test_malformed_yaml_rejected(self, tmp_data_dir, monkeypatch, capsys):
        """Invalid YAML at the resolved URL fails cleanly."""
        _seed_registry_cache(tmp_data_dir, EMPTY_REGISTRY)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=": : : not valid yaml : :\n  :")

        _install_httpx_mock(monkeypatch, handler)

        code = main(["pack", "import", "gh:stranger/pack:pack.yaml", "--unverified"])
        err = capsys.readouterr().err
        assert code == 1
        assert "invalid YAML" in err or "Validation error" in err


# ---------------------------------------------------------------------------
# local-path regression
# ---------------------------------------------------------------------------


class TestLocalPathRegression:
    def test_local_path_import_unchanged(self, tmp_data_dir, capsys):
        """Local YAML import goes through the unchanged local branch."""
        pfile = tmp_data_dir / "local-pack.yaml"
        pfile.write_text("name: Local\npersonas:\n  - name: Cara\n    age: 29\n")

        code = main(["pack", "import", str(pfile)])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        assert "Imported pack 'Local'" in captured.out
        # Remote-only features must not appear in the local branch.
        assert "Checksum:" not in captured.err
        assert "synthpanel registry" not in captured.err
