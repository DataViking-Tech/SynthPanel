"""Rename compatibility shims: ``synthpanel`` and ``synth_panel`` → ``althing``.

After the 2026-07 rename the canonical package, CLI, and module are all
``althing``. Two legacy import names remain for one deprecation cycle:

* ``synthpanel`` — the pre-rename PyPI/CLI name (and one-word module shim
  from #509)
* ``synth_panel`` — the pre-rename canonical (PEP 8) module name

These tests pin the externally-observable contract:

* ``python -m synthpanel`` / ``python -m synth_panel`` behave identically to
  ``python -m althing`` (plus a rename notice on stderr).
* Importing either legacy name emits a ``DeprecationWarning`` and exposes the
  canonical ``__version__`` and ``__all__`` surface.
* Deep/dotted imports resolve through the ``__path__`` redirect, and plain
  attribute access reaches loaded submodules via the PEP 562 ``__getattr__``
  delegation (``synthpanel.sdk`` after ``import althing.sdk``).
* Legacy ``SYNTHPANEL_*`` environment variables are bridged to ``ALTHING_*``
  at first import of ``althing``.

Module identity for lazily-imported submodules is intentionally not asserted
— the ``__path__`` redirect can mint a sibling module under an alias for
files never canonically imported, and that is OK for the use case.
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

LEGACY_MODULES = ["synthpanel", "synth_panel"]


@pytest.mark.parametrize("alias", LEGACY_MODULES)
def test_python_m_legacy_version_matches_canonical(alias: str) -> None:
    """``python -m <legacy> --version`` works and matches ``python -m althing``."""
    legacy = subprocess.run(
        [sys.executable, "-m", alias, "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    canonical = subprocess.run(
        [sys.executable, "-m", "althing", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert legacy.stdout.strip(), "legacy alias produced no version output"
    assert legacy.stdout == canonical.stdout, (
        f"{alias} output {legacy.stdout!r} differs from canonical {canonical.stdout!r}"
    )
    # The legacy entry announces the rename on stderr without breaking stdout.
    assert "althing" in legacy.stderr.lower()
    assert legacy.returncode == 0
    assert canonical.returncode == 0


def test_python_m_canonical_help_works() -> None:
    """``python -m althing --help`` — argparse auto-help intact post-rename."""
    result = subprocess.run(
        [sys.executable, "-m", "althing", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "althing" in result.stdout.lower()
    assert "panel" in result.stdout.lower() or "prompt" in result.stdout.lower()


@pytest.mark.parametrize("alias", LEGACY_MODULES)
def test_import_legacy_warns_and_exposes_version(alias: str) -> None:
    """Importing a legacy name warns (once, on first import) and matches versions."""
    # A fresh interpreter gives a clean sys.modules so the import-time
    # DeprecationWarning is actually observable regardless of test order.
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    import {alias}\n"
        "import althing\n"
        f"assert {alias}.__version__ == althing.__version__\n"
        "assert any(issubclass(w.category, DeprecationWarning) for w in caught), (\n"
        "    'no DeprecationWarning on legacy import'\n"
        ")\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("alias", LEGACY_MODULES)
def test_legacy_reexports_canonical_public_surface(alias: str) -> None:
    """The legacy module's ``__all__`` mirrors ``althing.__all__``."""
    import althing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = __import__(alias)

    assert set(legacy.__all__) == set(althing.__all__)
    for name in althing.__all__:
        assert hasattr(legacy, name), f"{alias} missing public re-export: {name}"


def test_from_legacy_import_submodule_works() -> None:
    """``from synthpanel import sdk`` resolves to a usable module."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from synthpanel import sdk

    assert callable(sdk.run_panel)
    assert callable(sdk.quick_poll)


def test_from_legacy_dotted_path_import_works() -> None:
    """``from synthpanel.cli import commands`` resolves through the shim."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from synthpanel.cli import commands

    assert hasattr(commands, "handle_poll_summary")


def test_legacy_attribute_reaches_canonically_loaded_submodule() -> None:
    """PEP 562 delegation: ``synthpanel.sdk`` after ``import althing.sdk``.

    The ``sys.modules`` mirror can't bind parent-package attributes the way
    a real import does; without ``__getattr__`` delegation this raised
    ``AttributeError`` (caught during the rename).
    """
    import althing.sdk

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import synthpanel

    assert synthpanel.sdk is althing.sdk


@pytest.mark.parametrize("alias", LEGACY_MODULES)
def test_legacy_path_redirects_to_canonical_source_tree(alias: str) -> None:
    """Each legacy module's ``__path__`` falls through to the althing tree.

    Shape: ``[<shim's own dir>, *althing.__path__]`` — the shim dir comes
    first so its ``__main__.py`` (rename notice) wins, and the canonical
    tree follows so every other submodule lookup resolves against
    ``althing/``. If the althing suffix disappears, the dotted-path imports
    above silently start minting copies of every file under both names.
    """
    import althing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = __import__(alias)

    paths = list(legacy.__path__)
    assert paths[-len(list(althing.__path__)) :] == list(althing.__path__)
    assert paths[0].rstrip("/").endswith(alias), (
        f"first __path__ entry should be the {alias} shim dir, got {paths[0]!r}"
    )


def test_legacy_env_vars_bridge_to_althing() -> None:
    """``SYNTHPANEL_*`` env vars are honored via the ``ALTHING_*`` bridge."""
    code = (
        "import os, warnings\n"
        "os.environ['SYNTHPANEL_LOG_LEVEL'] = 'WARNING'\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import althing\n"
        "assert os.environ.get('ALTHING_LOG_LEVEL') == 'WARNING', 'bridge missed'\n"
        "assert any('SYNTHPANEL_' in str(w.message) for w in caught), 'no warning'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_env_bridge_does_not_override_explicit_althing_value() -> None:
    """An explicit ``ALTHING_*`` value wins over the legacy bridge."""
    code = (
        "import os\n"
        "os.environ['SYNTHPANEL_LOG_LEVEL'] = 'DEBUG'\n"
        "os.environ['ALTHING_LOG_LEVEL'] = 'ERROR'\n"
        "import althing\n"
        "assert os.environ['ALTHING_LOG_LEVEL'] == 'ERROR'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
