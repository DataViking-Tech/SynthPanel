"""sy-het: the ``althing`` compatibility shim.

The PyPI package and CLI are ``althing`` (one word), but the importable
Python module has always been ``althing`` (two words). Agents that infer
``python -m <pypi-name>`` from the package name hit ``ModuleNotFoundError``.
This module ships a thin alias so the obvious-looking forms work.

These tests pin the externally-observable contract:

* ``python -m althing --version`` is identical to ``python -m althing --version``.
* ``import althing`` exposes the same ``__version__`` and re-exports
  the canonical ``__all__`` surface.
* ``from althing import sdk`` resolves to a usable module (functions are
  callable). Module identity is intentionally not asserted — the
  ``__path__`` redirect can mint a sibling module under the alias for
  files that haven't been canonically imported yet, and that is OK for
  the use case (calling functions, reading ``__version__``).
"""

from __future__ import annotations

import subprocess
import sys


def test_python_m_althing_version_matches_canonical() -> None:
    """sy-het / #509: ``python -m althing --version`` works and matches the canonical form."""
    alias = subprocess.run(
        [sys.executable, "-m", "althing", "--version"],
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
    # argparse prints "althing <version>" to stdout in both cases.
    assert alias.stdout.strip(), "alias produced no version output"
    assert alias.stdout == canonical.stdout, (
        f"alias output {alias.stdout!r} differs from canonical {canonical.stdout!r}"
    )
    # Both forms should exit 0 on --version.
    assert alias.returncode == 0
    assert canonical.returncode == 0


def test_python_m_althing_help_works() -> None:
    """sy-het: the shim doesn't break ``--help`` either.

    Catches the failure mode where the shim's ``__main__.py`` re-imports
    the CLI in a way that breaks argparse's auto-help.
    """
    result = subprocess.run(
        [sys.executable, "-m", "althing", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    # Spot-check for content from the top-level parser so a future
    # accidental empty-stub regression fails loudly.
    assert "althing" in result.stdout.lower()
    assert "panel" in result.stdout.lower() or "prompt" in result.stdout.lower()


def test_import_althing_exposes_version() -> None:
    """``import althing`` works and ``althing.__version__`` matches the canonical."""
    import althing
    import althing

    assert althing.__version__ == althing.__version__


def test_althing_reexports_canonical_public_surface() -> None:
    """``althing.__all__`` mirrors ``althing.__all__``.

    Anything an agent imports via the public ``from X import Y`` path on the
    canonical name must also be importable via the alias — otherwise the
    shim breaks the "obvious-looking form" it exists to fix.
    """
    import althing
    import althing

    assert set(althing.__all__) == set(althing.__all__)
    for name in althing.__all__:
        assert hasattr(althing, name), f"althing missing public re-export: {name}"


def test_from_althing_import_submodule_works() -> None:
    """``from althing import sdk`` resolves to a usable module."""
    from althing import sdk

    # The exact module identity isn't part of the contract (see module
    # docstring), but the imported object must expose the canonical SDK
    # entry points and they must be callable.
    assert callable(sdk.run_panel)
    assert callable(sdk.quick_poll)


def test_from_althing_dotted_path_import_works() -> None:
    """``from althing.cli import commands`` resolves through the shim.

    Exercises the ``__path__`` redirect at depth > 1 so a regression that
    breaks deep imports lands here, not in a downstream consumer.
    """
    from althing.cli import commands

    assert hasattr(commands, "handle_poll_summary")


def test_althing_path_redirects_to_canonical_source_tree() -> None:
    """The shim's ``__path__`` points at the canonical source tree.

    This is the implementation hinge: setuptools finds ``althing`` as a
    real package, but ``althing.__path__`` aims Python's default
    ``PathFinder`` at the ``althing/`` directory so submodule lookups
    resolve against the canonical source. If this stops pointing at
    ``althing/``, the dotted-path import tests above silently start
    minting copies of every file under both names — easy to miss without
    a direct assertion.
    """
    import althing
    import althing

    assert list(althing.__path__) == list(althing.__path__)
