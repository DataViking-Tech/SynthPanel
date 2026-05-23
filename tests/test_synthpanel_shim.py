"""sy-het: the ``synthpanel`` compatibility shim.

The PyPI package and CLI are ``synthpanel`` (one word), but the importable
Python module has always been ``synth_panel`` (two words). Agents that infer
``python -m <pypi-name>`` from the package name hit ``ModuleNotFoundError``.
This module ships a thin alias so the obvious-looking forms work.

These tests pin the externally-observable contract:

* ``python -m synthpanel --version`` is identical to ``python -m synth_panel --version``.
* ``import synthpanel`` exposes the same ``__version__`` and re-exports
  the canonical ``__all__`` surface.
* ``from synthpanel import sdk`` resolves to a usable module (functions are
  callable). Module identity is intentionally not asserted — the
  ``__path__`` redirect can mint a sibling module under the alias for
  files that haven't been canonically imported yet, and that is OK for
  the use case (calling functions, reading ``__version__``).
"""

from __future__ import annotations

import subprocess
import sys


def test_python_m_synthpanel_version_matches_canonical() -> None:
    """sy-het / #509: ``python -m synthpanel --version`` works and matches the canonical form."""
    alias = subprocess.run(
        [sys.executable, "-m", "synthpanel", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    canonical = subprocess.run(
        [sys.executable, "-m", "synth_panel", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    # argparse prints "synthpanel <version>" to stdout in both cases.
    assert alias.stdout.strip(), "alias produced no version output"
    assert alias.stdout == canonical.stdout, (
        f"alias output {alias.stdout!r} differs from canonical {canonical.stdout!r}"
    )
    # Both forms should exit 0 on --version.
    assert alias.returncode == 0
    assert canonical.returncode == 0


def test_python_m_synthpanel_help_works() -> None:
    """sy-het: the shim doesn't break ``--help`` either.

    Catches the failure mode where the shim's ``__main__.py`` re-imports
    the CLI in a way that breaks argparse's auto-help.
    """
    result = subprocess.run(
        [sys.executable, "-m", "synthpanel", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    # Spot-check for content from the top-level parser so a future
    # accidental empty-stub regression fails loudly.
    assert "synthpanel" in result.stdout.lower()
    assert "panel" in result.stdout.lower() or "prompt" in result.stdout.lower()


def test_import_synthpanel_exposes_version() -> None:
    """``import synthpanel`` works and ``synthpanel.__version__`` matches the canonical."""
    import synth_panel
    import synthpanel

    assert synthpanel.__version__ == synth_panel.__version__


def test_synthpanel_reexports_canonical_public_surface() -> None:
    """``synthpanel.__all__`` mirrors ``synth_panel.__all__``.

    Anything an agent imports via the public ``from X import Y`` path on the
    canonical name must also be importable via the alias — otherwise the
    shim breaks the "obvious-looking form" it exists to fix.
    """
    import synth_panel
    import synthpanel

    assert set(synthpanel.__all__) == set(synth_panel.__all__)
    for name in synth_panel.__all__:
        assert hasattr(synthpanel, name), f"synthpanel missing public re-export: {name}"


def test_from_synthpanel_import_submodule_works() -> None:
    """``from synthpanel import sdk`` resolves to a usable module."""
    from synthpanel import sdk

    # The exact module identity isn't part of the contract (see module
    # docstring), but the imported object must expose the canonical SDK
    # entry points and they must be callable.
    assert callable(sdk.run_panel)
    assert callable(sdk.quick_poll)


def test_from_synthpanel_dotted_path_import_works() -> None:
    """``from synthpanel.cli import commands`` resolves through the shim.

    Exercises the ``__path__`` redirect at depth > 1 so a regression that
    breaks deep imports lands here, not in a downstream consumer.
    """
    from synthpanel.cli import commands

    assert hasattr(commands, "handle_poll_summary")


def test_synthpanel_path_redirects_to_canonical_source_tree() -> None:
    """The shim's ``__path__`` points at the canonical source tree.

    This is the implementation hinge: setuptools finds ``synthpanel`` as a
    real package, but ``synthpanel.__path__`` aims Python's default
    ``PathFinder`` at the ``synth_panel/`` directory so submodule lookups
    resolve against the canonical source. If this stops pointing at
    ``synth_panel/``, the dotted-path import tests above silently start
    minting copies of every file under both names — easy to miss without
    a direct assertion.
    """
    import synth_panel
    import synthpanel

    assert list(synthpanel.__path__) == list(synth_panel.__path__)
