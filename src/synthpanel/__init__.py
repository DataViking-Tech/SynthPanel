"""Compatibility shim — re-exports the ``synth_panel`` package (sy-het).

The PyPI distribution and CLI script are both named ``synthpanel`` (one word),
but the importable Python module has always been ``synth_panel`` (PEP 8). That
mismatch trips agents that infer ``python -m <pypi-name>`` from the package
name, so we ship a one-word ``synthpanel`` module alongside the canonical
``synth_panel`` one.

Everything here is a re-export. ``synth_panel`` remains the source of truth —
this module exists so the obvious-looking forms keep working:

* ``python -m synthpanel --version``     (see :mod:`synthpanel.__main__`)
* ``import synthpanel`` / ``from synthpanel import sdk``
* ``synthpanel.__version__``

Implementation note: we use the ``__path__`` redirect trick — setting
``synthpanel.__path__`` to point at ``synth_panel.__path__`` makes Python's
regular ``PathFinder`` resolve ``synthpanel.X`` by looking inside
``synth_panel/X.py`` exactly the way it would for ``synth_panel.X``. The
result is one canonical module object per file, no duplicate instances,
no custom loader / metapath finder gymnastics.

Closes GH #509.
"""

from __future__ import annotations

import sys as _sys

import synth_panel as _synth_panel
from synth_panel import __version__

# The __path__ redirect: when Python resolves ``import synthpanel.X``, it
# searches every entry in ``synthpanel.__path__``. Pointing that list at
# ``synth_panel.__path__`` makes Python find the same source files the
# canonical ``synth_panel.X`` import would find — so the resulting module
# objects ARE the same object, just registered under both names in
# ``sys.modules``. No custom loader needed.
__path__ = list(_synth_panel.__path__)

# Mirror modules that are already loaded under the canonical name so they
# show up under the alias too. Without this, code that does
# ``import synth_panel.poll_summary`` followed by ``import synthpanel.poll_summary``
# would otherwise mint a second module instance for the alias, because
# Python's import machinery sees no entry under the alias and re-executes
# the source file. ``setdefault`` keeps anything the user already wired up.
for _name, _module in list(_sys.modules.items()):
    if _name == "synth_panel" or _name.startswith("synth_panel."):
        _aname = "synthpanel" + _name[len("synth_panel") :]
        _sys.modules.setdefault(_aname, _module)

# Re-export the canonical public surface so ``from synthpanel import X`` works
# even when ``X`` isn't a submodule (e.g. ``run_panel`` is a function exported
# from ``synth_panel.__init__``).
__all__ = list(getattr(_synth_panel, "__all__", []))
for _attr in __all__:
    globals()[_attr] = getattr(_synth_panel, _attr)

__version__ = __version__
