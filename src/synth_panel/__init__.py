"""Deprecated compatibility shim — re-exports the ``althing`` package.

SynthPanel was renamed to **Althing** (see the 1.0 rename announcement at
https://althing.dev/rename). The canonical import is now::

    import althing

This module keeps the historical ``synth_panel`` import working for one
deprecation cycle. It uses the ``__path__`` redirect trick — pointing
``synthpanel.__path__`` at ``althing.__path__`` makes Python's regular
``PathFinder`` resolve ``synthpanel.X`` by looking inside ``althing/X.py``,
producing one canonical module object per file (no duplicate instances, no
custom loader).

A ``DeprecationWarning`` is emitted on import; the shim will be removed in a
future major release.
"""

from __future__ import annotations

import sys as _sys
import warnings as _warnings

import althing as _althing
from althing import __version__

_warnings.warn(
    "'synth_panel' has been renamed to 'althing' — update imports to "
    "'import althing' and install with 'pip install althing'. "
    "The 'synth_panel' alias will be removed in a future major release.",
    DeprecationWarning,
    stacklevel=2,
)

# Own directory first so the shim's __main__.py (rename notice) wins;
# everything else falls through to the canonical source tree.
__path__ = [str(__import__("pathlib").Path(__file__).parent), *_althing.__path__]

# Mirror modules already loaded under the canonical name so they show up
# under the alias too (see docstring; keeps module identity single).
for _name, _module in list(_sys.modules.items()):
    if _name == "althing" or _name.startswith("althing."):
        _aname = "synth_panel" + _name[len("althing") :]
        _sys.modules.setdefault(_aname, _module)

__all__ = list(getattr(_althing, "__all__", []))
for _attr in __all__:
    globals()[_attr] = getattr(_althing, _attr)

__version__ = __version__


def __getattr__(name: str):  # PEP 562 — delegate everything else to althing
    """Resolve attributes (incl. loaded submodules) from the canonical package.

    The ``sys.modules`` mirror above registers alias entries but cannot bind
    parent-package attributes the way a real import does, so ``synth_panel.sdk``
    would raise ``AttributeError`` without this hook.
    """
    return getattr(_althing, name)
