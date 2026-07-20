"""SynthPanel → Althing rename (1.0): legacy environment-variable bridge.

Imported for its side effect as the first statement in ``althing.__init__``:
honors legacy ``SYNTHPANEL_*`` environment variables for one deprecation
cycle by mirroring them into their ``ALTHING_*`` equivalents when the new
name is unset. Central bridge so every downstream ``os.environ`` reader
inherits the fallback.
"""

from __future__ import annotations

import os as _os
import warnings as _warnings

_legacy = [k for k in _os.environ if k.startswith("SYNTHPANEL_")]
for _k in _legacy:
    _new = "ALTHING_" + _k[len("SYNTHPANEL_") :]
    _os.environ.setdefault(_new, _os.environ[_k])
if _legacy:
    _warnings.warn(
        f"SYNTHPANEL_* environment variables are deprecated after the rename "
        f"to althing — set {', '.join(sorted(('ALTHING_' + k[len('SYNTHPANEL_'):]) for k in _legacy))} "
        f"instead. Legacy names will stop working in a future major release.",
        DeprecationWarning,
        stacklevel=2,
    )
del _legacy
