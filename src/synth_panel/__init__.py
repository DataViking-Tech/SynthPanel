from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("synthpanel")
except Exception:
    __version__ = "0.0.0-dev"
