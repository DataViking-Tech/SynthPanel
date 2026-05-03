"""SynthPanel JSON Schema assets — v1.0.0 frozen contract.

Schemas live as JSON files inside this package (``v{version}.json``) and
are loaded via :func:`load`. The returned mapping is recursively wrapped
in :class:`types.MappingProxyType` so callers cannot mutate the embedded
schema in-process — the contract is frozen at the file level *and* at the
in-memory level.

New contract versions ship as parallel files (``v1.1.0.json``,
``v2.0.0.json``), never as in-place edits to a published version.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources
from types import MappingProxyType
from typing import Any

_SUPPORTED_VERSIONS = frozenset({"1.0.0"})


def load(version: str = "1.0.0") -> Mapping[str, Any]:
    """Load the JSON Schema for the given contract version.

    Parameters
    ----------
    version:
        Contract version string (e.g. ``"1.0.0"``). Must match a shipped
        schema file in this package.

    Returns
    -------
    Mapping[str, Any]
        An immutable mapping over the parsed schema. Lists are returned as
        tuples; nested dicts are wrapped in ``MappingProxyType``.
    """
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(f"Unknown schema version {version!r}; supported: {sorted(_SUPPORTED_VERSIONS)}")
    raw = resources.files(__package__).joinpath(f"v{version}.json").read_text("utf-8")
    return _freeze(json.loads(raw))


def _freeze(obj: Any) -> Any:
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(x) for x in obj)
    return obj


__all__ = ["load"]
