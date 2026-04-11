"""Bundled extraction schemas for structured data extraction.

Each schema is a JSON Schema dict suitable for use with
:class:`~synth_panel.structured.output.StructuredOutputConfig`.

The registry supports lookup by name (``get_schema``) and
enumeration (``list_schemas``).  Unknown names raise
:class:`SchemaNotFoundError` at load time so instruments fail
fast rather than silently skipping extraction.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Bundled schemas
# ---------------------------------------------------------------------------

RANKING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "rank": {"type": "integer"},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "rank"],
            },
        },
    },
    "required": ["ranked"],
}

LIKERT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rating": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["rating"],
}

YES_NO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "boolean"},
        "reasoning": {"type": "string"},
    },
    "required": ["answer"],
}

PICK_ONE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["choice"],
}

_REGISTRY: dict[str, dict[str, Any]] = {
    "ranking": RANKING_SCHEMA,
    "likert": LIKERT_SCHEMA,
    "yes_no": YES_NO_SCHEMA,
    "pick_one": PICK_ONE_SCHEMA,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class SchemaNotFoundError(ValueError):
    """Raised when a schema name is not in the registry."""

    def __init__(self, name: str) -> None:
        known = ", ".join(sorted(_REGISTRY))
        super().__init__(f"Unknown extraction schema: {name!r} (known: {known})")
        self.name = name


def get_schema(name: str) -> dict[str, Any]:
    """Return a bundled extraction schema by name.

    Raises :class:`SchemaNotFoundError` if *name* is not registered.
    """
    if name not in _REGISTRY:
        raise SchemaNotFoundError(name)
    return _REGISTRY[name]


def list_schemas() -> list[dict[str, Any]]:
    """Return metadata for all registered extraction schemas."""
    return [{"name": name, "schema": schema} for name, schema in sorted(_REGISTRY.items())]


def is_known_schema(name: str) -> bool:
    """Return True if *name* is a registered extraction schema."""
    return name in _REGISTRY
