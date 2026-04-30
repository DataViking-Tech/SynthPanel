"""On-disk schema-version migrations for PanelCheckpoint records.

When a checkpoint was written by an older synthpanel, this module
brings it up to CURRENT_SCHEMA_VERSION so the rest of the codebase
can assume the latest shape.

Version history
---------------
v1 — synthpanel 0.11.x initial checkpointing (sp-hsk3); no cli_args field.
v2 — synthpanel 0.12.x; cli_args added (sy-ws76); best_model_for flag added.
     Absent or legacy ``version: 1`` key → treated as v1.
"""

from __future__ import annotations

from typing import Any

CURRENT_SCHEMA_VERSION: int = 2


def migrate_v1_to_v2(old: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v1 checkpoint (synthpanel 0.11.x) to v2 schema.

    v1 checkpoints predate the cli_args field (added in sy-ws76 for bare
    ``--resume <id>`` support). Missing optional fields receive safe defaults
    so ``PanelCheckpoint.from_dict`` always gets a complete record.
    """
    result = dict(old)
    result.setdefault("cli_args", None)
    result.setdefault("completed", [])
    result.setdefault("remaining", [])
    result.setdefault("usage", {})
    result.setdefault("abort_reason", None)
    result["schema_version"] = 2
    return result


# Each entry: (from_version, migration_fn).  Apply all entries where
# from_version >= the checkpoint's current version.
_CHAIN: list[tuple[int, Any]] = [
    (1, migrate_v1_to_v2),
]


def migrate_to_current(data: dict[str, Any]) -> dict[str, Any]:
    """Return *data* migrated to :data:`CURRENT_SCHEMA_VERSION`.

    Reads ``schema_version`` (absent, or legacy ``version`` key) — treats
    missing / zero as v1.  Applies all pending migrations in order and
    returns the updated dict.  Returns *data* unchanged when already at
    the current version.

    Raises ``ValueError`` if ``schema_version`` exceeds
    :data:`CURRENT_SCHEMA_VERSION` — caller should convert to an
    appropriate domain error.
    """
    version = int(data.get("schema_version") or data.get("version") or 1)
    if version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version {version} is newer than this synthpanel "
            f"installation (max supported: v{CURRENT_SCHEMA_VERSION}). "
            f"Upgrade synthpanel to resume this run."
        )
    if version == CURRENT_SCHEMA_VERSION:
        return data
    result = dict(data)
    for from_ver, migration_fn in _CHAIN:
        if version <= from_ver:
            result = migration_fn(result)
    return result
