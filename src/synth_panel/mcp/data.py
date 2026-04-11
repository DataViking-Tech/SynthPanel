"""Persistence layer for MCP persona packs and panel results.

Data is stored under ~/.synthpanel/ (configurable via SYNTH_PANEL_DATA_DIR).

Layout::

    $SYNTH_PANEL_DATA_DIR/
      persona_packs/
        <pack_id>.yaml
      results/
        <result_id>.json
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synth_panel.persistence import Session

from importlib.resources import files as _resource_files

import yaml


def _validate_pack_id(pack_id: str) -> None:
    """Reject pack IDs that could escape the data directory."""
    if "/" in pack_id or ".." in pack_id:
        raise ValueError(f"Invalid pack ID (path traversal characters not allowed): {pack_id!r}")


def _data_dir() -> Path:
    """Return the root data directory, creating it if needed."""
    d = Path(os.environ.get("SYNTH_PANEL_DATA_DIR", "~/.synthpanel")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _packs_dir() -> Path:
    d = _data_dir() / "persona_packs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _instrument_packs_dir() -> Path:
    d = _data_dir() / "packs" / "instruments"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Manifest fields shared across pack types (per F2-B spec).
_MANIFEST_FIELDS = ("name", "version", "description", "author")


def _extract_manifest(data: dict[str, Any], pack_id: str) -> dict[str, Any]:
    """Pull the four shared manifest fields out of a pack dict."""
    return {
        "id": pack_id,
        "name": data.get("name", pack_id),
        "version": data.get("version", ""),
        "description": data.get("description", ""),
        "author": data.get("author", ""),
    }


def _results_dir() -> Path:
    d = _data_dir() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Bundled packs (shipped with the package)
# ---------------------------------------------------------------------------


def _bundled_packs() -> dict[str, dict[str, Any]]:
    """Load persona packs bundled in synth_panel.packs.

    Returns a dict mapping pack_id (filename stem) to parsed YAML data.
    """
    result: dict[str, dict[str, Any]] = {}
    try:
        packs_pkg = _resource_files("synth_panel.packs")
        for item in packs_pkg.iterdir():
            if item.name.endswith(".yaml"):
                try:
                    data = yaml.safe_load(item.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        pack_id = item.name[: -len(".yaml")]
                        result[pack_id] = data
                except Exception:
                    continue
    except Exception:
        pass
    return result


def _bundled_instrument_packs() -> dict[str, dict[str, Any]]:
    """Load instrument packs bundled in synth_panel.packs.instruments.

    Returns a dict mapping pack_id (filename stem) to parsed YAML data.
    """
    result: dict[str, dict[str, Any]] = {}
    try:
        pkg = _resource_files("synth_panel.packs.instruments")
        for item in pkg.iterdir():
            if item.name.endswith(".yaml"):
                try:
                    data = yaml.safe_load(item.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        pack_id = item.name[: -len(".yaml")]
                        result[pack_id] = data
                except Exception:
                    continue
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Persona packs
# ---------------------------------------------------------------------------


def list_persona_packs() -> list[dict[str, Any]]:
    """Return metadata for all persona packs (bundled + user-saved).

    Bundled packs are listed first. If a user-saved pack has the same ID as
    a bundled pack, the user-saved version takes precedence.
    """
    seen: set[str] = set()
    packs: list[dict[str, Any]] = []

    # User-saved packs (take precedence)
    for p in sorted(_packs_dir().glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            personas = data.get("personas", []) if isinstance(data, dict) else []
            pack_id = p.stem
            seen.add(pack_id)
            packs.append(
                {
                    "id": pack_id,
                    "name": data.get("name", pack_id) if isinstance(data, dict) else pack_id,
                    "persona_count": len(personas),
                    "path": str(p),
                    "builtin": False,
                }
            )
        except Exception:
            continue

    # Bundled packs (only if not overridden by user)
    bundled = []
    for pack_id, data in sorted(_bundled_packs().items()):
        if pack_id in seen:
            continue
        personas = data.get("personas", [])
        bundled.append(
            {
                "id": pack_id,
                "name": data.get("name", pack_id),
                "persona_count": len(personas),
                "builtin": True,
            }
        )

    return bundled + packs


def get_persona_pack(pack_id: str) -> dict[str, Any]:
    """Load a persona pack by ID. User-saved packs override bundled ones."""
    _validate_pack_id(pack_id)
    # Check user-saved packs first
    p = _packs_dir() / f"{pack_id}.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid persona pack format in {pack_id}")
        data["id"] = pack_id
        return data

    # Fall back to bundled packs
    bundled = _bundled_packs()
    if pack_id in bundled:
        data = bundled[pack_id]
        data["id"] = pack_id
        return data

    raise FileNotFoundError(f"Persona pack not found: {pack_id}")


class PackValidationError(ValueError):
    """Raised when a persona pack fails schema validation."""


def validate_persona_pack(personas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize a list of persona dicts.

    Required fields per persona: ``name``.
    ``personality_traits`` is normalized to a list of lowercase stripped strings.

    Returns the normalized personas list.
    Raises :class:`PackValidationError` on invalid data.
    """
    if not isinstance(personas, list):
        raise PackValidationError("personas must be a list")
    if not personas:
        raise PackValidationError("personas list must not be empty")

    normalized: list[dict[str, Any]] = []
    for i, persona in enumerate(personas):
        if not isinstance(persona, dict):
            raise PackValidationError(f"persona at index {i} must be a dict")
        if "name" not in persona or not str(persona["name"]).strip():
            raise PackValidationError(f"persona at index {i} is missing required field 'name'")

        p = dict(persona)  # shallow copy to avoid mutating input

        # Normalize personality_traits
        traits = p.get("personality_traits")
        if traits is not None:
            if isinstance(traits, str):
                traits = [t.strip().lower() for t in traits.split(",") if t.strip()]
            elif isinstance(traits, list):
                traits = [str(t).strip().lower() for t in traits if str(t).strip()]
            else:
                raise PackValidationError(
                    f"persona '{p['name']}': personality_traits must be a list or comma-separated string"
                )
            p["personality_traits"] = traits

        normalized.append(p)
    return normalized


def save_persona_pack(
    name: str,
    personas: list[dict[str, Any]],
    pack_id: str | None = None,
) -> dict[str, Any]:
    """Save a persona pack and return its metadata.

    Validates personas before saving. Raises :class:`PackValidationError`
    on invalid data.
    """
    personas = validate_persona_pack(personas)
    pid = pack_id or f"pack-{uuid.uuid4().hex[:8]}"
    _validate_pack_id(pid)
    p = _packs_dir() / f"{pid}.yaml"
    data = {"name": name, "personas": personas}
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return {"id": pid, "name": name, "persona_count": len(personas), "path": str(p)}


# ---------------------------------------------------------------------------
# Instrument packs (single-file YAML, manifest at top level)
# ---------------------------------------------------------------------------


def list_instrument_packs() -> list[dict[str, Any]]:
    """Return manifest metadata for every available instrument pack.

    Includes both bundled packs (shipped under
    ``synth_panel.packs.instruments``) and user-saved packs under
    ``$SYNTH_PANEL_DATA_DIR/packs/instruments/``. User-saved packs take
    precedence over bundled packs of the same id.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    # User-saved packs first (take precedence over bundled).
    for p in sorted(_instrument_packs_dir().glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        meta = _extract_manifest(data, p.stem)
        meta["path"] = str(p)
        meta["type"] = "instrument"
        meta["source"] = "user"
        out.append(meta)
        seen.add(p.stem)

    # Bundled packs (only those not shadowed by a user-saved pack).
    for pack_id, data in sorted(_bundled_instrument_packs().items()):
        if pack_id in seen:
            continue
        meta = _extract_manifest(data, pack_id)
        meta["path"] = f"bundled:{pack_id}"
        meta["type"] = "instrument"
        meta["source"] = "bundled"
        out.append(meta)
    return out


def load_instrument_pack(name: str) -> dict[str, Any]:
    """Load an instrument pack by name. Returns the full YAML body.

    User-saved packs take precedence over bundled packs of the same id.
    """
    _validate_pack_id(name)
    p = _instrument_packs_dir() / f"{name}.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid instrument pack format in {name}")
        data["id"] = name
        return data

    # Fall back to bundled packs.
    bundled = _bundled_instrument_packs()
    if name in bundled:
        data = dict(bundled[name])
        data["id"] = name
        return data

    raise FileNotFoundError(f"Instrument pack not found: {name}")


def save_instrument_pack(name: str, content: dict[str, Any]) -> dict[str, Any]:
    """Save an instrument pack to disk and return its manifest metadata.

    ``content`` is the full YAML body — the manifest fields are
    expected to live at the top level alongside the instrument
    definition. The caller is responsible for parser-level validation.
    """
    _validate_pack_id(name)
    if not isinstance(content, dict):
        raise ValueError("instrument pack content must be a mapping")
    body = dict(content)
    # Ensure the manifest 'name' matches the pack id on disk.
    body.setdefault("name", name)
    p = _instrument_packs_dir() / f"{name}.yaml"
    p.write_text(yaml.dump(body, default_flow_style=False, sort_keys=False), encoding="utf-8")
    meta = _extract_manifest(body, name)
    meta["path"] = str(p)
    meta["type"] = "instrument"
    return meta


# ---------------------------------------------------------------------------
# Panel results
# ---------------------------------------------------------------------------


def _sessions_dir(result_id: str) -> Path:
    """Return the sessions directory for a given result, creating it if needed."""
    d = _results_dir() / f"{result_id}.sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_panel_sessions(
    result_id: str,
    sessions: dict[str, Session],
) -> Path:
    """Save per-panelist sessions to disk.

    Each session is stored as ``<PersonaName>.json`` under
    ``results/<result_id>.sessions/``.

    Returns the sessions directory path.
    """

    _validate_pack_id(result_id)
    sdir = _sessions_dir(result_id)
    for persona_name, session in sessions.items():
        # Sanitise persona name for use as filename
        safe_name = persona_name.replace("/", "_").replace("\\", "_")
        p = sdir / f"{safe_name}.json"
        p.write_text(json.dumps(session.to_dict(), indent=2) + "\n", encoding="utf-8")
    return sdir


def load_panel_sessions(result_id: str) -> dict[str, Session]:
    """Load per-panelist sessions from disk.

    Returns a dict mapping persona name to :class:`Session`.
    Raises :class:`FileNotFoundError` if the sessions directory doesn't exist.
    """
    from synth_panel.persistence import Session

    _validate_pack_id(result_id)
    sdir = _results_dir() / f"{result_id}.sessions"
    if not sdir.exists():
        raise FileNotFoundError(f"No sessions found for result: {result_id}")

    sessions: dict[str, Session] = {}
    for p in sorted(sdir.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        persona_name = p.stem.replace("_", " ")  # best-effort reverse of sanitisation
        # Prefer the original persona name if we can recover it from session messages
        sessions[persona_name] = Session.from_dict(data)
    return sessions


def update_panel_result(result_id: str, updated_data: dict[str, Any]) -> None:
    """Update a panel result, saving a pre-extend snapshot first.

    Creates ``<result_id>.pre-extend.json`` as a backup before overwriting
    the main result file.
    """
    _validate_pack_id(result_id)
    result_path = _results_dir() / f"{result_id}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Panel result not found: {result_id}")

    # Save pre-extend snapshot (overwrite any previous snapshot)
    snapshot_path = _results_dir() / f"{result_id}.pre-extend.json"
    snapshot_path.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Overwrite main result
    result_path.write_text(json.dumps(updated_data, indent=2) + "\n", encoding="utf-8")


def list_panel_results() -> list[dict[str, Any]]:
    """Return metadata for all saved panel results."""
    results: list[dict[str, Any]] = []
    for p in sorted(_results_dir().glob("*.json"), reverse=True):
        if p.name.endswith(".pre-extend.json"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append(
                {
                    "id": p.stem,
                    "created_at": data.get("created_at", ""),
                    "model": data.get("model", ""),
                    "persona_count": data.get("persona_count", 0),
                    "question_count": data.get("question_count", 0),
                }
            )
        except Exception:
            continue
    return results


def get_panel_result(result_id: str) -> dict[str, Any]:
    """Load a panel result by ID."""
    _validate_pack_id(result_id)
    p = _results_dir() / f"{result_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Panel result not found: {result_id}")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["id"] = result_id
    return data


def save_panel_result(
    results: list[dict[str, Any]],
    model: str,
    total_usage: dict[str, Any],
    total_cost: str,
    persona_count: int,
    question_count: int,
) -> str:
    """Save panel results and return the result ID."""
    rid = f"result-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    p = _results_dir() / f"{rid}.json"
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "results": results,
    }
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return rid
