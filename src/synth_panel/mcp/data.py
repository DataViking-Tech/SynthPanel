"""Persistence layer for MCP persona packs and panel results.

Data is stored under ~/.synth-panel/ (configurable via SYNTH_PANEL_DATA_DIR).

Layout::

    $SYNTH_PANEL_DATA_DIR/
      persona_packs/
        <pack_id>.yaml
      results/
        <result_id>.json
"""

from __future__ import annotations

import importlib.resources
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _data_dir() -> Path:
    """Return the root data directory, creating it if needed."""
    d = Path(os.environ.get("SYNTH_PANEL_DATA_DIR", "~/.synth-panel")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _packs_dir() -> Path:
    d = _data_dir() / "persona_packs"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        packs_pkg = importlib.resources.files("synth_panel.packs")
        for item in packs_pkg.iterdir():
            if item.name.endswith(".yaml"):
                try:
                    data = yaml.safe_load(item.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        result[item.name.removesuffix(".yaml")] = data
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
            packs.append({
                "id": pack_id,
                "name": data.get("name", pack_id) if isinstance(data, dict) else pack_id,
                "persona_count": len(personas),
                "path": str(p),
                "builtin": False,
            })
        except Exception:
            continue

    # Bundled packs (only if not overridden by user)
    bundled = []
    for pack_id, data in sorted(_bundled_packs().items()):
        if pack_id in seen:
            continue
        personas = data.get("personas", [])
        bundled.append({
            "id": pack_id,
            "name": data.get("name", pack_id),
            "persona_count": len(personas),
            "builtin": True,
        })

    return bundled + packs


def get_persona_pack(pack_id: str) -> dict[str, Any]:
    """Load a persona pack by ID. User-saved packs override bundled ones."""
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
            raise PackValidationError(
                f"persona at index {i} is missing required field 'name'"
            )

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
    p = _packs_dir() / f"{pid}.yaml"
    data = {"name": name, "personas": personas}
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return {"id": pid, "name": name, "persona_count": len(personas), "path": str(p)}


# ---------------------------------------------------------------------------
# Panel results
# ---------------------------------------------------------------------------

def list_panel_results() -> list[dict[str, Any]]:
    """Return metadata for all saved panel results."""
    results: list[dict[str, Any]] = []
    for p in sorted(_results_dir().glob("*.json"), reverse=True):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            results.append({
                "id": p.stem,
                "created_at": data.get("created_at", ""),
                "model": data.get("model", ""),
                "persona_count": data.get("persona_count", 0),
                "question_count": data.get("question_count", 0),
            })
        except Exception:
            continue
    return results


def get_panel_result(result_id: str) -> dict[str, Any]:
    """Load a panel result by ID."""
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
