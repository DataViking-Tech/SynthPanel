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
# Persona packs
# ---------------------------------------------------------------------------

def list_persona_packs() -> list[dict[str, Any]]:
    """Return metadata for all saved persona packs."""
    packs: list[dict[str, Any]] = []
    for p in sorted(_packs_dir().glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            personas = data.get("personas", []) if isinstance(data, dict) else []
            packs.append({
                "id": p.stem,
                "name": data.get("name", p.stem) if isinstance(data, dict) else p.stem,
                "persona_count": len(personas),
                "path": str(p),
            })
        except Exception:
            continue
    return packs


def get_persona_pack(pack_id: str) -> dict[str, Any]:
    """Load a persona pack by ID."""
    p = _packs_dir() / f"{pack_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Persona pack not found: {pack_id}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid persona pack format in {pack_id}")
    data["id"] = pack_id
    return data


def save_persona_pack(
    name: str,
    personas: list[dict[str, Any]],
    pack_id: str | None = None,
) -> dict[str, Any]:
    """Save a persona pack and return its metadata."""
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
