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
import urllib.parse
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


def list_persona_packs(*, warn_on_shadow: bool = False) -> list[dict[str, Any]]:
    """Return metadata for all persona packs (bundled + user-saved).

    Bundled packs are listed first. If a user-saved pack has the same ID as
    a bundled pack, the user-saved version takes precedence.

    ``warn_on_shadow``: when True, emit a ``UserWarning`` for every bundled
    pack hidden by a user-saved pack of the same id. This is the
    registry-import path; local callers leave it False to preserve the
    historical silent-shadow behavior.
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
            if warn_on_shadow:
                import warnings

                warnings.warn(
                    f"Bundled persona pack {pack_id!r} is shadowed by a user-saved pack",
                    UserWarning,
                    stacklevel=2,
                )
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
    """Load a persona pack by ID. User-saved packs override bundled ones.

    Pack-level manifest fields are normalized via
    :func:`validate_pack_manifest` — in particular ``version`` defaults to
    ``"1"`` when absent.
    """
    _validate_pack_id(pack_id)
    # Check user-saved packs first
    p = _packs_dir() / f"{pack_id}.yaml"
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid persona pack format in {pack_id}")
        data = validate_pack_manifest(data)
        data["id"] = pack_id
        return data

    # Fall back to bundled packs
    bundled = _bundled_packs()
    if pack_id in bundled:
        data = validate_pack_manifest(bundled[pack_id])
        data["id"] = pack_id
        return data

    raise FileNotFoundError(f"Persona pack not found: {pack_id}")


class PackValidationError(ValueError):
    """Raised when a persona pack fails schema validation."""


def validate_pack_manifest(data: dict[str, Any]) -> dict[str, Any]:
    """Validate pack-level manifest fields and apply defaults.

    Currently enforces one rule: the optional ``version`` field must be a
    string. When absent, it defaults to ``"1"``. Other manifest fields
    (``name``, ``description``, ``author``) are passed through unchanged.

    Returns a shallow copy of *data* with defaults applied. Raises
    :class:`PackValidationError` for type violations.
    """
    if not isinstance(data, dict):
        raise PackValidationError("pack manifest must be a mapping")
    out = dict(data)
    version = out.get("version")
    if version is None:
        out["version"] = "1"
    elif not isinstance(version, str):
        raise PackValidationError(f"pack version must be a string, got {type(version).__name__}")
    return out


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
    version: str | None = None,
) -> dict[str, Any]:
    """Save a persona pack and return its metadata.

    Validates personas before saving. Raises :class:`PackValidationError`
    on invalid data.

    When ``version`` is provided it must be a string and is written to the
    on-disk manifest. When omitted, no ``version:`` field is written — the
    pack loads as ``"1"`` via :func:`get_persona_pack`'s default.
    """
    personas = validate_persona_pack(personas)
    pid = pack_id or f"pack-{uuid.uuid4().hex[:8]}"
    _validate_pack_id(pid)
    p = _packs_dir() / f"{pid}.yaml"
    data: dict[str, Any] = {"name": name}
    if version is not None:
        validated = validate_pack_manifest({"version": version})
        data["version"] = validated["version"]
    data["personas"] = personas
    p.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    meta: dict[str, Any] = {
        "id": pid,
        "name": name,
        "persona_count": len(personas),
        "path": str(p),
    }
    if version is not None:
        meta["version"] = data["version"]
    return meta


def uninstall_persona_pack(pack_id: str) -> None:
    """Delete a user-saved persona pack from the local registry.

    Raises :class:`ValueError` when *pack_id* names a bundled pack (those
    cannot be removed).  Raises :class:`FileNotFoundError` when the pack is
    not found in the user-saved store.
    """
    _validate_pack_id(pack_id)
    if pack_id in _bundled_packs():
        raise ValueError(f"'{pack_id}' is a bundled pack and cannot be uninstalled")
    p = _packs_dir() / f"{pack_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Persona pack not found: {pack_id}")
    p.unlink()


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
        # URL-encode persona name for a lossless, filesystem-safe filename
        safe_name = urllib.parse.quote(persona_name, safe="")
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
        persona_name = urllib.parse.unquote(p.stem)
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
        # `<id>.attachments/refs.json` is a sidecar index, not a result.
        # The non-recursive glob above already excludes it, but a defensive
        # check protects callers who might pass a recursive pattern in.
        if p.parent.name.endswith(".attachments"):
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            entry: dict[str, Any] = {
                "id": p.stem,
                "created_at": data.get("created_at", ""),
                "model": data.get("model", ""),
                "persona_count": data.get("persona_count", 0),
                "question_count": data.get("question_count", 0),
            }
            vc = data.get("variant_count", 0)
            if vc:
                entry["variant_count"] = vc
            if "instrument_name" in data:
                entry["instrument_name"] = data["instrument_name"]
            if "models" in data:
                entry["models"] = data["models"]
            results.append(entry)
        except Exception:
            continue
    return results


def get_panel_result(result_id: str, *, load_attachments: bool = False) -> dict[str, Any]:
    """Load a panel result by ID.

    When *load_attachments* is True and a ``<result_id>.attachments/refs.json``
    sidecar exists, the parsed refs map is attached as
    ``data["attachments"]`` and ``data["_attachments_loaded"]`` is set to
    True. Default behavior is identical to pre-attachments code paths:
    no extra fields, no extra I/O. Existing consumers
    (``cost_summary.py``, ``analysis/inspect.py``) read the result via
    the default path and stay unchanged.
    """
    _validate_pack_id(result_id)
    p = _results_dir() / f"{result_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Panel result not found: {result_id}")
    data = json.loads(p.read_text(encoding="utf-8"))
    data["id"] = result_id
    if load_attachments:
        from synth_panel.attachments.models import AttachmentRef
        from synth_panel.attachments.store import refs_path

        rp = refs_path(_results_dir(), result_id)
        if rp.exists():
            try:
                refs_raw = json.loads(rp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                refs_raw = None
            if isinstance(refs_raw, dict):
                # v1.0.4: validate each entry through AttachmentRef so
                # drift across versions surfaces as a field-path-aware
                # ValidationError instead of dict-key-missing downstream.
                # Returned shape stays dict-of-dict for back-compat with
                # existing consumers (analysis/inspect.py et al.).
                validated: dict[str, dict[str, Any]] = {}
                for ref_id, raw in refs_raw.items():
                    ref = AttachmentRef.model_validate(raw)
                    validated[ref_id] = ref.model_dump(mode="json", exclude_none=True)
                data["attachments"] = validated
                data["_attachments_loaded"] = True
            else:
                data["_attachments_loaded"] = False
        else:
            data["_attachments_loaded"] = False
    return data


def save_panel_synthesis(
    source_result_id: str,
    timestamp: str,
    payload: dict[str, Any],
) -> str:
    """Write a sidecar synthesis file next to a saved panel result.

    Used by ``synthpanel panel synthesize`` (sp-5on.5) to persist a
    re-synthesis without mutating the original result. Returns the
    sidecar filename (not the full path).
    """
    _validate_pack_id(source_result_id)
    name = f"{source_result_id}.synthesis-{timestamp}.json"
    p = _results_dir() / name
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return name


def save_panel_result(
    results: list[dict[str, Any]],
    model: str,
    total_usage: dict[str, Any],
    total_cost: str,
    persona_count: int,
    question_count: int,
    variant_count: int = 0,
    *,
    instrument_name: str | None = None,
    questions: list[dict[str, Any]] | None = None,
    variants_config: dict[str, Any] | None = None,
    models: list[str] | None = None,
    attachments: dict[str, dict[str, Any]] | None = None,
    synthesis: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    decision_being_informed: str | None = None,
) -> str:
    """Save panel results and return the result ID.

    New optional fields (backward-compatible — omitted when *None*):

    * ``decision_being_informed``: the v1.0.0 contract field this panel
      run informs (real caller-supplied value, or the AC-4 grace
      placeholder ``"unspecified-legacy-call"``). Persisted at the top
      level so a saved result can be joined back to the decision it
      answered without re-reading the transcript.
    * ``instrument_name``: name/id of the instrument pack used.
    * ``questions``: question defs with ``text`` and optional
      ``extraction_schema``.
    * ``variants_config``: variant generation config (``n``, ``seed``).
    * ``models``: list of all model identifiers used in the run.
    * ``metadata``: the run provenance bundle produced by
      :func:`synth_panel.metadata.build_metadata` (version, config_hash,
      cost/pricing snapshot, timing, models). Persisted at the top level
      so ``synthpanel report`` can populate its provenance table for
      saved results instead of degrading every field to ``(unknown)``
      (#525). Omitting this field is the legacy shape.
    * ``attachments``: ``{ref_id: AttachmentRef}`` map written to a
      ``<result_id>.attachments/refs.json`` sidecar. When non-empty,
      ``result_format_version`` bumps from ``"1.0"`` to ``"1.1"`` so
      readback layers can branch on the schema version. Bytes always
      live in CAS (see :mod:`synth_panel.attachments.store`); only
      refs land here.
    * ``synthesis``: serialized ``SynthesisResult.to_dict()`` payload
      (summary/themes/agreements/disagreements/surprises/recommendation
      plus usage/cost/model). Persisted at the top level so
      :func:`get_panel_result` and :mod:`analysis.inspect` see it
      transparently. Omitting this field is the legacy shape; callers
      that ran synthesis should always thread it through.

    Per-result entries in *results* may contain ``_variant_of`` and
    ``_model`` fields; per-response dicts may contain an ``extraction``
    dict. These are passed through as-is.
    """
    rid = f"result-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    p = _results_dir() / f"{rid}.json"
    has_attachments = bool(attachments)
    data: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result_format_version": "1.1" if has_attachments else "1.0",
        "model": model,
        "persona_count": persona_count,
        "question_count": question_count,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "results": results,
    }
    if decision_being_informed is not None:
        data["decision_being_informed"] = decision_being_informed
    if variant_count > 0:
        data["variant_count"] = variant_count
    if instrument_name is not None:
        data["instrument_name"] = instrument_name
    if questions is not None:
        data["questions"] = questions
    if variants_config is not None:
        data["variants_config"] = variants_config
    if models is not None:
        data["models"] = models
    if synthesis is not None:
        data["synthesis"] = synthesis
    if metadata is not None:
        data["metadata"] = metadata
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    if has_attachments:
        from synth_panel.attachments.models import AttachmentRef
        from synth_panel.attachments.store import refs_path

        # v1.0.4: validate each entry through AttachmentRef so malformed
        # refs fail at write time (named field path) rather than later
        # at readback. The serialized shape — model_dump(mode="json",
        # exclude_none=True) — matches the v1.0.3 wire format byte-for-byte
        # for refs the SDK emits today.
        serialized: dict[str, dict[str, Any]] = {}
        assert attachments is not None  # narrowed by has_attachments
        for ref_id, raw in attachments.items():
            ref = raw if isinstance(raw, AttachmentRef) else AttachmentRef.model_validate(raw)
            serialized[ref_id] = ref.model_dump(mode="json", exclude_none=True)

        rp = refs_path(_results_dir(), rid)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")

    return rid
