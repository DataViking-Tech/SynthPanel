"""Diff two persona packs side-by-side (GH-308).

Compares packs by persona name (added / removed / unchanged / changed) and
summarizes composition deltas (age range, role distribution, gender split
when present). Loads either built-in / user-saved pack IDs or file paths.

Name-based matching is intentionally simple — renames will surface as
``added`` + ``removed`` rather than a single change. The README documents
this so users can interpret results correctly.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from synth_panel.mcp.data import get_persona_pack

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

# Persona-level fields the diff inspects when both packs contain a persona of
# the same name. Adding new fields here is the supported way to broaden the
# field-level diff without touching the matching logic.
_FIELD_KEYS: tuple[str, ...] = (
    "age",
    "occupation",
    "background",
    "personality_traits",
    "gender",
)

_SENIORITY_PREFIXES = {"junior", "senior", "staff", "principal", "lead", "chief"}


@dataclass
class CompositionStats:
    """Aggregate demographic snapshot for a single pack.

    ``gender_split`` is empty when no persona declares a ``gender`` field
    (the common case for the bundled packs). ``role_distribution`` is keyed
    on a one-word bucket derived from each persona's ``occupation``.
    """

    persona_count: int
    age_min: int | None
    age_max: int | None
    age_mean: float | None
    gender_split: dict[str, int] = field(default_factory=dict)
    role_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class PersonaChange:
    """Field-level diff for a persona present in both packs.

    ``changed`` maps each differing field to ``{"a": <a-value>, "b": <b-value>}``.
    For ``personality_traits`` the values are the raw lists; the renderer
    surfaces added/removed traits separately for readability.
    """

    name: str
    changed: dict[str, dict[str, Any]]


@dataclass
class PackDiff:
    pack_a_id: str
    pack_b_id: str
    pack_a_name: str
    pack_b_name: str
    composition_a: CompositionStats
    composition_b: CompositionStats
    added: list[str]
    removed: list[str]
    unchanged: list[str]
    changed: list[PersonaChange]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_pack(source: str) -> tuple[dict[str, Any], str]:
    """Load a persona pack from a file path or built-in / user-saved pack ID.

    File paths win when *source* points to an existing file — this lets
    callers diff unpublished packs without first installing them. Otherwise
    the loader falls back to :func:`synth_panel.mcp.data.get_persona_pack`,
    which checks user-saved packs first and then bundled packs.

    Returns ``(pack_dict, source_label)`` — the label is the file stem for
    paths and the original ID for installed packs, used for human-readable
    output.
    """
    p = Path(source)
    if p.is_file():
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Pack file does not contain a mapping: {source}")
        return data, p.stem
    return get_persona_pack(source), source


# ---------------------------------------------------------------------------
# Composition stats
# ---------------------------------------------------------------------------


def _normalize_traits(traits: Any) -> list[str]:
    """Collapse traits to a sorted list of stripped lowercase strings."""
    if traits is None:
        return []
    if isinstance(traits, str):
        items = [t.strip().lower() for t in traits.split(",") if t.strip()]
    elif isinstance(traits, list):
        items = [str(t).strip().lower() for t in traits if str(t).strip()]
    else:
        return []
    return sorted(items)


def _role_bucket(occupation: str) -> str:
    """Coarse single-word bucket for an occupation string.

    Drops seniority prefixes (Junior, Senior, Staff, Principal, Lead, Chief)
    and returns the first remaining alphabetical token, lowercased. Falls
    back to the raw stripped string when no tokens match.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]+", occupation)
    if not tokens:
        return occupation.strip().lower()
    cleaned = [t.lower() for t in tokens if t.lower() not in _SENIORITY_PREFIXES]
    if not cleaned:
        cleaned = [t.lower() for t in tokens]
    return cleaned[0]


def _compose(personas: list[dict[str, Any]]) -> CompositionStats:
    ages = [int(p["age"]) for p in personas if isinstance(p.get("age"), int)]
    age_min = min(ages) if ages else None
    age_max = max(ages) if ages else None
    age_mean = round(sum(ages) / len(ages), 1) if ages else None

    gender_counter: Counter[str] = Counter()
    for p in personas:
        g = p.get("gender")
        if g:
            gender_counter[str(g).lower().strip()] += 1

    role_counter: Counter[str] = Counter()
    for p in personas:
        occ = p.get("occupation")
        if occ:
            role_counter[_role_bucket(str(occ))] += 1

    return CompositionStats(
        persona_count=len(personas),
        age_min=age_min,
        age_max=age_max,
        age_mean=age_mean,
        gender_split=dict(gender_counter),
        role_distribution=dict(role_counter),
    )


# ---------------------------------------------------------------------------
# Field-level diff
# ---------------------------------------------------------------------------


def _persona_field_diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the subset of ``_FIELD_KEYS`` that differ between two personas.

    ``personality_traits`` is compared after normalization (sorted, lowercase,
    stripped) so cosmetic ordering or casing changes don't trip the diff.
    Other fields are compared with raw equality and ``None == None`` is not
    flagged.
    """
    changed: dict[str, dict[str, Any]] = {}
    for key in _FIELD_KEYS:
        av = a.get(key)
        bv = b.get(key)
        if key == "personality_traits":
            if _normalize_traits(av) != _normalize_traits(bv):
                changed[key] = {"a": av, "b": bv}
        else:
            if av is None and bv is None:
                continue
            if av != bv:
                changed[key] = {"a": av, "b": bv}
    return changed


def trait_delta(change: PersonaChange) -> tuple[list[str], list[str]]:
    """Return ``(added, removed)`` traits for a persona change, or empty lists."""
    payload = change.changed.get("personality_traits")
    if not payload:
        return [], []
    a = set(_normalize_traits(payload.get("a")))
    b = set(_normalize_traits(payload.get("b")))
    return sorted(b - a), sorted(a - b)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compute_pack_diff(
    pack_a: dict[str, Any],
    pack_b: dict[str, Any],
    *,
    pack_a_id: str = "",
    pack_b_id: str = "",
) -> PackDiff:
    """Compute a side-by-side diff of two persona packs.

    Personas are matched by ``name`` (exact). Renames appear as one entry in
    ``added`` and one in ``removed``. ``unchanged`` lists matched personas
    whose tracked fields are all equal; ``changed`` lists matched personas
    with at least one differing field.
    """
    personas_a = pack_a.get("personas") or []
    personas_b = pack_b.get("personas") or []
    if not isinstance(personas_a, list):
        personas_a = []
    if not isinstance(personas_b, list):
        personas_b = []

    by_name_a = {p["name"]: p for p in personas_a if isinstance(p, dict) and p.get("name")}
    by_name_b = {p["name"]: p for p in personas_b if isinstance(p, dict) and p.get("name")}

    names_a = set(by_name_a)
    names_b = set(by_name_b)

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)
    common = sorted(names_a & names_b)

    unchanged: list[str] = []
    changed: list[PersonaChange] = []
    for name in common:
        diff = _persona_field_diff(by_name_a[name], by_name_b[name])
        if diff:
            changed.append(PersonaChange(name=name, changed=diff))
        else:
            unchanged.append(name)

    a_label = pack_a_id or str(pack_a.get("id") or "") or "pack_a"
    b_label = pack_b_id or str(pack_b.get("id") or "") or "pack_b"

    return PackDiff(
        pack_a_id=a_label,
        pack_b_id=b_label,
        pack_a_name=str(pack_a.get("name") or a_label),
        pack_b_name=str(pack_b.get("name") or b_label),
        composition_a=_compose([p for p in personas_a if isinstance(p, dict)]),
        composition_b=_compose([p for p in personas_b if isinstance(p, dict)]),
        added=added,
        removed=removed,
        unchanged=unchanged,
        changed=changed,
    )
