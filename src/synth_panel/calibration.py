"""Pack-YAML calibration block read/merge/write (sp-sghl).

Helpers used by ``synthpanel pack calibrate`` to splice a freshly-computed
calibration entry into a persona pack YAML without disturbing the
surrounding persona definitions, comments, or whitespace.

The strategy is *targeted text surgery* rather than a full YAML
round-trip:

1. Parse the YAML once with :mod:`pyyaml` to validate it and obtain the
   current top-level structure (so we can reason about whether a
   ``calibration:`` block already exists).
2. Locate the existing top-level ``calibration:`` region in the raw
   text — if any — and replace it. If absent, append a new block at the
   end of the file.
3. The replacement block is rendered with ``yaml.safe_dump`` so the
   serialized fields are well-formed; only the calibration region is
   touched.

This keeps comments and persona definitions outside the calibration
block exactly as the user wrote them.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import yaml


@dataclass
class CalibrationEntry:
    """One calibration result against a single SynthBench baseline.

    A pack's ``calibration:`` field is a list of these. Re-running
    calibration against the same ``(dataset, question)`` pair replaces
    the prior entry rather than appending a duplicate (newest wins).
    """

    dataset: str
    question: str
    jsd: float
    n: int
    samples_per_question: int
    models: list[str]
    extractor: str
    panelist_cost_usd: float
    calibrated_at: str
    synthpanel_version: str
    methodology_url: str = "https://synthpanel.dev/docs/calibration"
    alignment_error: str | None = field(default=None)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Render as an ordered plain dict suitable for YAML emission."""
        d = asdict(self)
        # Drop None alignment_error so the wire shape matches the bead's
        # example for the happy path.
        if d.get("alignment_error") is None:
            d.pop("alignment_error", None)
        return d


def now_iso_utc() -> str:
    """RFC3339 UTC timestamp with second precision (no microseconds)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pack_yaml(path: str, *, min_personas: int = 1) -> tuple[str, dict[str, Any]]:
    """Read a pack YAML and return ``(raw_text, parsed_dict)``.

    Raises ``ValueError`` on parse failure, non-mapping top level, or when
    the pack contains fewer than ``min_personas`` persona entries.
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"pack YAML at {path} must be a mapping at top level")
    personas = data.get("personas")
    if not isinstance(personas, list) or len(personas) < min_personas:
        raise ValueError(
            f"no personas found in {path}; calibration requires at least {min_personas} persona(s)"
        )
    return raw, data


def merge_calibration(
    existing: list[dict[str, Any]] | None,
    new_entry: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge ``new_entry`` into ``existing`` calibration list.

    Replaces any prior entry with the same ``(dataset, question)`` pair
    so re-running calibration is idempotent — newest wins.
    """
    if existing is None:
        existing = []
    if not isinstance(existing, list):
        raise ValueError(f"pack 'calibration:' must be a list of calibration entries; got {type(existing).__name__}")
    target_dataset = new_entry.get("dataset")
    target_question = new_entry.get("question")
    out: list[dict[str, Any]] = []
    replaced = False
    for entry in existing:
        if not isinstance(entry, dict):
            out.append(entry)
            continue
        if entry.get("dataset") == target_dataset and entry.get("question") == target_question:
            if not replaced:
                out.append(new_entry)
                replaced = True
            continue
        out.append(entry)
    if not replaced:
        out.append(new_entry)
    return out


def _render_calibration_block(entries: list[dict[str, Any]]) -> str:
    """Render the ``calibration:`` block as YAML text.

    The output always ends with a single trailing newline. ``sort_keys``
    is disabled so field order matches the dataclass declaration.
    """
    rendered = yaml.safe_dump(
        {"calibration": entries},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered


_TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*:")


def _is_top_level_key_line(line: str) -> bool:
    """True when *line* begins (column 0) with a ``key:`` declaration.

    Excludes block list items (``- foo``), document separators (``---``),
    and continuation lines indented under a parent key.
    """
    return bool(_TOP_LEVEL_KEY_RE.match(line))


def _find_top_level_block(raw: str, key: str) -> tuple[int, int] | None:
    """Locate the byte span of an existing top-level ``key:`` block.

    Returns ``(start, end)`` where ``raw[start:end]`` covers the whole
    block including the trailing newline of its last line, or ``None``
    if no top-level ``key:`` is present.

    "Top level" means the line begins with ``key:`` at column 0. The
    block extends until the next top-level key or EOF. List items
    (``- ...``) rendered at column 0 by ``yaml.safe_dump`` are treated as
    continuations of the prior key, not as new top-level boundaries.
    """
    lines = raw.splitlines(keepends=True)
    start_line: int | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#") or not stripped.strip():
            continue
        if _is_top_level_key_line(line) and stripped.startswith(f"{key}:"):
            start_line = i
            break
    if start_line is None:
        return None

    end_line = len(lines)
    for j in range(start_line + 1, len(lines)):
        line = lines[j]
        stripped = line.lstrip()
        if not stripped.strip() or stripped.startswith("#"):
            continue
        if _is_top_level_key_line(line):
            # Another top-level key starts here.
            end_line = j
            break

    start = sum(len(line) for line in lines[:start_line])
    end = sum(len(line) for line in lines[:end_line])
    return start, end


def write_pack_calibration(
    raw_text: str,
    entries: list[dict[str, Any]],
) -> str:
    """Return ``raw_text`` with the ``calibration:`` block replaced/appended.

    Persona definitions and any other top-level keys outside the
    ``calibration:`` block are preserved verbatim.
    """
    new_block = _render_calibration_block(entries)
    span = _find_top_level_block(raw_text, "calibration")
    if span is None:
        sep = "" if raw_text.endswith("\n") or raw_text == "" else "\n"
        # Always separate from prior content with one blank line.
        prefix = raw_text + sep
        if prefix and not prefix.endswith("\n\n"):
            prefix += "\n"
        return prefix + new_block
    start, end = span
    return raw_text[:start] + new_block + raw_text[end:]


def update_pack_calibration_text(
    raw_text: str,
    parsed: dict[str, Any],
    new_entry: dict[str, Any],
) -> str:
    """High-level helper: merge ``new_entry`` into the pack and re-emit.

    Validates that the parsed pack's ``calibration`` field (if present)
    is a list, replaces any matching entry, and rewrites the YAML.
    """
    existing = parsed.get("calibration")
    merged = merge_calibration(existing if isinstance(existing, list) else None, new_entry)
    return write_pack_calibration(raw_text, merged)
