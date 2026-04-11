"""Model alias resolution (SPEC.md §2 — Provider Resolution).

Resolution order (highest priority wins):
1. SYNTHPANEL_MODEL_ALIASES env var (JSON string of alias→model pairs)
2. ~/.synthpanel/aliases.yaml file
3. Hardcoded defaults below
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

# Short alias → canonical model identifier (tier 3: hardcoded fallback).
_HARDCODED_ALIASES: dict[str, str] = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "grok": "grok-3",
    "gemini": "gemini-2.5-flash",
    "gemini-pro": "gemini-2.5-pro",
}

# Local model prefixes: prefix → default base URL (without /v1 suffix).
_LOCAL_PREFIXES: dict[str, str] = {
    "ollama:": "http://localhost:11434",
    "local:": "http://localhost:1234",
}

_ALIASES_FILE = Path.home() / ".synthpanel" / "aliases.yaml"
_ENV_VAR = "SYNTHPANEL_MODEL_ALIASES"

# Cached merged result; reset with _reset_cache() for testing.
_cached_aliases: dict[str, str] | None = None


def _load_file_aliases() -> dict[str, str]:
    """Load aliases from ~/.synthpanel/aliases.yaml (tier 2).

    Expected format::

        aliases:
          fast: claude-haiku-4-5-20251001
          smart: claude-opus-4-6
    """
    if not _ALIASES_FILE.is_file():
        return {}
    try:
        raw: Any = yaml.safe_load(_ALIASES_FILE.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(raw, dict):
        return {}
    aliases = raw.get("aliases", raw)
    if not isinstance(aliases, dict):
        return {}
    return {str(k): str(v) for k, v in aliases.items()}


def _load_env_aliases() -> dict[str, str]:
    """Load aliases from SYNTHPANEL_MODEL_ALIASES env var (tier 1, JSON)."""
    raw = os.environ.get(_ENV_VAR, "")
    if not raw:
        return {}
    try:
        parsed: Any = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items()}


def _build_aliases() -> dict[str, str]:
    """Merge aliases: hardcoded → file → env (env wins on conflicts)."""
    merged = dict(_HARDCODED_ALIASES)
    merged.update(_load_file_aliases())
    merged.update(_load_env_aliases())
    return merged


def _get_aliases() -> dict[str, str]:
    """Return the merged alias map, caching after first call."""
    global _cached_aliases
    if _cached_aliases is None:
        _cached_aliases = _build_aliases()
    return _cached_aliases


def _reset_cache() -> None:
    """Clear the cached alias map (for testing)."""
    global _cached_aliases
    _cached_aliases = None


# Public name kept for backward compat — now dynamically built.
MODEL_ALIASES = _HARDCODED_ALIASES


def resolve_alias(model: str) -> str:
    """Return the canonical model name, resolving short aliases and local prefixes."""
    for prefix in _LOCAL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    aliases = _get_aliases()
    return aliases.get(model, model)


def get_base_url_override(model: str) -> str | None:
    """Return the local base URL if *model* uses a local prefix, else None.

    ``ollama:llama3`` → ``"http://localhost:11434"``
    ``local:phi3``    → ``"http://localhost:1234"``
    """
    for prefix, url in _LOCAL_PREFIXES.items():
        if model.startswith(prefix):
            return url
    return None
